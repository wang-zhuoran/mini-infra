"""
ppo_rlhf.py — PPO for RLHF (LLM Alignment)

Concept: Use RL to align LLM with human preferences using 4 models.

4 Models:
  Actor     (π_θ)  — the LLM being trained
  Critic    (V_φ)  — estimates state value, used only for advantage computation
  Reference (π_ref)— frozen SFT model, defines KL baseline
  Reward    (r_ψ)  — frozen, gives scalar score for a full response

Training loop per batch:
  1. Rollout  : Actor generates responses → get token log-probs
  2. Score    : Reward model scores responses; Reference computes KL at each token
  3. GAE      : Critic estimates V(s_t); compute advantages via GAE-λ
  4. PPO update (K epochs on same rollout data):
               Actor  ← clipped surrogate loss
               Critic ← MSE loss on value targets

Interview questions this file answers:
  Q: Why clip the ratio? What does ε=0.2 mean geometrically?
  Q: Why KL penalty AND clipping — aren't they redundant?
  Q: Why freeze the reference model?
  Q: GAE-λ: what's the bias-variance trade-off vs Monte Carlo?
  Q: Why token-level policy gradient, not sequence-level?
  Q: Why multiple PPO epochs on the same rollout data?

Run: python ppo_rlhf.py  (single GPU, small toy model)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# ── Tiny LM for demo (replace with real model in practice) ───────────────────
VOCAB   = 64
HIDDEN  = 128
HEADS   = 4
LAYERS  = 2
SEQ_LEN = 16   # response length

class TinyLM(nn.Module):
    """Minimal decoder-only transformer. Real: GPT-2 / Qwen / LLaMA."""
    def __init__(self):
        super().__init__()
        self.embed  = nn.Embedding(VOCAB, HIDDEN)
        layer       = nn.TransformerEncoderLayer(HIDDEN, HEADS, HIDDEN * 4,
                                                  batch_first=True, norm_first=True)
        self.tfm    = nn.TransformerEncoder(layer, LAYERS)
        self.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)

    def forward(self, x):
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)
        h = self.tfm(self.embed(x), mask=mask, is_causal=True)
        return self.lm_head(h)   # [B, T, VOCAB]

class ValueHead(nn.Module):
    """Critic head on top of LM hidden states."""
    def __init__(self):
        super().__init__()
        self.embed   = nn.Embedding(VOCAB, HIDDEN)
        layer        = nn.TransformerEncoderLayer(HIDDEN, HEADS, HIDDEN * 4,
                                                   batch_first=True, norm_first=True)
        self.tfm     = nn.TransformerEncoder(layer, LAYERS)
        self.v_head  = nn.Linear(HIDDEN, 1)

    def forward(self, x):
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)
        h = self.tfm(self.embed(x), mask=mask, is_causal=True)
        return self.v_head(h).squeeze(-1)   # [B, T]

class RewardModel(nn.Module):
    """Gives a scalar reward for a full sequence. Frozen during PPO."""
    def __init__(self):
        super().__init__()
        self.embed  = nn.Embedding(VOCAB, HIDDEN)
        self.pool   = nn.Linear(HIDDEN, 1)

    def forward(self, x):
        h = self.embed(x).mean(dim=1)   # mean-pool over tokens
        return self.pool(h).squeeze(-1) # [B]

# ── Setup ─────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

actor   = TinyLM().to(device)
critic  = ValueHead().to(device)
ref     = TinyLM().to(device)          # frozen copy of initial actor
reward  = RewardModel().to(device)     # frozen

# Copy actor → ref, then freeze both ref and reward
ref.load_state_dict(actor.state_dict())
for m in [ref, reward]:
    for p in m.parameters():
        p.requires_grad_(False)

opt_actor  = torch.optim.Adam(actor.parameters(),  lr=3e-5)
opt_critic = torch.optim.Adam(critic.parameters(), lr=1e-4)

# ── Hyperparameters ───────────────────────────────────────────────────────────
B          = 4      # batch size
CLIP_EPS   = 0.2   # PPO clip range: ratio ∈ [1-ε, 1+ε]
KL_BETA    = 0.02  # KL penalty weight (β in InstructGPT)
GAMMA      = 0.99  # discount factor
LAM        = 0.95  # GAE-λ (λ=1 → MC returns; λ=0 → 1-step TD)
PPO_EPOCHS = 4     # reuse each rollout this many times

# ── Step 1: Rollout ───────────────────────────────────────────────────────────
# Actor autoregressively generates SEQ_LEN tokens.
# We record: sampled token ids + their log-probs under the actor.
# In real RLHF: prompt prepended, only response tokens trained on.

def rollout(actor, B, seq_len):
    tokens    = torch.zeros(B, 1, dtype=torch.long, device=device)  # BOS
    log_probs = []   # log π_actor(a_t | s_t) at each step

    actor.eval()
    with torch.no_grad():
        for _ in range(seq_len):
            logits  = actor(tokens)[:, -1, :]       # [B, VOCAB]
            dist    = Categorical(logits=logits)
            a       = dist.sample()                  # [B]
            lp      = dist.log_prob(a)               # [B]
            tokens  = torch.cat([tokens, a.unsqueeze(1)], dim=1)
            log_probs.append(lp)

    actor.train()
    tokens    = tokens[:, 1:]                        # remove BOS → [B, T]
    log_probs = torch.stack(log_probs, dim=1)        # [B, T]
    return tokens, log_probs

tokens, old_log_probs = rollout(actor, B, SEQ_LEN)   # log_probs are π_old

# ── Step 2: Score — reward + token-level KL penalty ─────────────────────────
# Final reward at last token: R_T = reward_model(full_response)
# KL penalty at each token:   r_t += -β * (log π_actor - log π_ref)
#
# Why KL penalty? Clipping alone constrains step size in parameter space,
# but KL keeps the policy semantically close to the SFT model.
# Without it, the actor collapses to reward-hacking behaviors.

with torch.no_grad():
    scalar_reward = reward(tokens)                    # [B]  sequence-level score

    ref_logits    = ref(tokens)                       # [B, T, VOCAB]
    actor_logits  = actor(tokens).detach()
    ref_lp        = F.log_softmax(ref_logits,   dim=-1)
    actor_lp      = F.log_softmax(actor_logits, dim=-1)

    # Token-level KL: KL(π_actor || π_ref) = Σ_a π_actor * (log π_actor - log π_ref)
    # Approximate with sampled token: kl_t ≈ log π_actor(a_t) - log π_ref(a_t)
    token_kl = (actor_lp - ref_lp).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)  # [B, T]

    # Token-level rewards: KL penalty everywhere + scalar reward at last step
    token_rewards             = -KL_BETA * token_kl                          # [B, T]
    token_rewards[:, -1]     += scalar_reward                                # add R_T at end

# ── Step 3: GAE — compute advantages ─────────────────────────────────────────
# δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
# A_t = δ_t + γλ * δ_{t+1} + (γλ)² * δ_{t+2} + ...
#
# λ=1 → full MC return (high variance, unbiased)
# λ=0 → 1-step TD (low variance, biased by critic error)
# λ=0.95 → good empirical balance

with torch.no_grad():
    values   = critic(tokens)                          # [B, T]
    values_t = torch.cat([values, torch.zeros(B, 1, device=device)], dim=1)  # pad V(s_{T+1})=0

    advantages = torch.zeros_like(token_rewards)
    gae = 0.0
    for t in reversed(range(SEQ_LEN)):
        delta      = token_rewards[:, t] + GAMMA * values_t[:, t+1] - values_t[:, t]
        gae        = delta + GAMMA * LAM * gae
        advantages[:, t] = gae

    returns    = advantages + values                   # value targets for critic
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # whiten

# ── Step 4: PPO update (K epochs) ────────────────────────────────────────────
# Reusing the same rollout data K times is what makes it "proximal":
# we stay close to π_old via the clip, so multiple gradient steps are safe.

for epoch in range(PPO_EPOCHS):
    # ── Actor loss: clipped surrogate ─────────────────────────────────────
    # r_t(θ) = π_θ(a_t|s_t) / π_old(a_t|s_t)  ← probability ratio
    # L = E[ min(r_t * A_t,  clip(r_t, 1-ε, 1+ε) * A_t) ]
    #
    # Why min? When A>0, we want r↑ but cap at 1+ε (don't over-exploit).
    #           When A<0, we want r↓ but cap at 1-ε (don't over-penalize).
    # The clip removes incentive to move ratio outside [1-ε, 1+ε].

    logits       = actor(tokens)                       # [B, T, VOCAB]
    log_probs    = F.log_softmax(logits, dim=-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)

    ratio        = torch.exp(log_probs - old_log_probs.detach())  # [B, T]
    clipped      = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
    actor_loss   = -torch.min(ratio * advantages, clipped * advantages).mean()

    opt_actor.zero_grad()
    actor_loss.backward()
    nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
    opt_actor.step()

    # ── Critic loss: MSE on value targets ─────────────────────────────────
    v_pred       = critic(tokens)                      # [B, T]
    critic_loss  = F.mse_loss(v_pred, returns.detach())

    opt_critic.zero_grad()
    critic_loss.backward()
    opt_critic.step()

    # Approx KL for early stopping (optional, used by OpenAI)
    with torch.no_grad():
        approx_kl = (old_log_probs - log_probs).mean().item()

    print(f"  epoch {epoch+1}/{PPO_EPOCHS}  "
          f"actor={actor_loss.item():.4f}  critic={critic_loss.item():.4f}  "
          f"approx_kl={approx_kl:.4f}")

# ── Summary of key numbers ────────────────────────────────────────────────────
print(f"\nConfig: B={B}, T={SEQ_LEN}, ε={CLIP_EPS}, β(KL)={KL_BETA}, λ(GAE)={LAM}")
print(f"Mean reward     : {scalar_reward.mean().item():.3f}")
print(f"Mean token KL   : {token_kl.mean().item():.4f}")
print(f"Mean |advantage|: {advantages.abs().mean().item():.4f}")
print()
print("Memory layout (real training, 7B model):")
print("  Actor (trainable) : ~14 GB params + ~56 GB grad + optimizer = ~112 GB")
print("  Critic (trainable): same architecture → another ~112 GB")
print("  Reference (frozen): ~14 GB  (no grad)")
print("  Reward (frozen)   : ~14 GB  (no grad, often smaller model)")
print("  → Total naive: ~252 GB  → need ZeRO-3 + offload, or separate actor/critic GPU pools")
print()
print("Why GRPO eliminates Critic:")
print("  Critic is expensive (same size as actor) and introduces bias.")
print("  GRPO replaces V(s) with group mean reward → no separate model needed.")
print("  Trade-off: higher variance (no baseline quality), but much simpler pipeline.")
