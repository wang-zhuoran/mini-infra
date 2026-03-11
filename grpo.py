"""
grpo.py — Group Relative Policy Optimization (DeepSeek-R1 / DeepSeekMath)

Concept: Replace PPO's Critic with a group baseline.
  For each prompt, sample G responses → normalize rewards within the group.
  No separate value network needed.

PPO  vs GRPO:
  PPO : A_t  = GAE(r_t, V(s_t))       ← needs Critic (same size as Actor)
  GRPO: A_i  = (R_i - mean_j R_j)     ← group mean is the baseline
               ────────────────
               std_j R_j + ε

Why this matters in practice (your resume):
  3B VLM Actor + 3B Critic = 6B params to train → doubles memory & comm.
  GRPO: only 3B Actor + 3B Reference (frozen) → cuts trainable params in half.
  On 8×RTX 4090 (24 GB each), this is the difference between fitting or not.

Interview questions this file answers:
  Q: How does GRPO compute advantages without a Critic?
  Q: Why does group size G affect training stability?
  Q: How is GRPO's KL penalty different from PPO's?
  Q: What's the variance of the group-mean estimator? When does G=8 suffice?
  Q: Why can rule-based rewards replace a reward model here?

Run: python grpo.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# ── Tiny LM (same as ppo_rlhf.py) ────────────────────────────────────────────
VOCAB   = 64
HIDDEN  = 128
HEADS   = 4
LAYERS  = 2
SEQ_LEN = 16

class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed   = nn.Embedding(VOCAB, HIDDEN)
        layer        = nn.TransformerEncoderLayer(HIDDEN, HEADS, HIDDEN * 4,
                                                   batch_first=True, norm_first=True)
        self.tfm     = nn.TransformerEncoder(layer, LAYERS)
        self.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)

    def forward(self, x):
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)
        h = self.tfm(self.embed(x), mask=mask, is_causal=True)
        return self.lm_head(h)   # [B, T, VOCAB]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

actor = TinyLM().to(device)
ref   = TinyLM().to(device)   # frozen SFT reference — the only "second model"

ref.load_state_dict(actor.state_dict())
for p in ref.parameters():
    p.requires_grad_(False)

optimizer = torch.optim.Adam(actor.parameters(), lr=1e-5)

# ── Hyperparameters ───────────────────────────────────────────────────────────
N_PROMPTS  = 4      # number of distinct prompts per batch
G          = 8      # group size: responses sampled per prompt
CLIP_EPS   = 0.2
KL_BETA    = 0.01   # typically smaller than PPO (GRPO already regularizes via grouping)
PPO_EPOCHS = 1      # GRPO usually 1 epoch per rollout (on-policy)

# ── Step 1: Rollout — sample G responses per prompt ──────────────────────────
# Shape convention: [N_PROMPTS * G, T] — flatten group dim for batched compute.
# In real code (veRL / OpenRLHF): vLLM generates all G*N sequences in one call.

def rollout_group(actor, n_prompts, G, seq_len):
    """
    Returns:
      tokens    : [N*G, T]   sampled token ids
      log_probs : [N*G, T]   log π_actor(a_t | s_t) under OLD policy
    """
    total  = n_prompts * G
    tokens = torch.zeros(total, 1, dtype=torch.long, device=device)   # BOS
    log_probs_list = []

    actor.eval()
    with torch.no_grad():
        for _ in range(seq_len):
            logits = actor(tokens)[:, -1, :]           # [N*G, VOCAB]
            dist   = Categorical(logits=logits)
            a      = dist.sample()                     # [N*G]
            log_probs_list.append(dist.log_prob(a))    # [N*G]
            tokens = torch.cat([tokens, a.unsqueeze(1)], dim=1)

    actor.train()
    tokens    = tokens[:, 1:]                          # remove BOS → [N*G, T]
    log_probs = torch.stack(log_probs_list, dim=1)     # [N*G, T]
    return tokens, log_probs

tokens, old_log_probs = rollout_group(actor, N_PROMPTS, G, SEQ_LEN)
# tokens: [N*G, T],  grouped as [prompt0_resp0..7, prompt1_resp0..7, ...]

# ── Step 2: Reward — rule-based or reward model ───────────────────────────────
# In your GUI agent: reward = coord_accuracy + action_type_match + format_check
# No LLM judge needed → deterministic, fast, no reward model memory cost.
# Here: dummy scalar reward per response.

def rule_based_reward(tokens):
    """Simulate a deterministic rule-based reward. Replace with your actual rules."""
    return tokens.float().mean(dim=-1) / VOCAB   # [N*G], range ~[0,1]

rewards = rule_based_reward(tokens)   # [N*G]

# ── Step 3: Group-normalized advantage ───────────────────────────────────────
#
#  Reshape rewards to [N, G], compute mean/std within each group of G responses.
#
#  A_i = (R_i - mean_j R_j) / (std_j R_j + ε)
#
#  This is the baseline: instead of V(s) predicted by a Critic, we use the
#  empirical mean of G sampled returns. With G→∞ this converges to E[R|prompt].
#
#  Variance of estimator: Var(mean) = σ²/G → larger G → lower variance baseline
#  In practice G=8~16 is good enough; G=1 degenerates (std=0, advantage=0).

rewards_grouped = rewards.view(N_PROMPTS, G)           # [N, G]
mean_r = rewards_grouped.mean(dim=1, keepdim=True)     # [N, 1]
std_r  = rewards_grouped.std(dim=1, keepdim=True)      # [N, 1]

advantages_grouped = (rewards_grouped - mean_r) / (std_r + 1e-8)   # [N, G]
advantages = advantages_grouped.view(N_PROMPTS * G)                  # [N*G]

# Advantage is sequence-level (one scalar per response, not per token).
# Broadcast to all tokens: A_t = A_i for all t in response i.
advantages_token = advantages.unsqueeze(1).expand_as(old_log_probs)  # [N*G, T]

print(f"Rewards  — mean: {rewards.mean():.3f}  std: {rewards.std():.3f}")
print(f"Advantage— mean: {advantages.mean():.3f}  std: {advantages.std():.3f}")
print(f"  (std≈1.0 is expected after group normalization)")

# ── Step 4: PPO-style clipped loss + KL penalty ───────────────────────────────
#
# L_GRPO = -E[ min(r_t·A_i, clip(r_t, 1-ε, 1+ε)·A_i) ] + β·KL(π_θ ‖ π_ref)
#
# Identical clipping to PPO — only the advantage source differs.

for epoch in range(PPO_EPOCHS):
    logits    = actor(tokens)                          # [N*G, T, VOCAB]
    log_probs = F.log_softmax(logits, dim=-1)\
                  .gather(-1, tokens.unsqueeze(-1)).squeeze(-1)   # [N*G, T]

    # Policy ratio
    ratio   = torch.exp(log_probs - old_log_probs.detach())       # [N*G, T]
    clipped = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
    policy_loss = -torch.min(ratio * advantages_token,
                             clipped * advantages_token).mean()

    # KL penalty (token-level, same as PPO-RLHF)
    with torch.no_grad():
        ref_log_probs = F.log_softmax(ref(tokens), dim=-1)\
                          .gather(-1, tokens.unsqueeze(-1)).squeeze(-1)  # [N*G, T]
    kl_loss = (log_probs - ref_log_probs.detach()).mean()   # ≈ KL(π_θ ‖ π_ref)

    loss = policy_loss + KL_BETA * kl_loss

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
    optimizer.step()

    print(f"\nepoch {epoch+1}: policy={policy_loss.item():.4f}  "
          f"kl={kl_loss.item():.4f}  total={loss.item():.4f}")

# ── PPO vs GRPO side-by-side ──────────────────────────────────────────────────
print("""
┌─────────────────┬──────────────────────────────┬────────────────────────────┐
│                 │ PPO                          │ GRPO                       │
├─────────────────┼──────────────────────────────┼────────────────────────────┤
│ Baseline        │ Critic V(s_t)  [trainable]   │ Group mean reward          │
│ Advantage       │ GAE-λ  (token-level)         │ (R_i - μ_G) / σ_G         │
│ Models          │ Actor + Critic + Ref + RM    │ Actor + Ref  (+ RM opt.)   │
│ Memory          │ ~2× actor params trainable   │ ~1× actor params trainable │
│ Reward signal   │ Reward model (LLM-based ok)  │ Rule-based preferred       │
│ Sample eff.     │ Multiple epochs (K=4)        │ Usually 1 epoch (on-policy)│
│ Variance        │ Lower (critic reduces var.)  │ Higher, decreases with G   │
└─────────────────┴──────────────────────────────┴────────────────────────────┘

Your project used G=8 on 8×RTX4090. Key questions to prepare:
  1. Why G=8?  → Var(mean) = σ²/G; G=8 balances variance vs VRAM for rollout
  2. Why ZeRO-3? → 3B actor params × (2+2+8) bytes = 48GB, needs sharding across 8 GPUs
  3. Why rule-based reward? → No reward model → saves 3B model VRAM + removes reward hacking
  4. Why 1 PPO epoch? → GRPO is strictly on-policy; stale advantages hurt more than in PPO
""")
