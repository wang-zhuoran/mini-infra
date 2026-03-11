"""
dpo.py — Direct Preference Optimization

Concept: Skip the reward model entirely. Directly optimize the policy from
         (prompt, chosen, rejected) preference pairs.

Derivation in 3 steps:
  1. RL objective:  max_π  E[r(x,y)] - β·KL(π ‖ π_ref)
     Optimal policy: π*(y|x) ∝ π_ref(y|x)·exp(r(x,y)/β)
     → r(x,y) = β·log[π*(y|x)/π_ref(y|x)] + β·log Z(x)

  2. Bradley-Terry preference model:
     p(y_w ≻ y_l | x) = σ(r(x,y_w) - r(x,y_l))

  3. Substitute step 1 into step 2:
     The β·log Z(x) terms CANCEL (same prompt x for both)
     → p(y_w ≻ y_l | x) = σ(β·[log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)])

  Loss = -E[ log σ(β · (logr_w - logr_l)) ]
         where logr = log π_θ(y|x) - log π_ref(y|x)  ← implicit reward

Key insight: Z(x) cancels because chosen and rejected share the same prompt x.
            This is WHY DPO works — the intractable partition function disappears.

PPO vs DPO:
  PPO : online  (generate new responses each step) → fresher distribution, costly
  DPO : offline (train on fixed preference dataset) → simple, but distribution shift

Interview questions this file answers:
  Q: Derive the DPO loss from first principles.
  Q: Why does the partition function Z(x) cancel?
  Q: What is the implicit reward in DPO?
  Q: When does DPO fail compared to PPO?
  Q: What is β and what happens at β→0 and β→∞?

Run: python dpo.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Tiny LM ───────────────────────────────────────────────────────────────────
VOCAB  = 64
HIDDEN = 128
HEADS  = 4
LAYERS = 2

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
ref   = TinyLM().to(device)
ref.load_state_dict(actor.state_dict())
for p in ref.parameters():
    p.requires_grad_(False)

optimizer = torch.optim.Adam(actor.parameters(), lr=1e-5)

# ── Hyperparameters ───────────────────────────────────────────────────────────
B    = 4      # batch size (preference pairs)
T    = 16     # sequence length
BETA = 0.1    # KL regularization strength
              # β→0: ignore KL, policy collapses to maximize any signal
              # β→∞: policy stays at π_ref, learns nothing
              # β=0.1~0.5: typical range in practice

# ── Fake preference dataset ───────────────────────────────────────────────────
# Real data: (prompt, chosen_response, rejected_response) triplets
# e.g. Anthropic HH-RLHF, UltraFeedback, etc.
torch.manual_seed(0)
prompt   = torch.randint(0, VOCAB, (B, 8),  device=device)   # [B, T_prompt]
chosen   = torch.randint(0, VOCAB, (B, T),  device=device)   # [B, T] preferred response
rejected = torch.randint(0, VOCAB, (B, T),  device=device)   # [B, T] dispreferred response

# Concatenate prompt + response (standard practice: train on response tokens only)
chosen_seq   = torch.cat([prompt, chosen],   dim=1)   # [B, T_prompt + T]
rejected_seq = torch.cat([prompt, rejected], dim=1)

# ── Core: compute sequence log-prob ──────────────────────────────────────────
def sequence_log_prob(model, seq):
    """
    Compute sum of log-probs over response tokens only.
    In real DPO: mask out prompt tokens, only sum over response positions.
    Returns: [B]  — one scalar per sequence
    """
    logits    = model(seq[:, :-1])                                # [B, T-1, VOCAB]
    log_probs = F.log_softmax(logits, dim=-1)                     # [B, T-1, VOCAB]
    # Gather log-prob of actual next token
    token_lp  = log_probs.gather(-1, seq[:, 1:].unsqueeze(-1)).squeeze(-1)  # [B, T-1]
    # Sum over response tokens (here: all tokens; real impl masks prompt)
    return token_lp.sum(dim=-1)   # [B]

# ── DPO Training loop ─────────────────────────────────────────────────────────
for step in range(5):

    # log π_θ(y_w|x) and log π_θ(y_l|x)
    lp_chosen_actor    = sequence_log_prob(actor, chosen_seq)     # [B]
    lp_rejected_actor  = sequence_log_prob(actor, rejected_seq)   # [B]

    # log π_ref(y_w|x) and log π_ref(y_l|x)  — no grad
    with torch.no_grad():
        lp_chosen_ref   = sequence_log_prob(ref, chosen_seq)      # [B]
        lp_rejected_ref = sequence_log_prob(ref, rejected_seq)    # [B]

    # Implicit reward = β · log(π_θ / π_ref)
    # r(x, y_w) - r(x, y_l) = β · [(lp_chosen_actor - lp_chosen_ref)
    #                               - (lp_rejected_actor - lp_rejected_ref)]
    logr_chosen   = lp_chosen_actor   - lp_chosen_ref    # [B]  log ratio for chosen
    logr_rejected = lp_rejected_actor - lp_rejected_ref  # [B]  log ratio for rejected

    # DPO loss: -log σ(β · (logr_w - logr_l))
    # Equivalent to binary cross-entropy where label=1 (chosen should win)
    loss = -F.logsigmoid(BETA * (logr_chosen - logr_rejected)).mean()

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
    optimizer.step()

    # Diagnostics
    reward_chosen   = BETA * logr_chosen.mean().item()
    reward_rejected = BETA * logr_rejected.mean().item()
    reward_margin   = reward_chosen - reward_rejected   # want this > 0
    chosen_win_rate = (logr_chosen > logr_rejected).float().mean().item()

    print(f"step {step+1}  loss={loss.item():.4f}  "
          f"r_chosen={reward_chosen:.3f}  r_rejected={reward_rejected:.3f}  "
          f"margin={reward_margin:.3f}  win_rate={chosen_win_rate:.0%}")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"""
DPO vs PPO:
  Models needed : DPO = Actor + Ref (frozen)
                  PPO = Actor + Critic + Ref (frozen) + Reward model (frozen)

  Data          : DPO = offline preference pairs  (y_w, y_l)  — static dataset
                  PPO = online rollouts            (generate → score → update)

  Distribution  : DPO trains on fixed data → distribution shift as policy changes
    shift problem  PPO always generates fresh → no shift, but expensive

  When DPO fails: if the preferred responses in the dataset are OOD for the
                  trained model, DPO can't recover (no online exploration).
                  PPO finds its way via online generation.

β interpretation (β={BETA}):
  β→0 : policy ignores KL → can diverge far from π_ref → reward hacking risk
  β→∞ : policy = π_ref   → no learning
  β=0.1: typical; controls how "bold" the policy can be vs staying safe
""")
