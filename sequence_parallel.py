"""
sequence_parallel.py — Megatron Sequence Parallelism (SP)

Concept: Standard TP replicates activations X on all ranks → activation memory
         is NOT reduced by TP. SP (Megatron-LM v3, 2022) shards X along the
         sequence dimension between TP blocks, cutting activation memory by TP×
         at zero extra communication cost.

Key Insight — replace AllReduce with AllGather + ReduceScatter:

  Standard TP:
    X [B, S, H] replicated
    ──────────────────────── col-parallel ──── row-parallel ────
                                                                 AllReduce
    Y [B, S, H] replicated

  With SP:
    X [B, S/TP, H] seq-sharded  ← SP region (LayerNorm, Dropout here)
         ↓ AllGather(seq)
    X [B, S,    H] replicated   ← TP region starts
    ──────────────────────── col-parallel ──── row-parallel ────
                                                 ReduceScatter(seq)
    Y [B, S/TP, H] seq-sharded  ← SP region resumes

  AllGather + ReduceScatter = same bytes as AllReduce (ring identity).
  But non-TP ops (LayerNorm, Dropout) now see only S/TP tokens → memory ÷TP.

Why LayerNorm works on a sequence shard:
  LayerNorm normalizes over H (hidden dim), independently per token.
  Sharding along S is orthogonal — each rank applies the full LN weight
  to its local tokens with zero communication.

Interview questions this file answers:
  Q: What is Sequence Parallelism and how does it differ from standard TP?
  Q: Why does AllGather + ReduceScatter equal AllReduce in bytes?
  Q: Why can LayerNorm be computed on a sequence shard without communication?
  Q: What is the activation memory saving with SP vs standard TP?
  Q: Where exactly does the SP region begin and end in a transformer block?

Run: torchrun --nproc_per_node=<N> sequence_parallel.py
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F

# ── Init ──────────────────────────────────────────────────────────────────────
dist.init_process_group(backend="nccl")
rank = dist.get_rank()
tp   = dist.get_world_size()
device = torch.device(f"cuda:{rank}")
torch.cuda.set_device(device)

# ── Config ────────────────────────────────────────────────────────────────────
B, S = 2, 512      # batch, total sequence length  (S must be divisible by TP)
H    = 1024        # hidden size
I    = 4096        # FFN intermediate (4×H is typical)
dtype = torch.bfloat16

assert S % tp == 0, f"S={S} must be divisible by TP={tp}"
S_local = S // tp   # tokens owned by this rank

torch.manual_seed(0)

# ── Build full tensors on rank 0, distribute to all ranks ─────────────────────
if rank == 0:
    W1_full = torch.randn(H, I, dtype=dtype, device=device)
    W2_full = torch.randn(I, H, dtype=dtype, device=device)
    b1_full = torch.zeros(I, dtype=dtype, device=device)
    b2      = torch.zeros(H, dtype=dtype, device=device)
    ln_w    = torch.ones(H,  dtype=dtype, device=device)   # LayerNorm weight
    ln_b    = torch.zeros(H, dtype=dtype, device=device)   # LayerNorm bias
    X_full  = torch.randn(B, S, H, dtype=dtype, device=device)

    # ── Reference (single GPU) ────────────────────────────────────────────
    X_norm  = F.layer_norm(X_full, [H], ln_w, ln_b)
    ref_out = F.gelu(X_norm @ W1_full + b1_full) @ W2_full + b2  # [B, S, H]
else:
    W1_full = torch.empty(H, I, dtype=dtype, device=device)
    W2_full = torch.empty(I, H, dtype=dtype, device=device)
    b1_full = torch.empty(I, dtype=dtype, device=device)
    b2      = torch.empty(H, dtype=dtype, device=device)
    ln_w    = torch.empty(H, dtype=dtype, device=device)
    ln_b    = torch.empty(H, dtype=dtype, device=device)
    X_full  = torch.empty(B, S, H, dtype=dtype, device=device)
    ref_out = torch.empty(B, S, H, dtype=dtype, device=device)

# Broadcast replicated tensors (LN weight, bias, b2, ref for verification)
for t in [b2, ln_w, ln_b, ref_out]:
    dist.broadcast(t, src=0)

# Scatter W1 column-shards [H, I] → [H, I/TP] per rank
W1_shard = torch.empty(H, I // tp, dtype=dtype, device=device)
dist.scatter(W1_shard,
             scatter_list=W1_full.split(I // tp, dim=1) if rank == 0 else None, src=0)

# Scatter b1 shards [I] → [I/TP] per rank
b1_shard = torch.empty(I // tp, dtype=dtype, device=device)
dist.scatter(b1_shard,
             scatter_list=b1_full.split(I // tp) if rank == 0 else None, src=0)

# Scatter W2 row-shards [I, H] → [I/TP, H] per rank
W2_shard = torch.empty(I // tp, H, dtype=dtype, device=device)
dist.scatter(W2_shard,
             scatter_list=W2_full.split(I // tp, dim=0) if rank == 0 else None, src=0)

# Scatter X along sequence: [B, S, H] → [B, S/TP, H] per rank
X_shard = torch.empty(B, S_local, H, dtype=dtype, device=device)
dist.scatter(X_shard,
             scatter_list=X_full.split(S_local, dim=1) if rank == 0 else None, src=0)

# ═══════════════════════════════════════════════════════════════════════════════
#  SP region begins: each rank holds [B, S/TP, H]
# ═══════════════════════════════════════════════════════════════════════════════

# ── Step 1: LayerNorm on local sequence shard — NO communication ──────────────
#
#  LayerNorm(x) = (x - mean) / std * γ + β,  stats computed over H per token.
#  Since H is fully local (not sharded), each rank computes exact LN independently.
#  This is the core win: all non-TP ops (LN, Dropout, residual add) live here,
#  costing S/TP memory instead of S.
#
X_norm_shard = F.layer_norm(X_shard, [H], ln_w, ln_b)   # [B, S/TP, H]

# ── Step 2: AllGather — restore full sequence before column-parallel linear ───
#
#  Column-parallel linear needs full X on every rank (each computes X @ W1_shard).
#  AllGather collects sequence shards from all TP ranks along dim=1.
#
#  Volume per rank: (TP-1)/TP × B×S×H × 2 bytes  (ring AllGather)
#
chunks = [torch.empty_like(X_norm_shard) for _ in range(tp)]
dist.all_gather(chunks, X_norm_shard.contiguous())
X_gathered = torch.cat(chunks, dim=1)          # [B, S, H]

# ═══════════════════════════════════════════════════════════════════════════════
#  TP region: standard Megatron col-parallel → row-parallel
# ═══════════════════════════════════════════════════════════════════════════════

# ── Step 3: Column-parallel linear + GELU — no communication ─────────────────
#
#  X [B, S, H] is replicated. W1_shard [H, I/TP] is a column slice.
#  Each rank computes a partial intermediate independently.
#
Z = F.gelu(X_gathered @ W1_shard + b1_shard)   # [B, S, I/TP]

# ── Step 4: Row-parallel linear — produces partial output ─────────────────────
#
#  W2_shard [I/TP, H] is a row slice. Z feeds directly in (shapes align).
#  Result is a partial sum; full output = sum over all ranks.
#
Y_partial = Z @ W2_shard                        # [B, S, H]  — partial sum

# ── Step 5: ReduceScatter — sum partials AND shard by sequence ────────────────
#
#  Standard TP uses AllReduce here: sum Y_partial → every rank holds full [B,S,H].
#  SP uses ReduceScatter instead:
#    - Split Y_partial into TP chunks along seq dim (one chunk per rank)
#    - Each rank receives the sum of chunk[rank] from all other ranks
#    - Result: rank r holds the fully-reduced output for tokens [r*S/TP : (r+1)*S/TP]
#
#  Volume per rank: (TP-1)/TP × B×S×H × 2 bytes  (ring ReduceScatter)
#  = same as AllGather above = same as AllReduce in standard TP
#
Y_shards = list(Y_partial.split(S_local, dim=1))   # TP × [B, S/TP, H]
out_shard = torch.empty(B, S_local, H, dtype=dtype, device=device)
dist.reduce_scatter(out_shard, Y_shards)           # [B, S/TP, H]

# Add bias (replicated, applied to local shard independently)
out_shard = out_shard + b2

# ═══════════════════════════════════════════════════════════════════════════════
#  SP region resumes: each rank holds [B, S/TP, H], ready for next block
# ═══════════════════════════════════════════════════════════════════════════════

# ── Verify ────────────────────────────────────────────────────────────────────
ref_shard = ref_out[:, rank * S_local : (rank + 1) * S_local, :]
max_err   = (out_shard - ref_shard).abs().max()

# Gather worst-case error across all ranks
dist.all_reduce(max_err, op=dist.ReduceOp.MAX)

if rank == 0:
    print(f"\nTP={tp}  BF16  [B={B}, S={S}, H={H}, I={I}]")
    print(f"Max |sp_out - ref| across all ranks = {max_err.item():.2e}"
          f"  (expect <1e-2 for BF16)")

    # ── Communication analysis ──────────────────────────────────────────────
    act_bytes   = B * S * H * 2                      # full activation, BF16
    ring_vol    = (tp - 1) / tp * act_bytes           # ring algo: each rank sends this much

    print(f"\nCommunication per MLP block  (ring algorithm, per rank):")
    print(f"  SP  AllGather     (step 2): {ring_vol/1024:7.1f} KB  → assembles full seq")
    print(f"  SP  ReduceScatter (step 5): {ring_vol/1024:7.1f} KB  → sums + re-shards")
    print(f"  SP  Total                 : {2*ring_vol/1024:7.1f} KB")
    print(f"  TP  AllReduce  (baseline) : {2*ring_vol/1024:7.1f} KB  (= RS + AG by ring identity)")
    print(f"  → SP and TP move IDENTICAL bytes; no communication overhead from SP")

    # ── Activation memory analysis ──────────────────────────────────────────
    act_full  = B * S       * H * 2   # standard TP: X replicated
    act_sp    = B * S_local * H * 2   # SP: X seq-sharded

    print(f"\nActivation memory for non-TP ops (e.g. LayerNorm input) per rank:")
    print(f"  Standard TP : {act_full/1024:7.1f} KB  (full [B, S, H] on every rank)")
    print(f"  With SP     : {act_sp/1024:7.1f} KB  ([B, S/TP, H], ÷{tp})")
    print(f"  → SP cuts activation memory for ALL non-TP ops by {tp}×")
    print(f"  → With gradient checkpointing, this {tp}× saving compounds across layers")

dist.destroy_process_group()
