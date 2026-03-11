"""
tp_linear.py — Megatron-style Tensor Parallelism

Concept: Split weight matrices across GPUs so each rank does 1/TP of the compute.

Key Insight (Megatron's real contribution):
  Pair Column-parallel(W1) → Row-parallel(W2) so intermediate activations
  NEVER need to be communicated. Only ONE AllReduce per MLP block.

  Naive approach : AllGather after col-linear + AllGather after row-linear = 2 comms
  Megatron       : ──────────────────────────────────── AllReduce at the end = 1 comm

Interview questions this file answers:
  Q: What's column-parallel vs row-parallel?
  Q: Why does Megatron only need 1 AllReduce per MLP block?
  Q: What is the AllReduce communication volume in bytes?
  Q: What is Sequence Parallelism and why does it help?

Run: torchrun --nproc_per_node=<N> tp_linear.py
"""

import os
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
B, S = 2, 512     # batch, sequence length
H    = 1024       # hidden size
I    = 4096       # FFN intermediate size (4×H is typical)
dtype = torch.bfloat16

torch.manual_seed(0)

# ── Build weights on rank 0, scatter shards to each rank ─────────────────────
# In real training these are initialized per-rank directly; we scatter here
# just to verify correctness against a single-GPU reference.
if rank == 0:
    W1_full = torch.randn(H, I, dtype=dtype, device=device)
    W2_full = torch.randn(I, H, dtype=dtype, device=device)
    b1_full = torch.zeros(I, dtype=dtype, device=device)
    b2      = torch.zeros(H, dtype=dtype, device=device)
    X       = torch.randn(B, S, H, dtype=dtype, device=device)
    # Reference output (single GPU)
    ref_out = F.gelu(X @ W1_full + b1_full) @ W2_full + b2
else:
    W1_full = torch.empty(H, I, dtype=dtype, device=device)
    W2_full = torch.empty(I, H, dtype=dtype, device=device)
    b1_full = torch.empty(I, dtype=dtype, device=device)
    b2      = torch.empty(H, dtype=dtype, device=device)
    X       = torch.empty(B, S, H, dtype=dtype, device=device)
    ref_out = None

# Broadcast X and b2 (replicated tensors) to all ranks
dist.broadcast(X,  src=0)
dist.broadcast(b2, src=0)

# Scatter W1 column-shards: W1_full [H, I] → each rank gets [H, I//TP]
W1_shard = torch.empty(H, I // tp, dtype=dtype, device=device)
dist.scatter(W1_shard, scatter_list=W1_full.split(I // tp, dim=1) if rank == 0 else None, src=0)

# Scatter b1 shards: b1_full [I] → each rank gets [I//TP]
b1_shard = torch.empty(I // tp, dtype=dtype, device=device)
dist.scatter(b1_shard, scatter_list=b1_full.split(I // tp) if rank == 0 else None, src=0)

# Scatter W2 row-shards: W2_full [I, H] → each rank gets [I//TP, H]
W2_shard = torch.empty(I // tp, H, dtype=dtype, device=device)
dist.scatter(W2_shard, scatter_list=W2_full.split(I // tp, dim=0) if rank == 0 else None, src=0)

# ── Step 1: Column-parallel linear + GELU ────────────────────────────────────
#
#  W1 [H, I] split along columns → rank r owns W1[:, r*I//TP : (r+1)*I//TP]
#  X is replicated → each rank computes independently, zero communication
#
Z = F.gelu(X @ W1_shard + b1_shard)   # [B, S, I//TP]

# ── Step 2: Row-parallel linear ───────────────────────────────────────────────
#
#  W2 [I, H] split along rows → rank r owns W2[r*I//TP : (r+1)*I//TP, :]
#  Z feeds directly into W2_shard (shapes align), giving a PARTIAL output
#
Y = Z @ W2_shard                       # [B, S, H]  — partial sum on each rank

# ── Step 3: AllReduce ─────────────────────────────────────────────────────────
#
#  Sum partial Y's across all ranks → each rank now holds the full output
#  This is the ONE communication per MLP block.
#
#  AllReduce (ring) volume per rank: 2 * (TP-1)/TP * numel * bytes
#
dist.all_reduce(Y, op=dist.ReduceOp.SUM)
Y = Y + b2   # bias added AFTER reduce (only once, not TP times)

# ── Verify on rank 0 ──────────────────────────────────────────────────────────
dist.broadcast(ref_out if rank == 0 else torch.empty_like(Y), src=0)
if rank == 0:
    max_err = (Y - ref_out).abs().max().item()
    print(f"\nTP={tp}  BF16  [B={B}, S={S}, H={H}, I={I}]")
    print(f"Max |tp_out - ref| = {max_err:.2e}  (expect <1e-2 for BF16)")

    # ── Communication analysis ─────────────────────────────────────────────
    numel         = B * S * H
    allreduce_vol = 2 * (tp - 1) / tp * numel * 2        # BF16 = 2 bytes
    naive_vol     = 2 * (B * S * I * 2)                  # naive: 2× AllGather of intermediate

    print(f"\nCommunication per MLP block:")
    print(f"  Megatron 1× AllReduce : {allreduce_vol/1024:7.1f} KB  (tensor size = B*S*H)")
    print(f"  Naive    2× AllGather : {naive_vol/1024:7.1f} KB  (tensor size = B*S*I)")
    print(f"  Saving                : {naive_vol/allreduce_vol:.1f}× less data (= I/H = {I//H}×)")

    # ── Sequence Parallelism note ──────────────────────────────────────────
    # Standard TP: X is replicated on all ranks → activation memory NOT reduced
    # Sequence Parallel (Megatron v3): shard X along seq dim between TP blocks
    #
    #   [rank r holds X[:, r*S//TP:(r+1)*S//TP, :]]
    #        ↓  AllGather(X) along seq before col-parallel
    #   Column-parallel linear
    #        ↓  (no comm)
    #   Row-parallel linear
    #        ↓  ReduceScatter(Y) along seq  ← replaces AllReduce
    #   [rank r holds Y[:, r*S//TP:(r+1)*S//TP, :]]
    #
    # Result: activation memory ÷ TP, same communication bytes (AG + RS = AR)
    act_baseline = B * S * H * 2 / 1024
    print(f"\nSequence Parallelism activation memory:")
    print(f"  Standard TP : {act_baseline:.0f} KB per rank (X replicated)")
    print(f"  With SP     : {act_baseline/tp:.0f} KB per rank (÷{tp})")

dist.destroy_process_group()
