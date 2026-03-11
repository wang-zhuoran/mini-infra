"""
dp_zero.py — ZeRO-1 / 2 / 3 Data Parallel Training

Concept: Standard DDP replicates optimizer states + gradients + params on every
         rank → massive redundancy. ZeRO eliminates it stage by stage.

Memory per rank (Ψ params, mixed precision BF16 + FP32 optimizer):
  ┌──────────────────┬────────────┬────────────┬────────────┬────────────┐
  │                  │ Params     │ Grads      │ Opt states │ Total      │
  │                  │ (BF16, 2B) │ (BF16, 2B) │ (FP32, 8B) │            │
  ├──────────────────┼────────────┼────────────┼────────────┼────────────┤
  │ DDP baseline     │ 2Ψ         │ 2Ψ         │ 8Ψ         │ 12Ψ        │
  │ ZeRO-1           │ 2Ψ         │ 2Ψ         │ 8Ψ/DP      │ (4+8/DP)Ψ  │
  │ ZeRO-2           │ 2Ψ         │ 2Ψ/DP      │ 8Ψ/DP      │ (2+10/DP)Ψ │
  │ ZeRO-3           │ 2Ψ/DP      │ 2Ψ/DP      │ 8Ψ/DP      │ 12Ψ/DP     │
  └──────────────────┴────────────┴────────────┴────────────┴────────────┘
  * Opt states = Adam: master weights (4B) + momentum (4B) + variance (4B) ≈ 12B/param
    simplified to 8B here (master weight counted separately)

Communication vs DDP (one AllReduce = one ReduceScatter + one AllGather):
  ZeRO-1: AllReduce grads    (same as DDP)
  ZeRO-2: ReduceScatter grads (half the peak comm buffer vs AllReduce)
  ZeRO-3: AllGather params (fwd) + ReduceScatter grads (bwd) = 3× DDP volume

Interview questions this file answers:
  Q: What does each ZeRO stage actually shard?
  Q: Why is ZeRO-3 communication 3× DDP but ZeRO-2 is the same?
  Q: In your GRPO project with ZeRO-3, what's the memory per GPU for a 3B model?
  Q: What's ReduceScatter and how does it differ from AllReduce?
  Q: When would you choose ZeRO-2 over ZeRO-3?

Run: torchrun --nproc_per_node=<N> dp_zero.py
"""

import torch
import torch.nn as nn
import torch.distributed as dist

# ── Init ──────────────────────────────────────────────────────────────────────
dist.init_process_group(backend="nccl")
rank = dist.get_rank()
dp   = dist.get_world_size()
device = torch.device(f"cuda:{rank}")
torch.cuda.set_device(device)

# ── Tiny model ────────────────────────────────────────────────────────────────
HIDDEN = 1024
N_LAYERS = 4

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(HIDDEN, HIDDEN) for _ in range(N_LAYERS)
        ])
    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return x

model = MLP().to(device, dtype=torch.bfloat16)

total_params = sum(p.numel() for p in model.parameters())
param_bytes  = total_params * 2   # BF16
grad_bytes   = total_params * 2   # BF16
opt_bytes    = total_params * 12  # Adam: master(FP32=4B) + m(4B) + v(4B)

if rank == 0:
    print(f"Model: {total_params/1e6:.1f}M params")
    print(f"\nMemory per rank (DP={dp}):")
    print(f"  DDP    : {(param_bytes + grad_bytes + opt_bytes)/1e6:.0f} MB total")
    print(f"  ZeRO-1 : {(param_bytes + grad_bytes + opt_bytes/dp)/1e6:.0f} MB")
    print(f"  ZeRO-2 : {(param_bytes + (grad_bytes + opt_bytes)/dp)/1e6:.0f} MB")
    print(f"  ZeRO-3 : {(param_bytes + grad_bytes + opt_bytes)/dp/1e6:.0f} MB  (÷{dp})")
    print(f"\n  For 3B params @ DP=8 (your 8×4090 GRPO setup):")
    P = 3e9
    print(f"  DDP    : {(P*2 + P*2 + P*12)/1e9:.0f} GB per GPU")
    print(f"  ZeRO-3 : {(P*2 + P*2 + P*12)/8/1e9:.1f} GB per GPU  ← fits on 24GB")

# ─────────────────────────────────────────────────────────────────────────────
# ZERO-1: Shard optimizer states only
# ─────────────────────────────────────────────────────────────────────────────
# Every rank: full params, full grads, 1/DP optimizer state shard
#
# Forward/backward: identical to DDP
# Gradient sync: AllReduce (same as DDP)
# Update: each rank only updates its 1/DP param shard → AllGather to sync

def zero1_step(model, x, optimizer):
    """
    Simulates ZeRO-1: standard backward + AllReduce grads.
    Each rank would normally only hold opt states for its shard.
    Here we show the gradient communication pattern.
    """
    loss = model(x).mean()
    loss.backward()

    # AllReduce gradients (average across ranks) — same as DDP
    for p in model.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

    # In real ZeRO-1: each rank only steps its param shard's optimizer
    # then AllGather to restore full params. Simplified here:
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()

# ─────────────────────────────────────────────────────────────────────────────
# ZERO-2: Shard optimizer states + gradients
# ─────────────────────────────────────────────────────────────────────────────
# Key change: AllReduce → ReduceScatter
#
# ReduceScatter: reduce (sum) across ranks, then scatter shards
#   Each rank ends up with gradient for its OWN param shard only.
#   No AllGather needed — we update immediately from the shard.
#   After update: AllGather params to restore full model for next forward.
#
# Communication volume: same as DDP (AllReduce = ReduceScatter + AllGather,
#   but we skip the AllGather of grads — only AllGather params after update)

def zero2_step(model, x):
    """
    ZeRO-2: ReduceScatter gradients → each rank gets its own grad shard.
    """
    loss = model(x).mean()
    loss.backward()

    params  = [p for p in model.parameters() if p.grad is not None]
    grads   = [p.grad.data for p in params]

    # Flatten all grads into one buffer for efficient communication
    flat_grad = torch.cat([g.flatten() for g in grads])

    # Shard: rank r will receive flat_grad[r * shard : (r+1) * shard]
    total   = flat_grad.numel()
    # Pad to make divisible by dp
    pad     = (dp - total % dp) % dp
    if pad:
        flat_grad = torch.cat([flat_grad, flat_grad.new_zeros(pad)])
    shard_size = flat_grad.numel() // dp

    grad_shard = torch.empty(shard_size, dtype=flat_grad.dtype, device=device)

    # ReduceScatter: sum across ranks, each rank gets one contiguous shard
    dist.reduce_scatter_tensor(grad_shard, flat_grad, op=dist.ReduceOp.AVG)

    # Each rank now has gradient for its parameter shard.
    # Real ZeRO-2: update optimizer state for this shard only, then AllGather.
    # (Optimizer step omitted for clarity)

    return loss.item(), grad_shard

# ─────────────────────────────────────────────────────────────────────────────
# ZERO-3: Shard optimizer states + gradients + parameters
# ─────────────────────────────────────────────────────────────────────────────
# Each rank holds only 1/DP of parameters at rest.
# Before each layer forward: AllGather params for that layer.
# After each layer backward: ReduceScatter gradients for that layer.
# Free gathered params immediately to stay at 1/DP memory.
#
# Communication: 2× AllGather (fwd + bwd) + 1× ReduceScatter = 3× DDP
# Memory at peak: 1/DP params at rest, full layer params during compute

class ZeRO3Linear(nn.Module):
    """
    Minimal ZeRO-3 layer: params sharded, AllGather before compute.
    Real implementation: DeepSpeed wraps every parameter with hooks.
    """
    def __init__(self, in_f, out_f):
        super().__init__()
        # Each rank holds 1/dp of the output features
        shard_out = out_f // dp
        self.weight_shard = nn.Parameter(
            torch.randn(shard_out, in_f, device=device, dtype=torch.bfloat16) * 0.02
        )
        self.full_out = out_f

    def forward(self, x):
        # AllGather weight shards from all ranks → reconstruct full weight
        shards = [torch.empty_like(self.weight_shard) for _ in range(dp)]
        dist.all_gather(shards, self.weight_shard)
        full_weight = torch.cat(shards, dim=0)   # [out_f, in_f]

        out = torch.nn.functional.linear(x, full_weight)

        # After backward: gradient for full_weight → ReduceScatter back to shards
        # (handled by autograd hooks in DeepSpeed/FSDP)
        return out

# ── Run and compare ───────────────────────────────────────────────────────────
torch.manual_seed(rank)
x = torch.randn(8, HIDDEN, device=device, dtype=torch.bfloat16)

# Simple optimizer for ZeRO-1 demo
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

loss1 = zero1_step(model, x, optimizer)
loss2, grad_shard = zero2_step(model, x)

# ZeRO-3 layer demo
z3_layer = ZeRO3Linear(HIDDEN, HIDDEN).to(device)
out_z3   = z3_layer(x)

dist.barrier()
if rank == 0:
    print(f"\nZeRO-1 loss: {loss1:.4f}")
    print(f"ZeRO-2 grad shard shape: {grad_shard.shape}  "
          f"(full grad would be {sum(p.numel() for p in model.parameters())})")
    print(f"ZeRO-3 layer output: {out_z3.shape}")

    print(f"""
Communication summary (Ψ = total params, N = DP size):
  DDP    : AllReduce grads          = 2Ψ  bytes  (ReduceScatter + AllGather)
  ZeRO-1 : AllReduce grads          = 2Ψ  bytes  (same)
  ZeRO-2 : ReduceScatter grads      = Ψ   bytes  (skip AllGather of grads)
            + AllGather params       = Ψ   bytes  (after update)
           Total                    = 2Ψ  bytes  (same as DDP)
  ZeRO-3 : AllGather params  (fwd)  = Ψ   bytes
            AllGather params  (bwd)  = Ψ   bytes
            ReduceScatter grads      = Ψ   bytes
           Total                    = 3Ψ  bytes  (3× DDP)

When to use what:
  ZeRO-1 → small model, optimizer state is bottleneck (e.g. large embedding)
  ZeRO-2 → medium model, gradient memory is bottleneck; same comm as DDP
  ZeRO-3 → large model (3B+), param memory bottleneck; accept 3× comm cost
            Your 3B GRPO on 8×4090: ZeRO-3 is the only option that fits.
""")

dist.destroy_process_group()
