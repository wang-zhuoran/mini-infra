"""
pp_schedule.py — Pipeline Parallelism: GPipe vs 1F1B

Concept: Split model layers across GPUs (stages). Microbatches flow
         through stages like an assembly line.

Key Insight:
  GPipe  : all-forward then all-backward → simple, but large activation memory
  1F1B   : steady-state alternates 1 Forward + 1 Backward → memory = O(PP) not O(M)
  Both have the same bubble fraction: (PP-1) / (M + PP - 1)
  Interleaved 1F1B (virtual stages): bubble ÷ V  but more P2P comm overhead

Interview questions this file answers:
  Q: What is pipeline bubble and how do you calculate it?
  Q: Why does 1F1B use less memory than GPipe despite same bubble?
  Q: What is the interleaved schedule and when is it worth it?
  Q: What are the P2P communication primitives between stages?

Run: torchrun --nproc_per_node=<N> pp_schedule.py
"""

import time
import torch
import torch.nn as nn
import torch.distributed as dist

# ── Init ──────────────────────────────────────────────────────────────────────
dist.init_process_group(backend="nccl")
rank  = dist.get_rank()
pp    = dist.get_world_size()   # pipeline parallel size = number of stages
device = torch.device(f"cuda:{rank}")
torch.cuda.set_device(device)

is_first = rank == 0
is_last  = rank == pp - 1

# ── Config ────────────────────────────────────────────────────────────────────
M          = 8        # number of microbatches (should be >> PP to amortize bubble)
HIDDEN     = 1024
LAYERS_PER_STAGE = 2  # transformer layers per GPU
dtype      = torch.bfloat16

# ── Model: each rank holds LAYERS_PER_STAGE linear layers ────────────────────
# In real Megatron: TransformerLayer × layers_per_stage
stage = nn.Sequential(
    *[nn.Linear(HIDDEN, HIDDEN, dtype=dtype) for _ in range(LAYERS_PER_STAGE)]
).to(device)

optimizer = torch.optim.Adam(stage.parameters(), lr=1e-4)

# ── P2P helpers ───────────────────────────────────────────────────────────────
# In Megatron: megatron/core/pipeline_parallel/p2p_communication.py
ACT_SHAPE = (4, HIDDEN)   # (local_batch, hidden) per microbatch

def send_fwd(tensor):
    """Send activation to next stage."""
    dist.send(tensor.contiguous(), dst=rank + 1)

def recv_fwd():
    """Receive activation from previous stage."""
    buf = torch.empty(ACT_SHAPE, dtype=dtype, device=device)
    dist.recv(buf, src=rank - 1)
    return buf

def send_bwd(grad):
    """Send gradient to previous stage."""
    dist.send(grad.contiguous(), dst=rank - 1)

def recv_bwd():
    """Receive gradient from next stage."""
    buf = torch.empty(ACT_SHAPE, dtype=dtype, device=device)
    dist.recv(buf, src=rank + 1)
    return buf

# ── Forward / Backward per microbatch ─────────────────────────────────────────
def forward_step(x):
    """Run one microbatch forward through this stage. Returns (output, saved_input)."""
    x = x.detach().requires_grad_(True)
    y = stage(x)
    return y, x

def backward_step(y, x, grad_y):
    """Run one microbatch backward. Returns grad w.r.t. input (to send upstream)."""
    y.backward(grad_y)
    return x.grad

# ─────────────────────────────────────────────────────────────────────────────
# GPIPE SCHEDULE
# ─────────────────────────────────────────────────────────────────────────────
# Timeline (PP=4, M=8):
#
#  rank  t=0  t=1  t=2  t=3  t=4  t=5  t=6  t=7 | t=8  t=9  t=10 t=11 t=12 t=13 t=14 t=15
#    0   F0   F1   F2   F3   F4   F5   F6   F7  |  B0   B1   B2   B3   B4   B5   B6   B7
#    1   idle F0   F1   F2   F3   F4   F5   F6  |  F7   B0   B1   B2   B3   B4   B5   B6   B7
#    2   idle idle F0   F1   F2   F3   F4   F5  |  F6   F7   B0   B1   B2   B3   B4   B5   B6   B7
#    3   idle idle idle F0   F1   F2   F3   F4  |  F5   F6   F7   B0   B1   B2   B3   B4   B5   B6   B7
#                 ^^^bubble^^^                         ^all stages busy = no bubble^
#
# Bubble fraction = (PP-1) / (M + PP - 1)
# Memory: must store ALL M microbatch activations across all stages simultaneously

def run_gpipe():
    dist.barrier()
    t0 = time.perf_counter()
    optimizer.zero_grad()

    fwd_cache = {}   # microbatch_id → (output, saved_input)

    # ── Phase 1: All M microbatches forward ───────────────────────────────────
    for m in range(M):
        if is_first:
            x = torch.randn(ACT_SHAPE, dtype=dtype, device=device)
        else:
            x = recv_fwd()

        y, x_saved = forward_step(x)
        fwd_cache[m] = (y, x_saved)

        if not is_last:
            send_fwd(y)
        # Last rank: compute dummy loss (no send)

    # ── Phase 2: All M microbatches backward ──────────────────────────────────
    for m in range(M):
        y, x_saved = fwd_cache[m]

        if is_last:
            grad_y = torch.ones_like(y)   # dL/dy from loss
        else:
            grad_y = recv_bwd()

        grad_x = backward_step(y, x_saved, grad_y)

        if not is_first:
            send_bwd(grad_x)

    optimizer.step()
    dist.barrier()
    return time.perf_counter() - t0

# ─────────────────────────────────────────────────────────────────────────────
# 1F1B SCHEDULE (PipeDream-Flush)
# ─────────────────────────────────────────────────────────────────────────────
# Timeline (PP=4, M=8):
#
#  rank  0    1    2    3    4    5    6    7    8    9    10   11   12   13
#    0   F0   F1   F2   F3   B0   F4   B1   F5   B2   F6   B3   F7   B4   B5   B6   B7
#    1   idle F0   F1   F2   F3   B0   F4   B1   F5   B2   F6   B3   F7   B4   B5   B6   B7
#    2   idle idle F0   F1   F2   F3   B0   F4   B1   F5   B2   F6   B3   F7   ...
#    3   idle idle idle F0   F1   F2   F3   B0   F1   B1   F2   B2   ...
#
# Warmup phase: first (PP - rank - 1) microbatches forward only
# Steady state: 1F + 1B interleaved  ← memory constant = PP microbatch activations
# Drain phase:  remaining backwards
#
# Bubble fraction = same (PP-1)/(M+PP-1), but memory = O(PP) not O(M)

def run_1f1b():
    dist.barrier()
    t0 = time.perf_counter()
    optimizer.zero_grad()

    fwd_cache = {}
    fwd_queue = list(range(M))   # microbatches to forward
    bwd_queue = []               # microbatches ready to backward

    warmup_steps = pp - rank - 1   # fill pipeline before steady state
    num_warmup   = min(warmup_steps, M)

    # ── Warmup: forward only ─────────────────────────────────────────────────
    for m in fwd_queue[:num_warmup]:
        if is_first:
            x = torch.randn(ACT_SHAPE, dtype=dtype, device=device)
        else:
            x = recv_fwd()
        y, x_saved = forward_step(x)
        fwd_cache[m] = (y, x_saved)
        if not is_last:
            send_fwd(y)
        bwd_queue.append(m)

    # ── Steady state: 1F + 1B ─────────────────────────────────────────────
    remaining_fwd = fwd_queue[num_warmup:]
    for m_fwd in remaining_fwd:
        # Forward
        if is_first:
            x = torch.randn(ACT_SHAPE, dtype=dtype, device=device)
        else:
            x = recv_fwd()
        y, x_saved = forward_step(x)
        fwd_cache[m_fwd] = (y, x_saved)
        if not is_last:
            send_fwd(y)
        bwd_queue.append(m_fwd)

        # Backward (oldest microbatch in flight)
        m_bwd = bwd_queue.pop(0)
        y, x_saved = fwd_cache.pop(m_bwd)
        if is_last:
            grad_y = torch.ones_like(y)
        else:
            grad_y = recv_bwd()
        grad_x = backward_step(y, x_saved, grad_y)
        if not is_first:
            send_bwd(grad_x)

    # ── Drain: backward only ─────────────────────────────────────────────────
    for m_bwd in bwd_queue:
        y, x_saved = fwd_cache.pop(m_bwd)
        if is_last:
            grad_y = torch.ones_like(y)
        else:
            grad_y = recv_bwd()
        grad_x = backward_step(y, x_saved, grad_y)
        if not is_first:
            send_bwd(grad_x)

    optimizer.step()
    dist.barrier()
    return time.perf_counter() - t0

# ── Run both schedules and compare ───────────────────────────────────────────
t_gpipe = run_gpipe()
t_1f1b  = run_1f1b()

if rank == 0:
    bubble = (pp - 1) / (M + pp - 1)
    print(f"\nPP={pp}  M={M} microbatches  {LAYERS_PER_STAGE} layers/stage")
    print(f"Bubble fraction : (PP-1)/(M+PP-1) = ({pp}-1)/({M}+{pp}-1) = {bubble:.1%}")
    print(f"  → reduce by increasing M (more microbatches per batch)")
    print(f"\nTiming:")
    print(f"  GPipe : {t_gpipe*1000:.1f} ms")
    print(f"  1F1B  : {t_1f1b*1000:.1f} ms")
    print(f"\nMemory in steady state:")
    print(f"  GPipe : stores {M} microbatch activations  (O(M))")
    print(f"  1F1B  : stores {pp} microbatch activations (O(PP))  ← {M//pp}× less")
    print(f"\nInterleaved 1F1B (virtual stages V):")
    for V in [2, 4]:
        print(f"  V={V}: bubble = (PP-1)/(M*V) = {(pp-1)/(M*V):.1%}  "
              f"(÷{V} vs basic, but {V}× more P2P comms)")

dist.destroy_process_group()
