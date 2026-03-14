"""
paged_attention.py — Paged Attention (vLLM-style KV Cache)

Concept: Contiguous KV cache pre-allocates max_seq_len slots per sequence,
         causing two problems:
           1. Internal fragmentation — slots past the current token count are
              wasted (a 10-token sequence in a 4096-slot reservation wastes 99.8%).
           2. External fragmentation — finished sequences free their contiguous
              slabs, but new sequences may need different sizes, leaving unusable
              gaps (like heap fragmentation in C).

         Paged attention (Kwon et al. 2023, vLLM) fixes this by treating the KV
         cache like OS virtual memory:
           • Physical KV store: a pool of fixed-size blocks (pages), each holding
             block_size tokens.
           • Block table: per-sequence mapping of logical_block → physical_block_id.
           • Block allocator: a free list of physical blocks; allocate/free on demand.

         Sequences grow one block at a time. Freed blocks return to the pool
         immediately and can be reused by any new sequence.

Key Insight — Block Table is virtual memory for KV cache:

  Virtual memory (OS)          Paged Attention
  ─────────────────────────    ─────────────────────────────────────────────
  page table                   block_table[seq][logical_block] → phys_block
  page frame pool              free_blocks (pool of physical block IDs)
  page fault → allocate        token appended past block boundary → new block
  page sharing (copy-on-write) prefix caching, beam search sharing
  TLB                          (no HW analog; CUDA kernel does the lookup)

Key Insight — Prefix Caching (RadixAttention in SGLang):

  If two requests share a common prefix (e.g. a long system prompt), they can
  map their first N logical blocks to the SAME physical blocks.
  No extra memory, no recomputation — just reference the same physical pages.
  SGLang generalizes this to a radix tree over token sequences.

Key Insight — Real vLLM kernel vs this demo:

  This demo   : gather K/V from scattered blocks → contiguous buffer → attention.
                Simple to understand; costs one extra memory copy.

  Real vLLM   : custom CUDA kernel reads block_table and fetches K/V directly
                from physical blocks without materializing a contiguous buffer.
                Threads within a warp each handle one block; no gather copy.

Key Insight — Memory utilization:

  Contiguous  : worst case 0% utilization (1 token in a 4096-slot reservation).
                Average ~50% waste for the last partial block per sequence.
  Paged       : waste ≤ (block_size - 1) tokens per sequence (last partial block).
                With block_size=16, waste ≤ 15 tokens regardless of max_seq_len.

Interview questions this file answers:
  Q: What is paged attention and what problem does it solve?
  Q: How does the block table work? Analogy to OS virtual memory?
  Q: What is internal vs external KV cache fragmentation?
  Q: How does prefix caching (SGLang RadixAttention) work?
  Q: What is the difference between the gather approach and a real vLLM kernel?
  Q: What is the worst-case memory waste with paged attention?

Run: python paged_attention.py
"""

import torch
import torch.nn.functional as F
import math
from typing import List

# ── Config ────────────────────────────────────────────────────────────────────
H      = 8     # Q heads
H_KV   = 2     # KV heads  (GQA G=4)
D      = 32    # head dim  →  model_dim = H × D = 256
L      = 2     # transformer layers
dtype  = torch.float32

BLOCK_SIZE     = 4     # tokens per physical block (vLLM default is 16)
NUM_PHY_BLOCKS = 32    # total physical blocks in the pool

G         = H // H_KV
model_dim = H * D
kv_dim    = H_KV * D
scale     = math.sqrt(D)

assert H % H_KV == 0
torch.manual_seed(0)

# ── Weights ───────────────────────────────────────────────────────────────────
layers = [{
    "W_q": torch.randn(model_dim, model_dim) * 0.02,
    "W_k": torch.randn(kv_dim,    model_dim) * 0.02,
    "W_v": torch.randn(kv_dim,    model_dim) * 0.02,
    "W_o": torch.randn(model_dim, model_dim) * 0.02,
} for _ in range(L)]

# ═══════════════════════════════════════════════════════════════════════════════
#  PHYSICAL KV STORE
#
#  Shape: [NUM_PHY_BLOCKS, 2, H_KV, BLOCK_SIZE, D]
#          ↑               ↑  ↑     ↑            ↑
#          physical block  K/V KV-head  slot in block  head dim
#
#  Compare with contiguous KV cache: [L, 2, B, H_KV, S_max, D]
#  Paged: ALL sequences share one pool; blocks are assigned dynamically.
#
# ═══════════════════════════════════════════════════════════════════════════════
kv_store = torch.zeros(NUM_PHY_BLOCKS, 2, H_KV, BLOCK_SIZE, D, dtype=dtype)


# ── Block Allocator ───────────────────────────────────────────────────────────
class BlockAllocator:
    """
    Free list of physical block IDs.
    allocate() → physical block id  (raises if OOM)
    free(block_id)                   (returns block to pool)
    """
    def __init__(self, num_blocks: int):
        self.free_blocks: List[int] = list(range(num_blocks))

    def allocate(self) -> int:
        if not self.free_blocks:
            raise RuntimeError("Out of KV cache memory (no free physical blocks)")
        return self.free_blocks.pop()

    def free(self, block_id: int):
        self.free_blocks.append(block_id)

    def num_free(self) -> int:
        return len(self.free_blocks)


allocator = BlockAllocator(NUM_PHY_BLOCKS)


# ── Sequence State ────────────────────────────────────────────────────────────
class SeqState:
    """
    Per-sequence paging state.

    block_table : list of physical block IDs in logical order.
                  block_table[i] = physical block for tokens [i*BS : (i+1)*BS].
    seq_len     : number of tokens currently stored for this sequence.
    """
    def __init__(self):
        self.block_table: List[int] = []
        self.seq_len: int = 0

    def last_block_offset(self) -> int:
        """Offset of the next token within the current last physical block."""
        return self.seq_len % BLOCK_SIZE

    def needs_new_block(self) -> bool:
        return (self.seq_len == 0) or (self.last_block_offset() == 0)


# ── Write K/V for one token ───────────────────────────────────────────────────
def append_token_kv(seq: SeqState, k: torch.Tensor, v: torch.Tensor,
                    layer_idx: int, alloc: BlockAllocator):
    """
    k, v: [H_KV, D]  — K and V for the new token (one token at a time).

    If the last block is full (or there are no blocks yet), allocate a new one.
    Write k/v into kv_store at the appropriate physical block + slot offset.
    """
    if seq.needs_new_block():
        phys = alloc.allocate()
        seq.block_table.append(phys)

    slot = seq.last_block_offset()          # position within the current block
    phys = seq.block_table[-1]

    # Write K and V into the physical block at the correct slot
    kv_store[phys, 0, :, slot, :] = k      # K: [H_KV, D]
    kv_store[phys, 1, :, slot, :] = v

    seq.seq_len += 1


# ── Gather K/V for attention ──────────────────────────────────────────────────
def gather_kv(seq: SeqState, layer_idx: int) -> tuple:
    """
    Gather all stored K/V for a sequence from (possibly non-contiguous)
    physical blocks into a contiguous buffer for attention.

    Returns:
      k_full: [H_KV, seq_len, D]
      v_full: [H_KV, seq_len, D]

    Real vLLM: the CUDA kernel reads block_table entries per-warp and fetches
    K/V directly from kv_store without this gather copy.
    """
    num_blocks = len(seq.block_table)
    chunks_k, chunks_v = [], []

    for i, phys in enumerate(seq.block_table):
        # Last block may be partially filled
        tokens_in_block = (
            seq.seq_len - i * BLOCK_SIZE
            if i == num_blocks - 1
            else BLOCK_SIZE
        )
        chunks_k.append(kv_store[phys, 0, :, :tokens_in_block, :])   # [H_KV, t, D]
        chunks_v.append(kv_store[phys, 1, :, :tokens_in_block, :])

    k_full = torch.cat(chunks_k, dim=1)   # [H_KV, seq_len, D]
    v_full = torch.cat(chunks_v, dim=1)
    return k_full, v_full


# ── GQA Attention ─────────────────────────────────────────────────────────────
def gqa_attention(q, k, v, is_causal: bool):
    """
    q : [1, H,    S_q, D]   (batch=1 per sequence in this demo)
    k : [1, H_KV, S_k, D]
    v : [1, H_KV, S_k, D]
    Returns [1, H, S_q, D]
    """
    S_k = k.shape[2]
    if G > 1:
        k = k.unsqueeze(2).expand(-1, -1, G, -1, -1).reshape(1, H, S_k, D)
        v = v.unsqueeze(2).expand(-1, -1, G, -1, -1).reshape(1, H, S_k, D)

    scores = torch.matmul(q, k.transpose(-2, -1)) / scale   # [1, H, S_q, S_k]

    if is_causal and q.shape[2] > 1:
        S_q = q.shape[2]
        mask = torch.ones(S_q, S_k, dtype=torch.bool).tril(diagonal=S_k - S_q)
        scores = scores.masked_fill(~mask, float("-inf"))

    return torch.matmul(F.softmax(scores, dim=-1), v)        # [1, H, S_q, D]


# ── Forward with Paged KV ─────────────────────────────────────────────────────
def paged_forward_token(h: torch.Tensor, seq: SeqState,
                        layer_idx: int, alloc: BlockAllocator) -> torch.Tensor:
    """
    h   : [1, 1, model_dim]  — embedding of one new token
    seq : paging state for this sequence

    1. Project to Q, K, V
    2. Append K, V to paged store (allocate block if needed)
    3. Gather full K/V history
    4. Attend and project output
    """
    w = layers[layer_idx]
    q = (h @ w["W_q"].T).view(1, 1, H,    D).transpose(1, 2)   # [1, H,    1, D]
    k = (h @ w["W_k"].T).view(1, 1, H_KV, D).transpose(1, 2)   # [1, H_KV, 1, D]
    v = (h @ w["W_v"].T).view(1, 1, H_KV, D).transpose(1, 2)

    # ── Write to paged store ──────────────────────────────────────────────────
    append_token_kv(seq, k[0, :, 0, :], v[0, :, 0, :], layer_idx, alloc)

    # ── Gather full history and attend ────────────────────────────────────────
    k_full, v_full = gather_kv(seq, layer_idx)
    k_full = k_full.unsqueeze(0)    # [1, H_KV, seq_len, D]
    v_full = v_full.unsqueeze(0)

    out = gqa_attention(q, k_full, v_full, is_causal=False)   # single query, no mask needed
    return (out.transpose(1, 2).reshape(1, 1, model_dim)) @ w["W_o"].T


def paged_forward_prefill(h: torch.Tensor, seq: SeqState,
                          layer_idx: int, alloc: BlockAllocator) -> torch.Tensor:
    """
    h : [1, S_p, model_dim]  — all prompt tokens

    Process all tokens in parallel (like training), then store each token's
    K/V into paged blocks sequentially.
    The attention itself is over the prefill batch (full causal mask).
    K/V are written token-by-token into the paged store.
    """
    _, S_p, _ = h.shape
    w = layers[layer_idx]

    q = (h @ w["W_q"].T).view(1, S_p, H,    D).transpose(1, 2)  # [1, H,    S_p, D]
    k = (h @ w["W_k"].T).view(1, S_p, H_KV, D).transpose(1, 2)  # [1, H_KV, S_p, D]
    v = (h @ w["W_v"].T).view(1, S_p, H_KV, D).transpose(1, 2)

    # Write K/V token-by-token into paged store
    for t in range(S_p):
        append_token_kv(seq, k[0, :, t, :], v[0, :, t, :], layer_idx, alloc)

    # Gather and run full causal attention over S_p tokens
    k_full, v_full = gather_kv(seq, layer_idx)
    k_full = k_full.unsqueeze(0)
    v_full = v_full.unsqueeze(0)

    out = gqa_attention(q, k_full, v_full, is_causal=True)
    return (out.transpose(1, 2).reshape(1, S_p, model_dim)) @ w["W_o"].T


# ── Reference: contiguous KV cache (from kv_cache.py) ────────────────────────
def attn_no_cache(h, layer_idx):
    """Full causal attention, no paging — ground truth."""
    Bs, Sq, _ = h.shape
    w = layers[layer_idx]
    q = (h @ w["W_q"].T).view(Bs, Sq, H,    D).transpose(1, 2)
    k = (h @ w["W_k"].T).view(Bs, Sq, H_KV, D).transpose(1, 2)
    v = (h @ w["W_v"].T).view(Bs, Sq, H_KV, D).transpose(1, 2)

    S_k = k.shape[2]
    if G > 1:
        k = k.unsqueeze(2).expand(-1, -1, G, -1, -1).reshape(Bs, H, S_k, D)
        v = v.unsqueeze(2).expand(-1, -1, G, -1, -1).reshape(Bs, H, S_k, D)

    scores = torch.matmul(q, k.transpose(-2, -1)) / scale
    mask = torch.ones(Sq, S_k, dtype=torch.bool).tril()
    scores = scores.masked_fill(~mask, float("-inf"))
    out = torch.matmul(F.softmax(scores, dim=-1), v)
    return (out.transpose(1, 2).reshape(Bs, Sq, model_dim)) @ w["W_o"].T


# ═══════════════════════════════════════════════════════════════════════════════
#  DEMO — two sequences with different lengths
# ═══════════════════════════════════════════════════════════════════════════════
S_p   = 6    # prompt length
S_new = 3    # decode steps

# Build input sequences (batch=1 each, processed independently)
X_seq0 = torch.randn(1, S_p + S_new, model_dim, dtype=dtype)
X_seq1 = torch.randn(1, S_p + S_new, model_dim, dtype=dtype)

print(f"Config: H={H}  H_KV={H_KV}  D={D}  L={L}")
print(f"        BLOCK_SIZE={BLOCK_SIZE}  NUM_PHY_BLOCKS={NUM_PHY_BLOCKS}")
print(f"        S_prompt={S_p}  S_decode={S_new}")
print(f"        Physical blocks free at start: {allocator.num_free()}")

# ── Reference ─────────────────────────────────────────────────────────────────
def run_reference(X):
    h = X.clone()
    for l in range(L):
        h = attn_no_cache(h, l)
    return h

ref0 = run_reference(X_seq0)   # [1, S_p+S_new, model_dim]
ref1 = run_reference(X_seq1)

# ── Paged: Sequence 0 ─────────────────────────────────────────────────────────
# Each sequence has one SeqState per layer (the block_table tracks per-layer KV)
# For simplicity we use a flat SeqState and encode layer as a block_table offset.
# In vLLM, each layer has its own block_table / physical KV store partition.
# Here we use a SeqState per (seq, layer) tuple.

states_0 = [SeqState() for _ in range(L)]
states_1 = [SeqState() for _ in range(L)]

# Prefill: sequence 0
h0 = X_seq0[:, :S_p, :].clone()
for l in range(L):
    h0 = paged_forward_prefill(h0, states_0[l], l, allocator)

paged_prefill_out0 = h0.clone()   # [1, S_p, model_dim]

# Prefill: sequence 1
h1 = X_seq1[:, :S_p, :].clone()
for l in range(L):
    h1 = paged_forward_prefill(h1, states_1[l], l, allocator)

print(f"\n  After prefill (2 sequences × {S_p} tokens each):")
print(f"    Physical blocks used: {NUM_PHY_BLOCKS - allocator.num_free()}")
print(f"    Physical blocks free: {allocator.num_free()}")

# Decode: interleave both sequences for S_new steps
decode_outs_0, decode_outs_1 = [], []

for step in range(S_new):
    # Decode sequence 0
    tok0 = X_seq0[:, S_p + step : S_p + step + 1, :].clone()
    for l in range(L):
        tok0 = paged_forward_token(tok0, states_0[l], l, allocator)
    decode_outs_0.append(tok0.clone())

    # Decode sequence 1
    tok1 = X_seq1[:, S_p + step : S_p + step + 1, :].clone()
    for l in range(L):
        tok1 = paged_forward_token(tok1, states_1[l], l, allocator)
    decode_outs_1.append(tok1.clone())

print(f"\n  After decode ({S_new} steps, both sequences):")
print(f"    Physical blocks used: {NUM_PHY_BLOCKS - allocator.num_free()}")
print(f"    Physical blocks free: {allocator.num_free()}")

# ── Verify correctness ────────────────────────────────────────────────────────
# Compare paged decode output vs reference at each decode step
errors_0 = [
    (decode_outs_0[i][:, 0, :] - ref0[:, S_p + i, :]).abs().max().item()
    for i in range(S_new)
]
errors_1 = [
    (decode_outs_1[i][:, 0, :] - ref1[:, S_p + i, :]).abs().max().item()
    for i in range(S_new)
]
print(f"\n  Correctness (max |paged - reference|):")
print(f"    seq0 decode steps: {[f'{e:.1e}' for e in errors_0]}  (should be ~0)")
print(f"    seq1 decode steps: {[f'{e:.1e}' for e in errors_1]}  (should be ~0)")

# ── Free sequence 0's blocks (simulates sequence finishing) ───────────────────
for s in states_0:
    for phys in s.block_table:
        allocator.free(phys)

print(f"\n  After sequence 0 finishes and its blocks are freed:")
print(f"    Physical blocks free: {allocator.num_free()}  (immediately reusable)")

# ═══════════════════════════════════════════════════════════════════════════════
#  MEMORY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
S_total = S_p + S_new

print("\n── Block Table Layout ───────────────────────────────────────────────────────")
print(f"  Sequence 0 (after {S_total} tokens, {L} layers):")
for l, s in enumerate(states_0):
    blocks_used = len(s.block_table)
    last_fill   = s.seq_len % BLOCK_SIZE or BLOCK_SIZE
    wasted      = BLOCK_SIZE - last_fill
    print(f"    layer {l}: phys_blocks={s.block_table}  last_block_fill={last_fill}/{BLOCK_SIZE}  wasted_slots={wasted}")

print(f"\n  BLOCK_SIZE={BLOCK_SIZE}: max waste = {BLOCK_SIZE-1} tokens per seq per layer,")
print(f"  regardless of max_seq_len (vs contiguous which wastes S_max - S_actual)")

print("\n── Fragmentation Comparison ─────────────────────────────────────────────────")
S_actual = S_total
S_max_contiguous = 512   # typical pre-allocation

contig_alloc  = S_max_contiguous * H_KV * D * 2 * L * 2    # 2 seqs, K+V, float32=4 actually use 2 for elem
paged_alloc   = math.ceil(S_actual / BLOCK_SIZE) * BLOCK_SIZE * H_KV * D * 4 * L * 2   # per seq * 2 seqs
wasted_contig = (S_max_contiguous - S_actual) * H_KV * D * 4 * L * 2
wasted_paged  = (BLOCK_SIZE - (S_actual % BLOCK_SIZE or BLOCK_SIZE)) * H_KV * D * 4 * L * 2

print(f"  Scenario: 2 sequences × {S_actual} actual tokens, max_seq_len={S_max_contiguous}")
print(f"  Contiguous:")
print(f"    Allocated : {contig_alloc/1024:.1f} KB")
print(f"    Wasted    : {wasted_contig/1024:.1f} KB  ({100*wasted_contig//contig_alloc}%)")
print(f"  Paged (block_size={BLOCK_SIZE}):")
print(f"    Allocated : {paged_alloc/1024:.1f} KB  (only used blocks × block_size)")
print(f"    Wasted    : {wasted_paged/1024:.1f} KB  (last partial block only, ≤ block_size-1 tokens)")

print("\n── Prefix Caching ───────────────────────────────────────────────────────────")
print("  Mechanism:")
print("    1. Hash the token IDs of a completed prefix")
print("    2. Store: prefix_hash → [phys_block_0, phys_block_1, ...]  (block table)")
print("    3. New request with same prefix → copy those physical block IDs into")
print("       its own block_table (no memory copy; both sequences read same pages)")
print("    4. First write (decode) → copy-on-write: allocate new block, copy, modify")
print(f"\n  Example: system prompt of 1024 tokens uses {1024 // BLOCK_SIZE} blocks")
print(f"           shared across 100 concurrent requests → 99× memory saving on prefix")

print("\n── vLLM PagedAttention CUDA Kernel (conceptual) ─────────────────────────────")
print("  This demo:    gather K/V into contiguous buffer → run attention")
print("  vLLM kernel:  each thread block handles one query head × one KV block")
print("                reads block_table[seq_id][block_idx] → physical_block_id")
print("                loads K/V directly from kv_store[phys_id] — no gather copy")
print("                accumulates attention scores across blocks in a streaming loop")
print("  Result:       same O(S) KV reads but zero extra memory for the contiguous buffer")
