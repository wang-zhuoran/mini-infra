"""
kv_cache.py — KV Cache for Autoregressive Decoding

Concept: Autoregressive LLM decoding generates one token at a time. Without KV
         cache, each step re-computes K and V for every prior token from scratch,
         giving O(S²·D) total compute for a sequence of S tokens. KV cache stores
         past K/V tensors so each decode step only computes KV for the ONE new
         token — O(S·D) total, S× cheaper.

Key Insight — Prefill vs Decode are fundamentally different phases:

  Prefill  : process all S_prompt tokens in PARALLEL (identical to training fwd)
             Uses causal mask. Compute-bound: matmuls dominate.

  Decode   : process ONE new token per step, Q attends over all S cached K/V.
             Memory-bandwidth-bound: GPU spends most time reading the KV cache,
             not doing arithmetic. Arithmetic intensity ≈ H/H_KV ops/byte.

  This asymmetry is why vLLM / TensorRT-LLM use separate CUDA kernels for each.

Key Insight — KV cache memory formula:

  bytes = 2 (K+V) × L × B × H_KV × S_max × D × elem_bytes

  LLaMA-3 70B  (H=64, H_KV=8, D=128, L=80, BF16=2 bytes):
    Per token, per batch item: 2 × 80 × 8 × 128 × 2 = 327,680 bytes ≈ 320 KB
    batch=1,  8K ctx  →  2.5 GB   (fits on one A100)
    batch=32, 8K ctx  →  81.9 GB  (needs paged attention or tensor parallelism)

Key Insight — GQA / MQA trades KV cache size for quality:

  MHA (H_KV = H)    : baseline quality and memory
  GQA (H_KV = H/G)  : KV cache ÷ G, minor quality loss  (LLaMA-3, Mistral, Qwen)
  MQA (H_KV = 1)    : KV cache ÷ H, stronger quality loss

  Implementation: expand K/V from H_KV heads → H heads by repeating each KV
  head G times before attention (no extra parameters, no extra communication).

Key Insight — Paged Attention (vLLM) solves KV cache fragmentation:

  Problem  : pre-allocated contiguous KV tensors waste memory for variable-length
             sequences; freed slots are not reusable (external fragmentation).
  Solution : split KV cache into fixed-size blocks (e.g. 16 tokens/block).
             A block table maps logical positions → physical block addresses.
             Like OS virtual memory, but for KV cache.
  Result   : near-zero fragmentation; enables memory sharing (prefix caching,
             beam search), high-throughput continuous batching.

Interview questions this file answers:
  Q: What is KV cache and why does it exist?
  Q: What is the KV cache memory formula? Give a concrete example.
  Q: What is the difference between prefill and decode phases?
  Q: Why is decode memory-bandwidth bound? What is arithmetic intensity?
  Q: What is GQA/MQA? How does it reduce KV cache memory?
  Q: What problem does paged attention solve and how?

Run: python kv_cache.py
"""

import torch
import torch.nn.functional as F
import math

# ── Config ────────────────────────────────────────────────────────────────────
B     = 2     # batch size
H     = 16    # Q heads
H_KV  = 4     # KV heads  (GQA: H_KV=4 means G=4 queries share each KV head)
D     = 64    # head dimension  →  model_dim = H × D = 1024
L     = 4     # transformer layers
S_p   = 32    # prompt length (prefill)

G         = H // H_KV   # queries per KV head
model_dim = H * D
kv_dim    = H_KV * D
S_total   = S_p + 1     # prefill + 1 decode step (for correctness check)
dtype     = torch.float32   # float32 for exact numerical verification

assert H % H_KV == 0, "H must be divisible by H_KV"
torch.manual_seed(42)
scale = math.sqrt(D)

# ── Weights: one set per layer ────────────────────────────────────────────────
# W_q: [model_dim, model_dim]  (Q projects to all H heads)
# W_k: [model_dim, kv_dim]     (K projects to only H_KV heads — the GQA saving)
# W_v: [model_dim, kv_dim]     (V same)
# W_o: [model_dim, model_dim]
layers = [{
    "W_q": torch.randn(model_dim, model_dim) * 0.02,
    "W_k": torch.randn(kv_dim,    model_dim) * 0.02,
    "W_v": torch.randn(kv_dim,    model_dim) * 0.02,
    "W_o": torch.randn(model_dim, model_dim) * 0.02,
} for _ in range(L)]

# ── KV Cache ──────────────────────────────────────────────────────────────────
# Pre-allocate for S_total positions.  Shape: [L, 2, B, H_KV, S_total, D]
#
#   [layer_idx, K/V, batch, kv_head, seq_pos, head_dim]
#                ↑                   ↑
#               0=K, 1=V         grows as we decode
#
# Real systems (vLLM) use a block table over fixed-size pages instead of a
# single contiguous tensor, enabling memory reuse across variable-length seqs.
#
kv_cache = torch.zeros(L, 2, B, H_KV, S_total, D, dtype=dtype)


# ── Attention kernel (supports GQA) ──────────────────────────────────────────
def gqa_attention(q, k, v, is_causal: bool):
    """
    q : [B, H,    S_q, D]
    k : [B, H_KV, S_k, D]   ← H_KV ≤ H
    v : [B, H_KV, S_k, D]
    Returns [B, H, S_q, D]

    GQA expand: each KV head serves G = H // H_KV query heads.
    K/V are expanded by repeating each head G times along the head axis.
    No extra weights; just a view + expand (no memory copy needed in practice).
    """
    Bq, _, S_k, _ = k.shape

    # [B, H_KV, S_k, D] → [B, H_KV, G, S_k, D] → [B, H, S_k, D]
    if G > 1:
        k = k.unsqueeze(2).expand(-1, -1, G, -1, -1).reshape(Bq, H, S_k, D)
        v = v.unsqueeze(2).expand(-1, -1, G, -1, -1).reshape(Bq, H, S_k, D)

    # [B, H, S_q, S_k]
    scores = torch.matmul(q, k.transpose(-2, -1)) / scale

    if is_causal:
        S_q = q.shape[2]
        if S_q > 1:
            # tril offset = S_k - S_q so that query at position i (0-indexed
            # within this chunk) can attend to cache positions 0 .. start+i.
            # In decode S_q=1 so we skip the mask entirely — the single query
            # already attends to the full cache which is always valid.
            mask = torch.ones(S_q, S_k, dtype=torch.bool, device=q.device).tril(
                diagonal=S_k - S_q
            )
            scores = scores.masked_fill(~mask, float("-inf"))

    attn_w = F.softmax(scores, dim=-1)
    return torch.matmul(attn_w, v)   # [B, H, S_q, D]


# ── Reference path: full causal attention, no cache ──────────────────────────
def attn_no_cache(h, layer_idx: int):
    """Standard full-sequence causal attention. Used only for correctness check."""
    Bs, Sq, _ = h.shape
    w = layers[layer_idx]
    q = (h @ w["W_q"].T).view(Bs, Sq, H,    D).transpose(1, 2)
    k = (h @ w["W_k"].T).view(Bs, Sq, H_KV, D).transpose(1, 2)
    v = (h @ w["W_v"].T).view(Bs, Sq, H_KV, D).transpose(1, 2)
    out = gqa_attention(q, k, v, is_causal=True)
    return (out.transpose(1, 2).reshape(Bs, Sq, model_dim)) @ w["W_o"].T


# ── Cached path: write new K/V, read full history ────────────────────────────
def attn_with_cache(h, layer_idx: int, start_pos: int):
    """
    h         : [B, S_q, model_dim]   (S_q = S_p in prefill, 1 in decode)
    start_pos : index of h's first token in the overall sequence

    Steps:
      1. Project h → Q, K, V
      2. Write K, V into cache at [start_pos : start_pos + S_q]
      3. Read full K/V history [0 : start_pos + S_q] from cache
      4. Attention: Q attends to full cached K/V
    """
    Bs, Sq, _ = h.shape
    end_pos   = start_pos + Sq
    w         = layers[layer_idx]

    q = (h @ w["W_q"].T).view(Bs, Sq, H,    D).transpose(1, 2)   # [B, H,    S_q, D]
    k = (h @ w["W_k"].T).view(Bs, Sq, H_KV, D).transpose(1, 2)   # [B, H_KV, S_q, D]
    v = (h @ w["W_v"].T).view(Bs, Sq, H_KV, D).transpose(1, 2)   # [B, H_KV, S_q, D]

    # ── Write new K/V to cache ────────────────────────────────────────────────
    kv_cache[layer_idx, 0, :Bs, :, start_pos:end_pos, :] = k
    kv_cache[layer_idx, 1, :Bs, :, start_pos:end_pos, :] = v

    # ── Read full K/V from cache ──────────────────────────────────────────────
    # Prefill : reads own K/V (positions 0..S_p-1), causal mask applied
    # Decode  : reads entire history (positions 0..S_p), no mask needed (S_q=1)
    k_full = kv_cache[layer_idx, 0, :Bs, :, :end_pos, :]   # [B, H_KV, end_pos, D]
    v_full = kv_cache[layer_idx, 1, :Bs, :, :end_pos, :]

    out = gqa_attention(q, k_full, v_full, is_causal=True)
    return (out.transpose(1, 2).reshape(Bs, Sq, model_dim)) @ w["W_o"].T


# ── Input ─────────────────────────────────────────────────────────────────────
X = torch.randn(B, S_total, model_dim, dtype=dtype)

# ═══════════════════════════════════════════════════════════════════════════════
#  REFERENCE — full causal attention over the entire sequence, no cache
# ═══════════════════════════════════════════════════════════════════════════════
h_ref = X.clone()
for l in range(L):
    h_ref = attn_no_cache(h_ref, l)

ref_at_Sp = h_ref[:, S_p, :]   # [B, model_dim] — ground truth at position S_p

# ═══════════════════════════════════════════════════════════════════════════════
#  PREFILL — process all S_p prompt tokens in parallel, populate KV cache
# ═══════════════════════════════════════════════════════════════════════════════
#
#  Identical to a training forward pass: all S_p tokens processed simultaneously.
#  Causal mask ensures token i only attends to tokens 0..i.
#  After this loop, kv_cache[:, :, :, :, 0:S_p, :] is fully populated.
#
h = X[:, :S_p, :].clone()   # [B, S_p, model_dim]
for l in range(L):
    h = attn_with_cache(h, l, start_pos=0)

# h is now [B, S_p, model_dim] — the prefill hidden states

# ═══════════════════════════════════════════════════════════════════════════════
#  DECODE — generate one new token, reading from KV cache
# ═══════════════════════════════════════════════════════════════════════════════
#
#  In production: new_token_embed = embedding(sampled_token_id_from_last_step)
#  Here we use the true X[:, S_p, :] for a controlled correctness check.
#
#  For each layer:
#    1. Compute Q, K, V for the single new token  (tiny: 1 × model_dim matmul)
#    2. Append K, V to the cache at position S_p
#    3. Attend over all S_p+1 cached K/V entries  (reads most of the KV cache)
#
#  Step 3 dominates: reading  2 × H_KV × S × D × 2 bytes of KV
#                   vs doing  2 × H × S × D FLOPs
#  Arithmetic intensity = H / (2 × H_KV) ops/byte — far below GPU roofline.
#
h_new = X[:, S_p:S_p+1, :].clone()   # [B, 1, model_dim]
for l in range(L):
    h_new = attn_with_cache(h_new, l, start_pos=S_p)

cached_at_Sp = h_new[:, 0, :]   # [B, model_dim]

# ═══════════════════════════════════════════════════════════════════════════════
#  VERIFY
# ═══════════════════════════════════════════════════════════════════════════════
max_err = (cached_at_Sp - ref_at_Sp).abs().max().item()
print(f"Config: B={B}  H={H} (Q-heads)  H_KV={H_KV} (KV-heads)  D={D}  L={L}")
print(f"        GQA group G={G}  model_dim={model_dim}  S_prompt={S_p}")
print(f"\nCorrectness check:")
print(f"  max |cached_decode - reference| = {max_err:.2e}  (should be ~0)")

# ═══════════════════════════════════════════════════════════════════════════════
#  ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
def kv_mem_bytes(L, B, H_KV, S, D, elem_bytes=2):
    return 2 * L * B * H_KV * S * D * elem_bytes

print("\n── KV Cache Memory ──────────────────────────────────────────────────────────")
print("  Formula: 2 × L × B × H_KV × S × D × elem_bytes")
print(f"           (2 = K + V)")

demo_gqa = kv_mem_bytes(L, B, H_KV, S_total, D, elem_bytes=4)   # float32 for this demo
demo_mha = kv_mem_bytes(L, B, H,    S_total, D, elem_bytes=4)
print(f"\n  This demo  GQA (H_KV={H_KV:2d}): {demo_gqa/1024:.1f} KB")
print(f"  This demo  MHA (H_KV={H:2d}): {demo_mha/1024:.1f} KB  ({G}× more)")

print(f"\n  LLaMA-3 70B  (H=64, H_KV=8, D=128, L=80, BF16=2 bytes):")
configs = [
    ("batch=1,  ctx=2K",   1,  2048),
    ("batch=1,  ctx=8K",   1,  8192),
    ("batch=32, ctx=8K",  32,  8192),
    ("batch=1,  ctx=128K", 1, 131072),
]
for label, b, s in configs:
    mb = kv_mem_bytes(80, b, 8, s, 128, elem_bytes=2) / 1024**2
    print(f"    {label:<22}: {mb:8.1f} MB")

print("\n── Prefill vs Decode ────────────────────────────────────────────────────────")
prefill_flops = 2 * S_p * S_p * H * D          # O(S²) QK matmul, dominant term
decode_flops  = 2 * (S_p + 1) * H * D          # O(S) QK matmul for 1 query token
kv_read_bytes = 2 * H_KV * (S_p + 1) * D * 4  # read K and V, float32 here

arith_intensity = decode_flops / kv_read_bytes

print(f"  Prefill : {S_p} tokens in parallel, causal mask, compute-bound")
print(f"            FLOPs ≈ {prefill_flops/1e6:.2f} MFLOPs  per layer  (QK term only)")
print(f"  Decode  : 1 token, reads full KV cache, memory-bandwidth-bound")
print(f"            FLOPs ≈ {decode_flops/1e3:.1f} KFLOPs  per layer")
print(f"            KV read ≈ {kv_read_bytes/1024:.1f} KB  per layer")
print(f"            Arithmetic intensity ≈ {arith_intensity:.1f} FLOPs/byte")
print(f"            (A100 needs ~200 FLOPs/byte to be compute-bound → decode is ~{int(200/max(arith_intensity,0.1))}× below roofline)")

print("\n── GQA / MQA Memory Saving ──────────────────────────────────────────────────")
print(f"  MHA   H_KV = H    = {H:2d} :  1× KV cache  (baseline)")
for g in [2, 4, 8]:
    if H % g == 0:
        hkv = H // g
        pct = 100 // g
        print(f"  GQA   H_KV = H/{g:<2d} = {hkv:2d} : ÷{g}  KV cache  ({pct}% of MHA)   G={g}")
print(f"  MQA   H_KV = 1       :  ÷{H}  KV cache  ({100//H}% of MHA)   G={H}")

print("\n── Paged Attention (vLLM) ───────────────────────────────────────────────────")
print("  Problem : contiguous KV tensor → internal + external fragmentation")
print("            seq finishes → its memory can't be reused for a new seq")
print("  Solution: KV cache split into blocks of K tokens (e.g. block_size=16)")
print("            block_table[seq_id][logical_block] → physical_block_id")
print("            physical blocks allocated/freed on demand, like OS page frames")
print("  Gains   : near-zero KV fragmentation")
print("            copy-on-write for beam search (share prefix blocks)")
print("            prefix caching (share KV of common system prompts)")
print("            continuous batching (pack many short + long seqs together)")
