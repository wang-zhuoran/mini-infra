# mini-infra

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)

Minimal, single-file implementations of core LLM training infrastructure.
Each file runs on real GPUs via `torchrun`, no abstractions, no magic—just PyTorch + NCCL.

## Demos

| File | Concept | Key Insight | Run |
|------|---------|-------------|-----|
| `tp_linear.py` | Tensor Parallelism (Megatron-style) | Col-parallel→Row-parallel = only 1 AllReduce per MLP block | `torchrun --nproc_per_node=N tp_linear.py` |
| `pp_schedule.py` | Pipeline Parallelism: GPipe vs 1F1B | Same bubble `(PP-1)/(M+PP-1)`, but 1F1B memory = O(PP) not O(M) | `torchrun --nproc_per_node=N pp_schedule.py` |
| `ppo_rlhf.py` | PPO for RLHF (4-model setup) | Clipped surrogate + KL penalty + GAE-λ；为什么GRPO能去掉Critic | `python ppo_rlhf.py` |
| `grpo.py` | GRPO (DeepSeek-R1 style) | Group mean替代Critic；G=8的方差分析；rule-based reward的优势 | `python grpo.py` |
| `dp_zero.py` | ZeRO-1/2/3 Data Parallel | 每个stage shard什么；ZeRO-3通信3×DDP但内存÷DP | `torchrun --nproc_per_node=N dp_zero.py` |
| `sequence_parallel.py` | Sequence Parallelism (Megatron SP) | AllGather+ReduceScatter=AllReduce字节数；非TP算子激活内存÷TP | `torchrun --nproc_per_node=N sequence_parallel.py` |
| `dpo.py` | Direct Preference Optimization | Z(x)为什么抵消；implicit reward；DPO vs PPO的distribution shift问题 | `python dpo.py` |

## Roadmap

**Tier 1 — Distributed Training Core**
- [x] `tp_linear.py` — Tensor Parallelism (col/row parallel, AllReduce, Sequence Parallel)
- [x] `pp_schedule.py` — Pipeline Parallelism (GPipe bubble, 1F1B schedule)
- [x] `dp_zero.py` — ZeRO-1/2/3 (optimizer state / grad / param sharding)
- [ ] `nccl_collectives.py` — AllReduce, AllGather, ReduceScatter, P2P
- [ ] `mixed_precision.py` — BF16/FP16, dynamic loss scaling, master weights

**Tier 2 — Advanced Training**
- [x] `sequence_parallel.py` — Megatron SP (AllGather + ReduceScatter replaces AllReduce)
- [ ] `activation_checkpoint.py` — Gradient checkpointing, recompute vs memory trade-off
- [ ] `3d_parallel.py` — TP + PP + DP combined process groups

**Tier 3 — Post-training / Alignment**
- [x] `ppo_rlhf.py` — PPO for LLM (actor/critic/ref model, clip ratio, advantage)
- [x] `dpo.py` — Direct Preference Optimization
- [x] `grpo.py` — Group Relative Policy Optimization (DeepSeek-R1 style)
- [ ] `rollout.py` — Generation loop inside training (vLLM integration point)

**Tier 4 — Inference**
- [ ] `kv_cache.py` — KV cache layout, memory math
- [ ] `paged_attention.py` — Block table, virtual memory for KV
- [ ] `speculative_decoding.py` — Draft + verify, acceptance rate
