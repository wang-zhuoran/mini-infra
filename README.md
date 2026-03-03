# mini-infra

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)
![Status](https://img.shields.io/badge/Status-Interview%20Ready-brightgreen.svg)

Minimal, self-contained implementations of core AI training infrastructure concepts. 
Inspired by [minimind](https://github.com/jingyaogong/minimind)'s philosophy: **understand by building**.

Each demo is &lt;100 lines, single-file, and interview-optimized. No abstractions, no magic—just PyTorch and NCCL.

## 🎯 Demos

| File | Concept | Lines | Key Insight |
|------|---------|-------|-------------|
| `demo_tensor_parallel.py` | Tensor Parallelism (Megatron-style) | ~80 | Column/Row parallel linear layers, All-Gather communication cost |
| `demo_rlhf_ppo.py` | PPO for LLM alignment | ~100 | Clipped surrogate objective, KL penalty, advantage estimation |
| `demo_zero_optimizer.py` | ZeRO-1 sharded optimizer | ~60 | Partition optimizer states across data parallel ranks |
| `demo_pipeline_bubble.py` | Pipeline parallelism bubble | ~40 | GPipe schedule, idle time calculation |

## 🚀 Quick Start

```bash
# Tensor Parallel demo (simulates 2 GPUs on CPU)
python demo_tensor_parallel.py

# RLHF PPO demo (random data, no model download)
python demo_rlhf_ppo.py
```
