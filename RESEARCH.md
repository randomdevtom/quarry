# CPU MoE engine — research notes (2026-11-21)

Goal: run MoE models that don't fit in RAM, CPU-only, much faster than llama.cpp.
Target model: Qwen3-Coder-30B-A3B-Instruct Q3_K_S (13.3GB) at
`~/models/qwen3-coder-30b/`. The laptop has 11GB RAM.

## Hardware (measured)
- CPU: i3-1005G1, 2 cores / 4 threads, AVX-512 with VNNI, 4MB L3
- RAM read bandwidth: 27.5 GB/s
- NVMe random 2.6MB reads (O_DIRECT): 0.95 GB/s, 2.7ms each. Parallel reads don't help.
- Our own 4-bit AVX-512 VNNI matvec: 23.1 GB/s (84% of RAM bandwidth), correct
  vs a reference implementation (`research/matvec_bench.c`)

## Baselines
- llama.cpp, Qwen3-Coder-30B-A3B Q3_K_S (bigger than RAM, via mmap): 2.29 tok/s
- llama.cpp, LFM2.5-8B-A1B Q4_K_M (fits in RAM): 14.6 tok/s, against a ~32 tok/s
  bandwidth ceiling (0.85GB read per token)

## Model layout (Qwen3-Coder-30B-A3B Q3_K_S)
- 48 layers x 128 experts, 8 used per token; 2.03MB per expert
- experts 12.46GB total, dense (non-expert) 0.70GB, embeddings 0.13GB
- per token: 384 experts = 0.78GB of experts, plus the dense weights

## Routing trace (`research/trace.c`, 26 prompt + 200 generated tokens, coding task)
- consecutive-token expert reuse: 35%
- 5061 of 6144 experts touched in 200 tokens

## Cache simulation (`research/cache_sim.py`), hit rate on generated tokens
| expert RAM | share of experts | LRU | LFU | oracle | est tok/s (no overlap / full overlap) |
|---|---|---|---|---|---|
| 4G | 32% | 75% | 72% | 89% | 3.4 / 4.4 |
| 5G | 40% | 82% | 80% | 92% | 4.4 / 6.2 |
| 6G | 48% | 87% | 86% | 95% | 5.7 / 8.9 |
| 7G | 56% | 91% | 91% | 96% | 7.2 / 13.3 |

Conclusion: 3-4x over llama.cpp looks realistic with about 6GB of expert cache.
LRU beats LFU. The oracle (perfect foresight) gap suggests that predicting
the next layer's experts and prefetching them could add more.

## Engine design (planned)
1. GGUF loader; dense weights fully in RAM, experts read on demand with
   O_DIRECT, 2MB at a time
2. Expert cache: LRU, fixed-size slab of expert slots
3. Prefetch: run the next layer's router early on the current hidden state, and
   issue SSD reads while the current layer computes (an I/O thread)
4. Kernels: Q3_K / Q4_K / Q6_K / Q8_0 dot products with AVX-512 VNNI
5. Qwen3-MoE forward pass: RMSNorm, GQA attention + RoPE + QK-norm, top-8
   softmax router, SwiGLU experts, KV cache
6. Tokenizer (from GGUF), OpenAI-compatible server
Validate each stage against llama.cpp logits.

## Open questions
- Hit rate over long sessions (1000+ tokens, mixed tasks)
- Next-layer router prediction accuracy
- Whether the 2 physical cores can do compute and I/O at the same time

## Build progress (Quarry)
- [x] Stage 1: GGUF loader (`quarry/src/gguf.c`, `quarry-info`). Parses all 579
  tensors of Qwen3-Coder-30B-A3B. data_offset 5973152 matches gguf-py, and the
  last tensor ends exactly at the end of the file. Split: 0.83GB dense / 12.46GB experts.
  Types used: Q3_K (almost all), Q6_K (output), F32 (norms, router).
- [x] Stage 2: Q3_K/Q6_K kernels. Dequant is bit-exact against gguf-py. AVX-512 VNNI
  dots: Q3_K 13.9 GB/s (from 6.3), Q6_K 22 GB/s.
- [x] Stage 3: qwen3moe forward pass matches llama.cpp logits: cosine 0.9993,
  identical top-5 (tests/test_forward.c, tests/ref_logits.c)
- [x] Stage 4: expert cache (LRU slab, O_DIRECT, 2 I/O threads, demand loads
  jump the queue, next-layer router prediction + prefetch, pinning).
  Results on the same prompt ("LRU cache" task), 128 generated tokens:
  - llama.cpp (llama-completion, page cache warm): 3.75 tok/s
  - Quarry, 7GB cache, OMP_WAIT_POLICY=PASSIVE: **6.08 tok/s (1.62x)**
  - expert hit rate at layer start 96%, prediction accuracy 82%,
    71 MB/token from SSD, only 5% of time blocked on I/O
  Profile per token: attention matvecs ~36ms, experts ~95ms (compute-bound),
  router 10ms, output 15ms. Next speedups are on the compute side.
- [ ] Stage 5: tokenizer, generation loop, OpenAI-compatible server
