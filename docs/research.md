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

## Generation speed work (2026-11-22)

Qwen3-Coder generation went from ~5.9 to ~7.0-7.3 tok/s (exact) on the laptop above.

What helped:
- Q3_K scale decoding wrote 4 x u32 to the stack and reloaded them as one 128-bit
  value: a store-forwarding stall on every block. Doing it in registers doubled the
  single-thread kernel speed (6.3 -> 12.8 GB/s); 4 threads from RAM now ~17-21 GB/s.
- Each expert is 3 file pieces of ~0.7MB (gate/up/down). Reading them one after another
  ran at ~1.0 GB/s; submitting all three with Linux AIO runs at ~1.5 GB/s (2MB reads reach
  ~1.7 GB/s).
- The router's F32 values are exact bf16, so it is stored as bf16: half the reads,
  bit-identical results.
- RoPE cos/sin once per token instead of once per head and layer; vectorized attention
  core and SiLU.

What didn't (measured, reverted):
- Rearranging Q3_K high bits into lane masks (+7% compute, nothing from RAM), a 4-row
  kernel (exact but slower: more streams per thread), software prefetch, OpenMP
  wait policies / spin counts / pinning / dynamic schedules, a bigger cache (7GB: fewer
  SSD waits but memory pressure cancels it).
- Speculative decoding with n-gram drafts: verifying 8 tokens in one batch touches the
  union of their experts (~5x the bytes), so even with every draft accepted it is slower
  than decoding one token at a time. Not a fit for MoE on this hardware.

Where the time goes (exact, ~140ms/token): experts ~60ms compute + ~15ms SSD waits,
attention projections ~29ms, output layer ~14ms, router ~5ms, rest ~10ms. RAM read speed
is ~22GB/s, and ~1.4GB is read per token, so a perfect engine would reach ~15 tok/s; the
fixed (non-expert) part alone is ~58ms. Using 6 of the 8 routed experts (`--experts 6`)
cuts expert bytes and SSD reads by 25% and kept the same HumanEval subset score (19/21).

## LFM2.5-8B-A1B (2026-11-24)

Q4_K_M, 5.2GB, fits in RAM: a compute and RAM-speed problem, not an SSD one. Per token
~1.0GB is read: experts ~0.58GB (gate/up Q4_K, down Q4_K or Q6_K), short-conv projections
0.17GB, output layer (tied to the Q6_K embedding) 0.21GB, the rest ~0.08GB.

Generation went 4.3 -> 14.1 tok/s (llama.cpp at the same time: 12.4):
- Q4_K scale unpack stored 4 x u32 and reloaded them as one vector: the same store-forwarding
  stall as Q3_K had. Done in registers: ~1.5x single-thread on Q4_K/Q5_K.
- Q6_K rewritten at 512 bits (was 256-bit with a `mullo` per 32 values): ~1.3x.
- One OpenMP region per stage for all ready experts (was two per expert).
- `Q4_KR`: quants reordered inside each block so int16 lane k belongs to sub-block k/4 in
  every 64-value chunk. One scale permute and one `madd` per block instead of four; bit-exact
  per row. +9% generation.
- Output layer from a 4-bit copy (candidates) + exact top-64 rows: 12.5 -> 10ms/token.
- Under the browser load at the time (load average 6-9), dynamic OpenMP scheduling helped a
  little; ACTIVE wait policy with 4 threads halved the speed (3.5 tok/s).

Measured no gain: LTO, a 2-row decode kernel (shared activation loads, two chains), 3 vs 4
threads. Serial work outside the matvecs is ~2ms/token, and the matvec regions stream at
~14.5GB/s against ~21GB/s measured RAM bandwidth (16.6GB/s from one thread).

Prompts went 14 -> 38 tok/s (llama.cpp pp512: 35.7):
- the int8 requantized rows of the Qwen path (one scale per 256 values) were too lossy here:
  logits cosine 0.975. Float rows are exact but compute-bound (~20 tok/s).
- exact integer batched kernel: u8 weights as stored x int8 activations with `dpbusd`. In
  Q4_KR every int32 lane of every chunk is in one sub-block, so all four chunks accumulate
  in one register and the scale applies once per block. Q6_K applies scales per chunk.
  2 rows x 4 tokens per tile, specialized at compile time (+11%). ~118 GMAC/s on Q4_KR,
  ~70 on Q6_K, 4 threads. On Ice Lake client both 512-bit VNNI and FMA issue only on port 0,
  which bounds it.
