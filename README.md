# Quarry

CPU-only inference for Mixture-of-Experts models that are **bigger than your RAM**.

Strata-style engines split a MoE model between GPU and RAM. Quarry does the same
between **RAM and SSD**: the dense weights stay in RAM, and the experts are streamed
from the SSD into an expert cache. Each layer's router is applied one layer early to
predict which experts come next, and they're prefetched while the current layer computes.

## Results

Qwen3-Coder-30B-A3B-Instruct (Q3_K_S, 13.3GB) on a laptop with an i3-1005G1
(2 cores), **11GB RAM** and a ~0.95GB/s NVMe. Same prompt, 128 generated tokens:

| engine | tok/s |
|---|---|
| llama.cpp (mmap, page cache warm) | 3.75 |
| **Quarry** (7GB expert cache) | **6.08** (1.62x) |

- expert cache hit rate: 96% when a layer starts
- next-layer expert prediction accuracy: 82%
- 71MB read from the SSD per token; 5% of the time is spent waiting on I/O
- the output matches llama.cpp logits (cosine 0.999, identical top-5)

## Build

Needs gcc with AVX-512 VNNI support, OpenMP, and libllama (used only for the tokenizer).

```sh
cd quarry && make
```

## Use

```sh
# chat in the terminal
QUARRY_CACHE_GB=6.5 ./build/quarry run -m Qwen3-Coder-30B-A3B-Instruct-Q3_K_S.gguf

# OpenAI-compatible server (streaming, prompt-prefix reuse across turns)
QUARRY_CACHE_GB=6.5 ./build/quarry serve -m model.gguf --port 8090
# -> http://127.0.0.1:8090/v1/chat/completions
```

Set `QUARRY_CACHE_GB` to most of your free RAM minus ~1.5GB for the dense weights and
the KV cache. A bigger cache means fewer SSD reads.

## How it works

- `src/qgguf.c`: GGUF reader. Dense tensors are loaded into RAM, experts are addressed by file offset.
- `src/quants.c`: Q3_K / Q6_K kernels (AVX-512 VNNI). Dequant is bit-exact
  against gguf-py. Matvec runs at 13.9GB/s (Q3_K) and 22GB/s (Q6_K) on 2 cores.
- `src/experts.c`: the expert cache. Slots are evicted LRU; reads use O_DIRECT, so
  Quarry, not the page cache, decides what stays in RAM. Demand loads jump the I/O
  queue, prefetches go last, and experts in use are pinned.
- `src/model.c`: qwen3moe forward pass. Experts that are already cached are computed
  first, as one batch, while missing ones load.
- `src/main.c`: CLI and HTTP server.

`RESEARCH.md` has the measurements and the routing-trace and cache-simulation work
behind the design.

## Limits (v0.1)

- architecture: qwen3moe only; quant types: Q3_K, Q6_K, F32
- one sequence at a time; the prompt is processed token by token (no batched prefill yet)
- tokenizer comes from libllama (vocab-only load)
- KV cache is fp32
