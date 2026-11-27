// Quarry GPU expert tier (optional; built with `make GPU=1` against a llama.cpp whose ggml has a
// GPU backend: Vulkan, CUDA, ROCm, Metal...).
//
// The most-used experts live in VRAM, a fixed number of slots per layer, stored the way
// llama.cpp stores MoE weights ([cols, rows, slot] per layer and part), so one mul_mat_id runs
// all of a token's GPU-resident experts. Per layer and token: the GPU part starts first and runs
// asynchronously while the CPU does the shared expert and the experts that are not in VRAM;
// then the GPU result is added. Experts the CPU keeps using get copied into VRAM (replacing
// colder ones), so VRAM fills with the hot set by itself.
#pragma once
#include <stdint.h>

typedef struct qgpu qgpu;

// types: [n_layer * 3] ggml types of gate, up, down per layer. gate/up are [D cols x F rows],
// down is [F cols x D rows] per expert. gb <= 0: use the free VRAM minus a margin.
// Returns NULL when there is no GPU (or the build has no GPU support); everything then stays on the CPU.
qgpu *gpu_open(int n_layer, int n_expert, int n_used, int D, int F, const int *types, double gb);
void gpu_close(qgpu *g);

// slot of expert e of layer l in VRAM, or -1
int gpu_slot(const qgpu *g, int l, int e);
// the CPU just computed expert e of layer l (its weights are in RAM at gate/up/down): count the
// use, and copy it into VRAM if it has become hot enough. Call outside a gpu_moe_start/finish pair.
// Returns 1 if it was copied in (the RAM copy is then no longer needed).
int gpu_offer(qgpu *g, int l, int e, const uint8_t *gate, const uint8_t *up, const uint8_t *down);
// VRAM used for experts; whether the GPU's memory is system RAM (integrated GPU)
uint64_t gpu_bytes(const qgpu *g);
int gpu_shares_ram(const qgpu *g);

// start the GPU part of layer l for T tokens (T <= 2): X is T rows of D floats; per token, n
// (slot, weight) pairs in slot[t * n + j], w[t * n + j] (pad with weight 0). Asynchronous.
void gpu_moe_start(qgpu *g, int l, int T, const float *const *X, int n, const int *slot, const float *w);
// wait for it and add its output into out[t] (D floats each)
void gpu_moe_finish(qgpu *g, float *const *out);

void gpu_stats(const qgpu *g, char *buf, int len);
