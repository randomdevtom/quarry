// Quarry: Laguna (poolside, e.g. Laguna XS 2.1) forward pass.
//
// 40 layers, all attention: every 4th (0, 4, ...) is full attention with YaRN RoPE over the
// first 64 of 128 head dims, the others a 512-token sliding window with plain RoPE over all
// 128. Head counts differ per layer (48 full, 64 sliding), 8 KV heads, QK RMSNorm, and a
// softplus output gate per head computed from the layer input. Layer 0 has a dense SwiGLU
// FFN; the others a sigmoid-routed MoE (selection bias, normalized weights x 2.5) plus one
// always-on shared expert. Untied output layer. Mirrors llama.cpp's src/models/laguna.cpp.
#pragma once
#include <stdint.h>
#include "model.h"
#include "qgguf.h"

typedef struct {
    int swa, is_moe, n_head;
    float *attn_norm, *ffn_norm;
    qmat wq, wk, wv, wo, wg;        // wg: output gate, one row per head
    float *q_norm, *k_norm;
    qmat ffn_gate, ffn_up, ffn_down;   // dense layer / shared expert of a MoE layer
    float *router;                  // [n_expert][n_embd]
    float *probs_b;                 // [n_expert]
    qmat gate_exps, up_exps, down_exps;   // file offsets only; rows are per expert
} lag_layer;

typedef struct {
    int n_layer, n_embd, n_vocab, n_ctx;
    int n_head_max, n_head_kv, hd;
    int n_expert, n_used, n_ff, n_ff_exp, n_dense;
    int n_swa, ring;                // sliding window; ring slots kept per sliding layer
    int n_rot, n_rot_swa;
    float eps, w_scale;
    float rope_base, rope_base_swa;
    float yarn_scale, yarn_mscale, yarn_lo, yarn_hi;   // full layers: freq scale, magnitude, ramp dims
} lag_cfg;

typedef struct lag_model {
    lag_cfg c;
    qg_file g;
    qmat tok_embd, output;
    float *output_norm;
    lag_layer *layers;
    struct expert_store *experts;
    int *kv_idx, n_full, n_swa_l;   // index of a layer's cache among full / sliding layers
    uint16_t *kc, *vc;              // full layers: [n_full][n_ctx][kvd] fp16
    uint16_t *rk, *rv;              // sliding layers: [n_swa_l][ring][kvd] fp16, slot = pos % ring
    float *logits;
    uint8_t *out4;                  // 4-bit copy of the output layer for the candidate search
    int out4_rb;
    float *cos_full, *sin_full;     // unused placeholders (rope is computed per position)
    float *inv_full, *inv_swa;      // inverse frequencies
    uint64_t expert_bytes_read, expert_loads;
    int prefetch_on;
    int *pred;                      // [n_layer][16] guessed experts
    uint64_t pred_total, pred_correct;
} lag_model;

int lag_detect(const char *path);
int lag_load(lag_model *m, const char *path, int n_ctx);
float *lag_forward(lag_model *m, int token, int pos);
// n tokens at positions pos0.. in batches; returns the logits of the last one
float *lag_prefill(lag_model *m, const int *tokens, int n, int pos0);
// the sliding-window rings are a state like a recurrent one: a rewind needs a snapshot
// (the full-attention KV is positional and needs none)
size_t lag_state_bytes(const lag_model *m);
void lag_state_save(const lag_model *m, void *dst);
void lag_state_load(lag_model *m, const void *src);
void lag_state_reset(lag_model *m);
extern double lag_prof[8];
