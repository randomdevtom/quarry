// Quarry: LFM2 MoE (lfm2moe, e.g. LFM2.5-8B-A1B) forward pass.
//
// Hybrid stack: most layers are gated short convolutions (in_proj -> B, C, x; y = C *
// conv(B * x) over the last l_cache inputs; out_proj), the rest are GQA attention with
// QK-norm and NEOX RoPE. The first leading_dense_block_count layers have a dense SwiGLU
// FFN; the others a sigmoid-routed MoE (selection bias exp_probs_b, normalized weights).
// The output layer is tied to the token embedding. Mirrors llama.cpp's
// src/models/lfm2moe.cpp and lfm-base.cpp.
#pragma once
#include <stdint.h>
#include "model.h"
#include "qgguf.h"

typedef struct {
    int is_attn, is_moe;
    float *attn_norm, *ffn_norm;
    // attention
    qmat wq, wk, wv, wo;
    float *q_norm, *k_norm;
    // short conv
    qmat in_proj, out_proj;
    float *conv_w;                  // [n_embd][l_cache]
    // dense ffn
    qmat ffn_gate, ffn_up, ffn_down;
    // moe
    float *router;                  // [n_expert][n_embd]
    float *probs_b;                 // [n_expert]
    qmat gate_exps, up_exps, down_exps;   // file offsets only; rows are per expert
} lfm_layer;

typedef struct {
    int n_layer, n_embd, n_vocab, n_ctx;
    int n_head, n_head_kv, hd;
    int n_expert, n_used, n_ff, n_ff_exp, n_dense, l_cache;
    float eps, rope_base, w_scale;
} lfm_cfg;

typedef struct lfm_model {
    lfm_cfg c;
    qg_file g;
    qmat tok_embd;                  // in RAM: embedding and (tied) output layer
    float *output_norm;
    lfm_layer *layers;
    struct expert_store *experts;
    int *attn_idx, *conv_idx, n_attn, n_conv;
    uint16_t *kc, *vc;              // [n_attn][n_ctx][n_head_kv * hd] fp16
    float *conv;                    // [n_conv][l_cache - 1][n_embd], oldest first
    float *logits;
    uint8_t *out4;                  // 4-bit copy of the output layer for the candidate search
    int out4_rb;
    float *rope_inv;                // [hd / 2] inverse frequencies
    uint64_t expert_bytes_read, expert_loads;
    int prefetch_on;
    int *pred;                      // [n_layer][16] guessed experts
    uint64_t pred_total, pred_correct;
} lfm_model;

int lfm_detect(const char *path);
int lfm_load(lfm_model *m, const char *path, int n_ctx);
float *lfm_forward(lfm_model *m, int token, int pos);
// n tokens at positions pos0.. in batches; returns the logits of the last one
float *lfm_prefill(lfm_model *m, const int *tokens, int n, int pos0);
// conv state snapshot (the attention KV is positional and needs no snapshot)
size_t lfm_state_bytes(const lfm_model *m);
void lfm_state_save(const lfm_model *m, void *dst);
void lfm_state_load(lfm_model *m, const void *src);
void lfm_state_reset(lfm_model *m);
extern double lfm_prof[8];
