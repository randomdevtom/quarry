// Quarry: Qwen3.5 / Qwen3.6 MoE (qwen35moe) forward pass.
//
// 40 layers; 3 of every 4 are Gated DeltaNet (linear attention with a 128x128 state
// per value head and a short causal convolution), every 4th is gated full attention
// (16 heads of 256, 2 KV heads, RoPE on the first 64 dims). Every layer has a
// 256-expert MoE (8 used, softmax + renormalize) plus one shared expert gated by a
// sigmoid. Mirrors llama.cpp's src/models/qwen35moe.cpp and the CPU
// gated_delta_net op. The multi-token-prediction layer at the end is not used.
#pragma once
#include <stdint.h>
#include "model.h"
#include "qgguf.h"

typedef struct {
    int is_attn;
    float *attn_norm, *post_norm;
    // full attention
    qmat wq, wk, wv, wo;            // wq: [n_head * 2 * hd] rows = per head [q | gate]
    float *q_norm, *k_norm;
    // gated delta net
    qmat wqkv, wz, walpha, wbeta, wout;
    float *conv_w;                  // [conv_dim][n_conv]
    float *conv_wt;                 // [n_conv][conv_dim] (transposed, for the batched path)
    float *dt_bias, *ssm_a, *ssm_norm;
    // moe
    uint16_t *router;               // bf16 [n_expert][n_embd] (exact: F32 file values are bf16)
    float *router_f32;              // used instead when the values are not exact bf16
    float *shexp_gate;              // [n_embd]
    qmat sh_gate, sh_up, sh_down;
    qmat gate_exps, up_exps, down_exps;   // file offsets only; rows are per expert
} q35_layer;

typedef struct {
    int n_layer, n_embd, n_vocab, n_ctx;
    int n_head, n_head_kv, hd, n_rot;
    int n_expert, n_used, n_ff, n_ff_sh;
    int n_kh, n_vh, dk, dv, n_conv;    // delta net: key heads, value heads, head dims, conv taps
    int key_dim, value_dim, conv_dim;
    float eps, rope_base;
} q35_cfg;

typedef struct q35_model {
    q35_cfg c;
    qg_file g;
    qmat tok_embd, output;
    uint8_t *out4;                  // 4-bit copy of output for candidate search: per row, per 64 values: f32 scale + 32 bytes
    int out4_rb;                    // its row bytes
    float *output_norm;
    q35_layer *layers;
    struct expert_store *experts;
    // state: KV for attention layers (fp16), recurrent + conv state for delta-net layers
    int *attn_idx, *rec_idx, n_attn, n_rec;
    uint16_t *kc, *vc;              // [n_attn][n_ctx][n_head_kv * hd]
    float *S;                       // [n_rec][n_vh][dv][dk]  (row j = column j of the state)
    float *S2;                      // second state buffer (2-token verify writes the 2nd token's state here)
    float **sp, **sp_alt;           // [n_rec] current state / alternate buffer per layer
    float *conv_bk;                 // conv state after the 1st token of a 2-token step
    float *conv;                    // [n_rec][n_conv - 1][conv_dim]
    float *logits;
    uint64_t expert_bytes_read, expert_loads;
    int prefetch_on;
    uint64_t pred_total, pred_correct;
    // multi-token prediction (MTP) layer: drafts the next token for speculative decoding
    int has_mtp;                    // layers[n_layer] is the MTP block
    qmat eh_proj;
    float *mtp_enorm, *mtp_hnorm, *mtp_head_norm;
    uint16_t *mtp_kc, *mtp_vc;      // its own KV, one row per draft call
    int mtp_n;
    float *hlast;                   // final normed hidden of the last forward (feeds the MTP)
    float *h2[2], *logits2[2];      // 2-token verify outputs
    uint64_t spec_steps, spec_accepted;
    double draft_seconds;
    int *pred;                      // [n_layer][16] guesses for the next layer
    int pred_n;                     // guesses per layer
} q35_model;

int q35_detect(const char *path);
int q35_load(q35_model *m, const char *path, int n_ctx);
float *q35_forward(q35_model *m, int token, int pos);
// n tokens at positions pos0.. in batches; returns the logits of the last one
float *q35_prefill(q35_model *m, const int *tokens, int n, int pos0);
// recurrent state snapshot (delta-net S + conv), for prompt-prefix reuse
size_t q35_state_bytes(const q35_model *m);
void q35_state_save(const q35_model *m, void *dst);
void q35_state_load(q35_model *m, const void *src);
void q35_state_reset(q35_model *m);
// speculative decoding: draft the token after `tok` (at `pos`) with the MTP layer, from hlast
int q35_draft(q35_model *m, int tok, int pos);
// run tok0 at pos and tok1 at pos + 1 together; logits2[0], logits2[1] and h2[] are set.
// Then q35_commit(m, 1) keeps both tokens, q35_commit(m, 0) keeps only tok0.
void q35_forward2(q35_model *m, int tok0, int tok1, int pos);
void q35_commit(q35_model *m, int keep_both);
extern double q35_prof[8], q35_mp[8];
