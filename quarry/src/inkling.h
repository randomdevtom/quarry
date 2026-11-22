// Quarry: Inkling 2 (inkling2_loop) forward pass.
//
// 24 physical layers; layers [loop_start, loop_end) run n_passes times, so a token
// goes through 48 "slots", each with its own KV cache and convolution state.
// Per layer: block attention residuals (attention over earlier block outputs),
// relative-position-biased attention with sinks (sliding window on most layers),
// short causal convolutions, a sigmoid-gated MoE with the SITU activation, and on
// looped passes a controller that mixes in messages from the previous pass.
#pragma once
#include <stdint.h>
#include "model.h"
#include "qgguf.h"

#define INK_MAX_SLOTS 64

typedef struct {
    int is_swa;
    float *attn_norm, *q_inp_norm, *k_inp_norm, *v_inp_norm;   // swa: attn_norm; full: q/k/v
    float *k_conv_proj, *v_conv_proj;                          // full only: [n_kv_head*n_conv x n_embd]
    qmat wq, wk, wv, wo, wgate, wrel;
    float *q_clip, *sink_proj, *rel_bias, *rel_clip;
    float *k_conv, *v_conv, *post_conv, *ffn_post_conv;        // [channels][n_conv]
    float *attn_res_query; int n_res_q;                         // NULL on layer 0
    float *ffn_res_query, *ffn_norm, *router, *probs_b;
    qmat gate_up, down;                                         // experts (file offsets only)
} ink_layer;

typedef struct {
    float *msg_query;                                           // [4][n_embd]
    float *ctrl_down[4], *ctrl_gain[4], *ctrl_forget[4];        // [64][2n_embd], [n_embd][64] x2
} ink_loop;

typedef struct {
    int n_phys, n_embd, n_head, n_kv, hd, n_expert, n_used, n_ff, n_vocab, n_vocab_pad, n_ctx;
    int loop_start, loop_end, n_passes, n_slots, gate_rank, block, n_conv;
    int n_rel, rel_extent, n_swa, log_floor;
    float eps, qk_clip, rel_clip_thr, log_alpha, w_scale, situ_beta, situ_lb, softcap;
} ink_cfg;

typedef struct ink_model {
    ink_cfg c;
    qg_file g;
    qmat tok_embd, output;
    float *tok_norm, *output_norm, *output_res_query;
    ink_layer *layers;
    ink_loop *loops;                                           // [loop_end - loop_start]
    struct expert_store *experts;
    // per slot state
    int slot_phys[INK_MAX_SLOTS], slot_cap[INK_MAX_SLOTS];
    float *kc[INK_MAX_SLOTS], *vc[INK_MAX_SLOTS], *conv[INK_MAX_SLOTS];
    float *logits;
    uint64_t expert_bytes_read, expert_loads;
    float *dbg_lout;                                           // optional: [slot][n_embd] of the last token
    int prefetch_on;
    uint64_t pred_total, pred_correct;
    int seq[INK_MAX_SLOTS], n_steps, step;                     // physical layer of each step in a token
    int pred[INK_MAX_SLOTS][16];                               // predicted experts per step
    int pred2[16];                                             // guess made at the start of the current layer
    uint64_t pred2_correct, skipped;
} ink_model;

int ink_load(ink_model *m, const char *path, int n_ctx);
float *ink_forward(ink_model *m, int token, int pos);
// is this GGUF an Inkling model?
int ink_detect(const char *path);
extern double ink_prof[8];
