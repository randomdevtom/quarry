// Quarry: Qwen3-MoE model (qwen3moe) forward pass.
#pragma once
#include <stdint.h>
#include "qgguf.h"

typedef struct {
    int n_layer, n_embd, n_head, n_head_kv, head_dim;
    int n_expert, n_expert_used, n_ff_exp, n_vocab, n_ctx;
    float rope_base, rms_eps;
} qcfg;

typedef struct {              // a quantized matrix: rows x cols
    int type, rows, cols;
    uint64_t row_bytes;
    uint8_t *data;            // in RAM (dense) or NULL (expert, streamed)
    uint64_t file_off;        // for streamed tensors
} qmat;

typedef struct {
    float *attn_norm, *ffn_norm, *q_norm, *k_norm;   // F32
    float *router;                                    // F32 [n_expert x n_embd]
    qmat wq, wk, wv, wo;
    qmat gate_exps, up_exps, down_exps;               // file offsets only
} qlayer;

struct expert_store;           // provides expert weights (stage 3: plain pread)

typedef struct {
    qcfg c;
    qg_file g;
    qmat tok_embd, output;
    float *output_norm;
    qlayer *layers;
    struct expert_store *experts;
    // runtime state
    float *k_cache, *v_cache;  // [layer][ctx][n_head_kv*head_dim]
    // scratch
    float *x, *xb, *xb2, *q, *k, *v, *att, *hb, *hb2, *logits, *router_logits;
    void *xq;                  // quantized activations scratch
    float *eh, *eg, *eo;       // batched expert scratch
    void *ehq;
    // stats
    uint64_t expert_bytes_read, expert_loads;
    int prefetch_on;           // predict + prefetch next layer's experts
    int *pred;                 // [layer][k] predicted experts
    uint64_t pred_total, pred_correct;
} qmodel;

int model_load(qmodel *m, const char *path, int n_ctx);
// run one token at position pos; returns pointer to logits (n_vocab)
float *model_forward(qmodel *m, int token, int pos);

// expert source API (implemented in experts.c)
struct expert_store *experts_open(qmodel *m);
// returns pointers to gate/up/down weights of expert e in layer l, valid until next call
void experts_get(struct expert_store *s, int l, int e, const uint8_t **gate, const uint8_t **up,
                 const uint8_t **down);
void experts_begin_layer(struct expert_store *s, int l, const int *sel, int k);
void experts_prefetch(struct expert_store *s, int l, const int *sel, int k);
int experts_ready(struct expert_store *s, int l, int e);
void experts_release(struct expert_store *s, int l, int e);
void experts_stats(struct expert_store *s, uint64_t *hits, uint64_t *misses, uint64_t *pf_issued,
                   uint64_t *pf_useful);
void experts_reset_stats(struct expert_store *s);
double experts_wait_seconds(struct expert_store *s);
