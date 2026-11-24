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
    uint16_t *router;                                 // bf16 [n_expert x n_embd] (the F32 file values are exact bf16)
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
    uint16_t *k_cache, *v_cache;  // fp16 [layer][ctx][n_head_kv*head_dim]
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
    void *pf;                  // batched prefill buffers (allocated on first use)
} qmodel;

int model_load(qmodel *m, const char *path, int n_ctx);
// run one token at position pos; returns pointer to logits (n_vocab)
float *model_forward(qmodel *m, int token, int pos);
// run n tokens at positions pos0.. as batches; returns logits of the last one
float *model_prefill(qmodel *m, const int *tokens, int n, int pos0);
extern double qt_pf[8];        // prefill profile: 0 attn mm, 1 attn core, 2 route, 3 experts, 4 io wait, 5 misc

// expert source API (implemented in experts.c)
typedef struct {
    int fd;
    uint64_t file_size;
    int n_layer, n_expert, n_parts;   // up to 3 parts per expert (e.g. gate, up, down)
    const uint64_t *off, *bytes;       // [layer * n_parts + part]: offset of expert 0, bytes per expert
    uint64_t *bytes_read, *loads;      // counters updated by the I/O threads
    int min_slots;
    const char *packed;                // optional file with each expert's parts back to back
    // optional: called by the I/O thread on each part after it is read (e.g. an in-place repack)
    void (*fixup)(void *ud, int layer, int part, uint8_t *data, uint64_t bytes);
    void *fixup_ud;
} expert_src;
// write src's experts to `path`, each expert's parts back to back in its cache-slot layout,
// so a miss is one contiguous read. Returns 0 on success.
int experts_write_packed(const expert_src *src, const char *path);
struct expert_store *experts_open_src(const expert_src *src);
struct expert_store *experts_open(qmodel *m);
// returns pointers to gate/up/down weights of expert e in layer l, valid until next call
void experts_get(struct expert_store *s, int l, int e, const uint8_t **gate, const uint8_t **up,
                 const uint8_t **down);
void experts_begin_layer(struct expert_store *s, int l, const int *sel, int k);
void experts_prefetch(struct expert_store *s, int l, const int *sel, int k);
int experts_ready(struct expert_store *s, int l, int e);
void experts_wait_any(struct expert_store *s, int l, const int *e, int n);
void experts_release(struct expert_store *s, int l, int e);
void experts_stats(struct expert_store *s, uint64_t *hits, uint64_t *misses, uint64_t *pf_issued,
                   uint64_t *pf_useful);
void experts_reset_stats(struct expert_store *s);
double experts_wait_seconds(struct expert_store *s);
int experts_try_pin(struct expert_store *s, int l, int e, const uint8_t **gate, const uint8_t **up,
                    const uint8_t **down);
int experts_stream(struct expert_store *s, int l, int e);
void experts_stream_get(struct expert_store *s, int h, const uint8_t **gate, const uint8_t **up,
                        const uint8_t **down);
void experts_stream_done(struct expert_store *s, int h);
void experts_demote(struct expert_store *s, int l, const int *sel, int k);
