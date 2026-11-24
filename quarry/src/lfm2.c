// LFM2 MoE forward pass (see lfm2.h). One code path for decode (T = 1) and batched
// prefill (T tokens): every weight row is read once per batch and dotted with all tokens.
#define _GNU_SOURCE
#include "lfm2.h"

#include <immintrin.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/mman.h>

#include "quants.h"
#include "qwen35.h"   // q35_mm_any: batched matmul for prefill

double lfm_prof[8];   // 0 conv, 1 attention, 2 dense ffn, 3 route, 4 experts, 5 output, 6 attn core
static double lnow(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

static void *xalloc(size_t n) {
    const size_t al = n >= (1u << 21) ? (1u << 21) : 64, sz = (n + al - 1) / al * al;
    void *p = aligned_alloc(al, sz);
    if (!p) { fprintf(stderr, "out of memory (%zu bytes)\n", n); exit(1); }
    if (al > 64) madvise(p, sz, MADV_HUGEPAGE);
    memset(p, 0, sz);
    return p;
}

static void read_at(qg_file *g, void *dst, uint64_t n, uint64_t off) {
    uint8_t *p = dst;
    while (n) {
        ssize_t r = pread(g->fd, p, n > (1u << 30) ? (1u << 30) : n, off);
        if (r <= 0) { perror("pread"); exit(1); }
        p += r; off += r; n -= r;
    }
}

static const qg_tensor *find(qg_file *g, const char *name, int required) {
    const qg_tensor *t = qg_find_tensor(g, name);
    if (!t && required) { fprintf(stderr, "lfm2: missing tensor %s\n", name); exit(1); }
    return t;
}

static float *f32(qg_file *g, const char *name) {
    const qg_tensor *t = find(g, name, 1);
    if (t->type != GT_F32) { fprintf(stderr, "lfm2: %s is not F32\n", name); exit(1); }
    float *p = xalloc(t->nbytes);
    read_at(g, p, t->nbytes, t->offset);
    return p;
}

static int use_kr(void) { static int v = -1; if (v < 0) v = !getenv("QUARRY_NO_KR"); return v; }

static qmat mat(qg_file *g, const char *name, int in_ram) {
    const qg_tensor *t = find(g, name, 1);
    qmat q = {t->type, (int)(t->ne[1] * t->ne[2]), (int)t->ne[0], qg_row_bytes(t->type, t->ne[0]), NULL, t->offset};
    if (in_ram) {
        q.data = xalloc(t->nbytes);
        read_at(g, q.data, t->nbytes, t->offset);
        if (q.type == GT_Q4_K && use_kr()) { q4k_repack_kr(q.data, t->nbytes / sizeof(block_q4_K)); q.type = GT_Q4_KR; }
    }
    return q;
}

// expert parts arrive from the file as Q4_K: repack them in place (types per layer and part)
static void expert_fixup(void *ud, int l, int part, uint8_t *data, uint64_t bytes) {
    const lfm_model *m = ud;
    const lfm_layer *L = &m->layers[l];
    const qmat *M = part == 0 ? &L->gate_exps : part == 1 ? &L->up_exps : &L->down_exps;
    if (M->type == GT_Q4_KR) q4k_repack_kr(data, bytes / sizeof(block_q4_K));
}

static int arr_int(qg_file *g, const char *key, int i, int def) {
    const qg_kv *kv = qg_find_kv(g, key);
    if (!kv) return def;
    if (kv->type != GV_ARR) return (int)kv->i;
    return (uint64_t)i < kv->arr_n ? (int)qg_arr_int(kv, i) : def;
}

int lfm_detect(const char *path) {
    qg_file g;
    if (qg_open(&g, path)) return 0;
    const char *a = qg_get_str(&g, "general.architecture");
    int r = a && !strcmp(a, "lfm2moe");
    qg_close(&g);
    return r;
}

int lfm_load(lfm_model *m, const char *path, int n_ctx) {
    memset(m, 0, sizeof *m);
    if (qg_open(&m->g, path)) return -1;
    qg_file *g = &m->g;
    const char *arch = qg_get_str(g, "general.architecture");
    if (!arch || strcmp(arch, "lfm2moe")) { fprintf(stderr, "lfm2: unsupported arch %s\n", arch); return -1; }
    lfm_cfg *c = &m->c;
#define K(s) "lfm2moe." s
    c->n_layer = qg_get_int(g, K("block_count"), 0);
    c->n_embd = qg_get_int(g, K("embedding_length"), 0);
    c->n_head = arr_int(g, K("attention.head_count"), 0, 0);
    c->hd = qg_get_int(g, K("attention.key_length"), c->n_embd / (c->n_head ? c->n_head : 1));
    c->n_expert = qg_get_int(g, K("expert_count"), 0);
    c->n_used = qg_get_int(g, K("expert_used_count"), 0);
    if (getenv("QUARRY_TOPK") && atoi(getenv("QUARRY_TOPK")) > 0 && atoi(getenv("QUARRY_TOPK")) < c->n_used) c->n_used = atoi(getenv("QUARRY_TOPK"));   // lossy
    c->n_ff = arr_int(g, K("feed_forward_length"), 0, 0);
    c->n_ff_exp = arr_int(g, K("expert_feed_forward_length"), 0, 0);
    c->n_dense = qg_get_int(g, K("leading_dense_block_count"), 0);
    c->l_cache = qg_get_int(g, K("shortconv.l_cache"), 3);
    c->eps = qg_get_float(g, K("attention.layer_norm_rms_epsilon"), 1e-5);
    c->rope_base = qg_get_float(g, K("rope.freq_base"), 1e6);
    c->w_scale = qg_get_float(g, K("expert_weights_scale"), 1);
    if (c->w_scale == 0) c->w_scale = 1;
    const int gating = qg_get_int(g, K("expert_gating_func"), 2);
    c->n_ctx = n_ctx;
    if (gating != 2 || c->hd % 32 || c->hd > 256 || c->n_used > 16 || c->n_expert > 256 || c->l_cache != 3 || c->n_embd % QK_K) {
        fprintf(stderr, "lfm2: unsupported shape\n");
        return -1;
    }

    m->tok_embd = mat(g, "token_embd.weight", 1);
    c->n_vocab = m->tok_embd.rows;
    if (!getenv("QUARRY_EXACT_OUTPUT") && c->n_embd % 256 == 0) {
        // 4-bit copy of the (tied) output layer: per 64 values an f32 scale and 32 bytes. It finds
        // the top candidates each step; their exact logits come from the real rows.
        const qmat *O = &m->tok_embd;
        const int nb = O->cols / 64;
        m->out4_rb = nb * 36;
        m->out4 = xalloc((size_t)O->rows * m->out4_rb);
        #pragma omp parallel
        {
            float *f = malloc((size_t)O->cols * 4);
            #pragma omp for schedule(static)
            for (int r = 0; r < O->rows; r++) {
                dequant_row(O->type, O->data + (size_t)r * O->row_bytes, f, O->cols);
                uint8_t *dst = m->out4 + (size_t)r * m->out4_rb;
                for (int b = 0; b < nb; b++) {
                    const float *v = f + b * 64;
                    float amax = 0;
                    for (int i = 0; i < 64; i++) amax = fabsf(v[i]) > amax ? fabsf(v[i]) : amax;
                    const float d = amax / 7.5f, id = d > 0 ? 1.0f / d : 0;
                    memcpy(dst + b * 36, &d, 4);
                    uint8_t *q = dst + b * 36 + 4;
                    for (int i = 0; i < 32; i++) {
                        int lo = (int)lrintf(v[i] * id - 0.5f) + 8, hi = (int)lrintf(v[i + 32] * id - 0.5f) + 8;
                        lo = lo < 0 ? 0 : lo > 15 ? 15 : lo; hi = hi < 0 ? 0 : hi > 15 ? 15 : hi;
                        q[i] = (uint8_t)(lo | hi << 4);
                    }
                }
            }
            free(f);
        }
    }
    if (find(g, "output.weight", 0)) { fprintf(stderr, "lfm2: untied output layer not supported\n"); return -1; }
    m->output_norm = f32(g, "token_embd_norm.weight");
    m->layers = calloc(c->n_layer, sizeof(lfm_layer));
    m->attn_idx = malloc(c->n_layer * sizeof(int));
    m->conv_idx = malloc(c->n_layer * sizeof(int));
    char n[128];
#define NM(s) (snprintf(n, sizeof n, "blk.%d.%s", l, s), n)
    for (int l = 0; l < c->n_layer; l++) {
        lfm_layer *L = &m->layers[l];
        const int nkv = arr_int(g, K("attention.head_count_kv"), l, 0);
        L->is_attn = nkv > 0;
        L->is_moe = l >= c->n_dense;
        L->attn_norm = f32(g, NM("attn_norm.weight"));
        L->ffn_norm = f32(g, NM("ffn_norm.weight"));
        if (L->is_attn) {
            if (c->n_head_kv && c->n_head_kv != nkv) { fprintf(stderr, "lfm2: varying KV heads not supported\n"); return -1; }
            c->n_head_kv = nkv;
            L->wq = mat(g, NM("attn_q.weight"), 1);
            L->wk = mat(g, NM("attn_k.weight"), 1);
            L->wv = mat(g, NM("attn_v.weight"), 1);
            L->wo = mat(g, NM("attn_output.weight"), 1);
            L->q_norm = f32(g, NM("attn_q_norm.weight"));
            L->k_norm = f32(g, NM("attn_k_norm.weight"));
            m->attn_idx[l] = m->n_attn++;
            m->conv_idx[l] = -1;
        } else {
            L->in_proj = mat(g, NM("shortconv.in_proj.weight"), 1);
            L->out_proj = mat(g, NM("shortconv.out_proj.weight"), 1);
            L->conv_w = f32(g, NM("shortconv.conv.weight"));
            m->conv_idx[l] = m->n_conv++;
            m->attn_idx[l] = -1;
        }
        if (L->is_moe) {
            L->router = f32(g, NM("ffn_gate_inp.weight"));
            L->probs_b = f32(g, NM("exp_probs_b.bias"));
            L->gate_exps = mat(g, NM("ffn_gate_exps.weight"), 0);
            L->up_exps = mat(g, NM("ffn_up_exps.weight"), 0);
            L->down_exps = mat(g, NM("ffn_down_exps.weight"), 0);
            L->gate_exps.rows /= c->n_expert;
            L->up_exps.rows /= c->n_expert;
            L->down_exps.rows /= c->n_expert;
        } else {
            L->ffn_gate = mat(g, NM("ffn_gate.weight"), 1);
            L->ffn_up = mat(g, NM("ffn_up.weight"), 1);
            L->ffn_down = mat(g, NM("ffn_down.weight"), 1);
        }
    }
#undef NM
#undef K
    if (c->n_head_kv == 0 || c->n_head % c->n_head_kv) { fprintf(stderr, "lfm2: bad head counts\n"); return -1; }
    const size_t kvd = (size_t)c->n_head_kv * c->hd;
    m->kc = xalloc((size_t)m->n_attn * n_ctx * kvd * 2);
    m->vc = xalloc((size_t)m->n_attn * n_ctx * kvd * 2);
    m->conv = xalloc((size_t)m->n_conv * (c->l_cache - 1) * c->n_embd * 4);
    m->logits = xalloc((size_t)c->n_vocab * 4);
    m->rope_inv = xalloc(c->hd / 2 * 4);
    for (int i = 0; i < c->hd / 2; i++) m->rope_inv[i] = powf(c->rope_base, -2.0f * i / c->hd);
    m->pred = calloc(c->n_layer * 16, sizeof(int));

    // experts: gate, up, down of the MoE layers (dense layers get zero-size parts)
    const int NL = c->n_layer;
    uint64_t *off = calloc(NL * 3, sizeof(uint64_t)), *bytes = calloc(NL * 3, sizeof(uint64_t));
    for (int l = 0; l < NL; l++) {
        const lfm_layer *L = &m->layers[l];
        if (!L->is_moe) continue;
        const qmat *M[3] = {&L->gate_exps, &L->up_exps, &L->down_exps};
        for (int k = 0; k < 3; k++) { off[l * 3 + k] = M[k]->file_off; bytes[l * 3 + k] = M[k]->row_bytes * M[k]->rows; }
    }
    for (int l = 0; l < NL; l++) {
        lfm_layer *L = &m->layers[l];
        qmat *M[3] = {&L->gate_exps, &L->up_exps, &L->down_exps};
        for (int k = 0; k < 3 && L->is_moe; k++) if (M[k]->type == GT_Q4_K && use_kr()) M[k]->type = GT_Q4_KR;
    }
    expert_src src = {g->fd, g->file_size, NL, c->n_expert, 3, off, bytes,
                      &m->expert_bytes_read, &m->expert_loads, 2 * c->n_used, NULL, expert_fixup, m};
    m->experts = experts_open_src(&src);
    free(off); free(bytes);
    m->prefetch_on = getenv("QUARRY_NO_PREFETCH") ? 0 : 1;
    // matvec rows are handed out in chunks of 8 (OMP_SCHEDULE overrides): when other programs
    // take CPU time, static shares make every region wait for a preempted thread
    if (!getenv("OMP_SCHEDULE")) omp_set_schedule(omp_sched_dynamic, 8);
    {   // when the cache holds every expert, read them all in the background now
        double all = 0;
        for (int l = 0; l < NL; l++) {
            const lfm_layer *L = &m->layers[l];
            if (L->is_moe) all += (double)(L->gate_exps.row_bytes * L->gate_exps.rows + L->up_exps.row_bytes * L->up_exps.rows +
                                           L->down_exps.row_bytes * L->down_exps.rows) * c->n_expert;
        }
        const double gb = getenv("QUARRY_CACHE_GB") ? atof(getenv("QUARRY_CACHE_GB")) : 5.5;
        if (gb * 1e9 >= all * 1.01 && !getenv("QUARRY_NO_PRELOAD")) {
            int ids[256];
            for (int e = 0; e < c->n_expert; e++) ids[e] = e;
            for (int l = 0; l < NL; l++) if (m->layers[l].is_moe) experts_prefetch(m->experts, l, ids, c->n_expert);
            if (getenv("QUARRY_PRELOAD_WAIT"))   // benchmarks: start with every expert in RAM
                for (int l = 0; l < NL; l++)
                    for (int e = 0; m->layers[l].is_moe && e < c->n_expert; e++) { const uint8_t *a, *b, *d; experts_get(m->experts, l, e, &a, &b, &d); }
        }
    }
    return 0;
}

size_t lfm_state_bytes(const lfm_model *m) { return (size_t)m->n_conv * (m->c.l_cache - 1) * m->c.n_embd * 4; }
void lfm_state_save(const lfm_model *m, void *dst) { memcpy(dst, m->conv, lfm_state_bytes(m)); }
void lfm_state_load(lfm_model *m, const void *src) { memcpy(m->conv, src, lfm_state_bytes(m)); }
void lfm_state_reset(lfm_model *m) { memset(m->conv, 0, lfm_state_bytes(m)); }

// ---------------- small ops ----------------

static inline float dotf(const float *a, const float *b, int n) {
    __m512 s0 = _mm512_setzero_ps(), s1 = s0;
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        s0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), s0);
        s1 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 16), _mm512_loadu_ps(b + i + 16), s1);
    }
    float s = _mm512_reduce_add_ps(_mm512_add_ps(s0, s1));
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}

static void rmsnorm(float *o, const float *x, const float *w, int n, float eps) {
    double ss = 0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    const float s = 1.0f / sqrtf((float)(ss / n) + eps);
    for (int i = 0; i < n; i++) o[i] = x[i] * s * w[i];
}

static inline __m512 exp_ps(__m512 x) {
    x = _mm512_min_ps(_mm512_max_ps(x, _mm512_set1_ps(-87.0f)), _mm512_set1_ps(87.0f));
    const __m512 n = _mm512_roundscale_ps(_mm512_mul_ps(x, _mm512_set1_ps(1.44269504f)), _MM_FROUND_TO_NEAREST_INT);
    const __m512 r = _mm512_fnmadd_ps(n, _mm512_set1_ps(0.693147181f), x);
    __m512 p = _mm512_set1_ps(1.0f / 720);
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f / 120));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f / 24));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f / 6));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(0.5f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f));
    return _mm512_scalef_ps(p, n);
}
// g = silu(g) * h, n a multiple of 16
static void silu_mul(float *g, const float *h, int n) {
    const __m512 one = _mm512_set1_ps(1.0f);
    for (int i = 0; i < n; i += 16) {
        const __m512 x = _mm512_loadu_ps(g + i);
        const __m512 s = _mm512_div_ps(x, _mm512_add_ps(one, exp_ps(_mm512_sub_ps(_mm512_setzero_ps(), x))));
        _mm512_storeu_ps(g + i, _mm512_mul_ps(s, _mm512_loadu_ps(h + i)));
    }
}

static inline void f32_to_f16(uint16_t *dst, const float *src, int n) {
    for (int i = 0; i < n; i += 16)
        _mm256_storeu_si256((__m256i *)(dst + i), _mm512_cvtps_ph(_mm512_loadu_ps(src + i), _MM_FROUND_TO_NEAREST_INT));
}
static inline __m512 f16x16(const uint16_t *p) { return _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)p)); }

// ---------------- quantized activations ----------------

// per token: input quantized for K-quant (q8_K) and 32-block weights, made on demand
typedef struct {
    const float *x;
    int n, have_k, have_32, have_kr;
    block_q8_K *qk, *qkr;
    act32 a;
} qin;

static void qin_need(qin *q, int type) {
    if (type == GT_Q4_KR) {
        if (!q->have_k) { quantize_row_q8_K(q->x, q->qk, q->n); q->have_k = 1; }
        if (!q->have_kr) { q8k_permute_kr(q->qk, q->qkr, q->n); q->have_kr = 1; }
    } else if (type == GT_Q8_0 || type == GT_Q5_0 || type == GT_Q4_0 || type == GT_Q4_1) {
        if (!q->have_32) { quantize_row_a32(q->x, q->a, q->n); q->have_32 = 1; }
    } else if (!q->have_k) { quantize_row_q8_K(q->x, q->qk, q->n); q->have_k = 1; }
}

static inline float row_dot(int type, const uint8_t *row, int n, const qin *q) {
    switch (type) {
    case GT_Q8_0: return vec_dot_q8_0_a32(n, (const block_q8_0 *)row, q->a);
    case GT_Q5_0: return vec_dot_q5_0_a32(n, (const block_q5_0 *)row, q->a);
    case GT_Q4_0: return vec_dot_q4_0_a32(n, (const block_q4_0 *)row, q->a);
    case GT_Q4_1: return vec_dot_q4_1_a32(n, (const block_q4_1 *)row, q->a);
    case GT_Q3_K: return vec_dot_q3_K_q8_K(n, (const block_q3_K *)row, q->qk);
    case GT_Q4_K: return vec_dot_q4_K_q8_K(n, (const block_q4_K *)row, q->qk);
    case GT_Q4_KR: return vec_dot_q4_KR_q8_KR(n, (const block_q4_K *)row, q->qkr);
    case GT_Q5_K: return vec_dot_q5_K_q8_K(n, (const block_q5_K *)row, q->qk);
    case GT_Q6_K: return vec_dot_q6_K_q8_K(n, (const block_q6_K *)row, q->qk);
    }
    fprintf(stderr, "lfm2: no dot for type %d\n", type);
    exit(1);
}

// a set of T quantized-activation slots, each for vectors of up to `cap` values
typedef struct {
    int T, cap;
    qin *q;
    block_q8_K *qk, *qkr;
    int8_t *qs; float *d; int32_t *s;
} qset;

static void qset_alloc(qset *s, int T, int cap) {
    s->T = T; s->cap = cap;
    s->q = calloc(T, sizeof(qin));
    s->qk = xalloc(sizeof(block_q8_K) * (size_t)T * (cap / QK_K + 1));
    s->qkr = xalloc(sizeof(block_q8_K) * (size_t)T * (cap / QK_K + 1));
    s->qs = xalloc((size_t)T * cap);
    s->d = xalloc((size_t)T * (cap / 32 + 1) * 4);
    s->s = xalloc((size_t)T * (cap / 32 + 1) * 4);
}
// slot t now holds vector x of n values (quantized when first needed)
static qin *qset_set(qset *s, int t, const float *x, int n) {
    qin *q = &s->q[t];
    q->x = x; q->n = n; q->have_k = q->have_32 = q->have_kr = 0;
    q->qk = s->qk + (size_t)t * (s->cap / QK_K + 1);
    q->qkr = s->qkr + (size_t)t * (s->cap / QK_K + 1);
    q->a = (act32){s->qs + (size_t)t * s->cap, s->d + (size_t)t * (s->cap / 32 + 1), s->s + (size_t)t * (s->cap / 32 + 1)};
    return q;
}

// Y[t * ldy + r] = W[r] . q[t] for rows [0, rows) and the job's tokens t in [0, T).
// Each weight row is read once and used for every token; all jobs share one parallel region.
typedef struct { const qmat *W; const uint8_t *base; float *Y; int ldy; int rows; qin *const *q; int T; } mm_job;

enum { MAXJ = 40 };
static void mm_multi(const mm_job *jobs, int nj) {
    const double t_mm = lnow();
    int start[MAXJ + 1] = {0};
    for (int j = 0; j < nj; j++) {
        for (int t = 0; t < jobs[j].T; t++) qin_need(jobs[j].q[t], jobs[j].W->type);
        start[j + 1] = start[j] + jobs[j].rows;
    }
    const int total = start[nj];
    #pragma omp parallel for schedule(runtime)
    for (int r = 0; r < total; r++) {
        int j = 0;
        while (r >= start[j + 1]) j++;
        const mm_job *J = &jobs[j];
        const int rr = r - start[j];
        const uint8_t *row = J->base + (uint64_t)rr * J->W->row_bytes;
        for (int t = 0; t < J->T; t++) J->Y[(size_t)t * J->ldy + rr] = row_dot(J->W->type, row, J->W->cols, J->q[t]);
    }
    lfm_prof[7] += lnow() - t_mm;
}
static void mm1(const qmat *W, float *Y, int ldy, qin *const *q, int T) {
    mm_job j = {W, W->data, Y, ldy, W->rows, q, T};
    mm_multi(&j, 1);
}

// ---------------- scratch ----------------

enum { TMAX = 256 };

// prefill: T >= this many tokens per weight use the batched int8 matmul (rows requantized
// once per batch, VNNI over many tokens); fewer use the per-token decode kernels
static int mm_batched(int T) { static int th = -1; if (th < 0) th = getenv("QUARRY_MM_T") ? atoi(getenv("QUARRY_MM_T")) : 8; return T >= th; }
static void mmf(const qmat *W, const uint8_t *base, const float *X, int ldx, int T, float *Y, int ldy) {
    static int mode = -1;   // 0: exact integer kernels, 1: float rows, 2: int8-requantized rows
    if (mode < 0) mode = getenv("QUARRY_PF_S8") ? 2 : getenv("QUARRY_PF_F32") ? 1 : 0;
    const int cols = W->cols;
    if (mode == 0 && kq_supported(W->type) && cols % QK_K == 0 && T <= TMAX) {
        static block_q8_K *qn, *qp;
        static float *xs;
        static size_t cap;
        const int nb = cols / QK_K;
        if ((size_t)T * nb > cap) {
            free(qn); free(qp); free(xs);
            cap = (size_t)TMAX * (8192 / QK_K);
            if (cap < (size_t)T * nb) cap = (size_t)T * nb;
            qn = xalloc(cap * sizeof(block_q8_K)); qp = xalloc(cap * sizeof(block_q8_K)); xs = xalloc(cap * 16 * 4);
        }
        const int kr = W->type == GT_Q4_KR;
        #pragma omp parallel for schedule(static)
        for (int t = 0; t < T; t++) {
            quantize_row_q8_K(X + (size_t)t * ldx, qn + (size_t)t * nb, cols);
            if (kr) q8k_permute_kr(qn + (size_t)t * nb, qp + (size_t)t * nb, cols);
            kq_xs(qn + (size_t)t * nb, cols, xs + (size_t)t * nb * 16);
        }
        const block_q8_K *xp[TMAX];
        const float *xsp[TMAX];
        for (int t = 0; t < T; t++) { xp[t] = (kr ? qp : qn) + (size_t)t * nb; xsp[t] = xs + (size_t)t * nb * 16; }
        kq_matmul(W->type, base, W->row_bytes, W->rows, cols, T, xp, xsp, Y, ldy);
        return;
    }
    if (mode == 2) q35_mm_any(W->type, base, W->row_bytes, W->rows, cols, X, ldx, T, Y, ldy);
    else q35_mm_f32(W->type, base, W->row_bytes, W->rows, cols, X, ldx, T, Y, ldy);
}

typedef struct {
    int T;
    float *x, *xn, *big, *big2, *att, *o;   // [T][...]
    float *eo;                              // [T][n_used][D] expert outputs, summed in routing order
    float *eg, *eh, *eob;                   // [T * n_used][F] expert gate / up, [T * n_used][D] down
    float *gx;                              // [T][D] inputs gathered for one expert
    qset qa, qb;                            // input / hidden activations
    int *sel; float *w;                     // [T][16]
    float *sc;                              // attention scores [threads][n_ctx]
} scratch;

static scratch *scr(lfm_model *m) {
    static scratch S;
    if (S.x) return &S;
    const lfm_cfg *c = &m->c;
    const int D = c->n_embd, T = TMAX;
    const int qd = c->n_head * c->hd;
    int big = 3 * D;
    if (c->n_ff > big) big = c->n_ff;
    if (qd + 2 * c->n_head_kv * c->hd > big) big = qd + 2 * c->n_head_kv * c->hd;
    S.x = xalloc((size_t)T * D * 4);
    S.xn = xalloc((size_t)T * D * 4);
    S.big = xalloc((size_t)T * big * 4);
    S.big2 = xalloc((size_t)T * big * 4);
    S.att = xalloc((size_t)T * (qd > D ? qd : D) * 4);
    S.o = xalloc((size_t)T * D * 4);
    S.eo = xalloc((size_t)T * c->n_used * D * 4);
    S.eg = xalloc((size_t)T * c->n_used * c->n_ff_exp * 4);
    S.eh = xalloc((size_t)T * c->n_used * c->n_ff_exp * 4);
    S.eob = xalloc((size_t)T * c->n_used * D * 4);
    S.gx = xalloc((size_t)T * D * 4);
    int cap = big > c->n_ff_exp ? big : c->n_ff_exp;
    if (qd > cap) cap = qd;
    qset_alloc(&S.qa, T, cap);
    qset_alloc(&S.qb, T * c->n_used, cap);
    S.sel = xalloc((size_t)T * 16 * 4);
    S.w = xalloc((size_t)T * 16 * 4);
    S.sc = xalloc((size_t)omp_get_max_threads() * (c->n_ctx + 16) * 4);
    return &S;
}

// ---------------- blocks ----------------

// gated short conv for T tokens: XN in, OUT [T][D]
static void shortconv(lfm_model *m, const lfm_layer *L, int ci, scratch *s, int T, float *out) {
    const lfm_cfg *c = &m->c;
    const int D = c->n_embd;
    qin *q[TMAX];
    float *bcx = s->big;                       // [T][3D]: B | C | x
    if (mm_batched(T)) mmf(&L->in_proj, L->in_proj.data, s->xn, D, T, bcx, 3 * D);
    else {
        for (int t = 0; t < T; t++) q[t] = qset_set(&s->qa, t, s->xn + (size_t)t * D, D);
        mm1(&L->in_proj, bcx, 3 * D, q, T);
    }
    float *st = m->conv + (size_t)ci * 2 * D;  // [2][D]: inputs t-2, t-1
    float *y = s->att;                         // [T][D]
    for (int t = 0; t < T; t++) {
        const float *B = bcx + (size_t)t * 3 * D, *C = B + D, *X = B + 2 * D;
        float *yt = y + (size_t)t * D;
        for (int ch = 0; ch < D; ch++) {
            const float bx = B[ch] * X[ch], *w = L->conv_w + (size_t)ch * 3;
            const float v = w[0] * st[ch] + w[1] * st[D + ch] + w[2] * bx;
            st[ch] = st[D + ch];
            st[D + ch] = bx;
            yt[ch] = C[ch] * v;
        }
    }
    if (mm_batched(T)) { mmf(&L->out_proj, L->out_proj.data, y, D, T, out, D); return; }
    for (int t = 0; t < T; t++) q[t] = qset_set(&s->qb, t, y + (size_t)t * D, D);
    mm1(&L->out_proj, out, D, q, T);
}

static void rope(float *v, int heads, int hd, int pos, const float *inv) {
    const int h2 = hd / 2;
    float cs[128], sn[128];
    for (int i = 0; i < h2; i++) { const float th = pos * inv[i]; cs[i] = cosf(th); sn[i] = sinf(th); }
    for (int h = 0; h < heads; h++) {
        float *x = v + h * hd;
        for (int i = 0; i < h2; i++) {
            const float a = x[i], b = x[i + h2];
            x[i] = a * cs[i] - b * sn[i];
            x[i + h2] = a * sn[i] + b * cs[i];
        }
    }
}

// GQA attention for T tokens at positions pos0..: XN in, OUT [T][D]
static void attention(lfm_model *m, const lfm_layer *L, int ai, scratch *s, int T, int pos0, float *out) {
    const lfm_cfg *c = &m->c;
    const int D = c->n_embd, HD = c->hd, NH = c->n_head, NKV = c->n_head_kv;
    const int qd = NH * HD, kvd = NKV * HD, W = qd + 2 * kvd, gqa = NH / NKV;
    qin *q[TMAX];
    float *P = s->big;                         // [T][W]: q | k | v
    if (mm_batched(T)) {
        mmf(&L->wq, L->wq.data, s->xn, D, T, P, W);
        mmf(&L->wk, L->wk.data, s->xn, D, T, P + qd, W);
        mmf(&L->wv, L->wv.data, s->xn, D, T, P + qd + kvd, W);
    } else {
    for (int t = 0; t < T; t++) q[t] = qset_set(&s->qa, t, s->xn + (size_t)t * D, D);
    mm_job jobs[3] = {{&L->wq, L->wq.data, P, W, L->wq.rows, q, T},
                      {&L->wk, L->wk.data, P + qd, W, L->wk.rows, q, T},
                      {&L->wv, L->wv.data, P + qd + kvd, W, L->wv.rows, q, T}};
    mm_multi(jobs, 3);
    }
    uint16_t *kc = m->kc + (size_t)ai * c->n_ctx * kvd, *vc = m->vc + (size_t)ai * c->n_ctx * kvd;
    for (int t = 0; t < T; t++) {
        float *Q = P + (size_t)t * W, *Kv = Q + qd, *V = Kv + kvd;
        for (int h = 0; h < NH; h++) rmsnorm(Q + h * HD, Q + h * HD, L->q_norm, HD, c->eps);
        for (int h = 0; h < NKV; h++) rmsnorm(Kv + h * HD, Kv + h * HD, L->k_norm, HD, c->eps);
        rope(Q, NH, HD, pos0 + t, m->rope_inv);
        rope(Kv, NKV, HD, pos0 + t, m->rope_inv);
        f32_to_f16(kc + (size_t)(pos0 + t) * kvd, Kv, kvd);
        f32_to_f16(vc + (size_t)(pos0 + t) * kvd, V, kvd);
    }
    const double t0 = lnow();
    const float scale = 1.0f / sqrtf((float)HD);
    float *att = s->att;                       // [T][qd]
    #pragma omp parallel for schedule(dynamic, 1)
    for (int th = 0; th < T * NH; th++) {
        const int t = th / NH, h = th % NH, kvh = h / gqa, S = pos0 + t + 1;
        const float *Q = P + (size_t)t * W + h * HD;
        float *sc = s->sc + (size_t)omp_get_thread_num() * (c->n_ctx + 16);
        float mx = -INFINITY;
        for (int r = 0; r < S; r++) {
            const uint16_t *k = kc + (size_t)r * kvd + kvh * HD;
            __m512 acc = _mm512_setzero_ps();
            for (int i = 0; i < HD; i += 16) acc = _mm512_fmadd_ps(_mm512_loadu_ps(Q + i), f16x16(k + i), acc);
            const float l = _mm512_reduce_add_ps(acc) * scale;
            sc[r] = l;
            if (l > mx) mx = l;
        }
        float sum = 0;
        for (int r = 0; r < S; r++) { sc[r] = expf(sc[r] - mx); sum += sc[r]; }
        __m512 o[16];
        const int nv = HD / 16;
        for (int i = 0; i < nv; i++) o[i] = _mm512_setzero_ps();
        for (int r = 0; r < S; r++) {
            const uint16_t *v = vc + (size_t)r * kvd + kvh * HD;
            const __m512 a = _mm512_set1_ps(sc[r]);
            for (int i = 0; i < nv; i++) o[i] = _mm512_fmadd_ps(a, f16x16(v + 16 * i), o[i]);
        }
        const __m512 inv = _mm512_set1_ps(1.0f / sum);
        float *ot = att + (size_t)t * qd + h * HD;
        for (int i = 0; i < nv; i++) _mm512_storeu_ps(ot + 16 * i, _mm512_mul_ps(o[i], inv));
    }
    lfm_prof[6] += lnow() - t0;
    if (mm_batched(T)) { mmf(&L->wo, L->wo.data, att, qd, T, out, D); return; }
    for (int t = 0; t < T; t++) q[t] = qset_set(&s->qb, t, att + (size_t)t * qd, qd);
    mm1(&L->wo, out, D, q, T);
}

// dense SwiGLU FFN for T tokens: XN in, OUT [T][D]
static void dense_ffn(lfm_model *m, const lfm_layer *L, scratch *s, int T, float *out) {
    const lfm_cfg *c = &m->c;
    const int D = c->n_embd, F = L->ffn_gate.rows;
    qin *q[TMAX];
    if (mm_batched(T)) {
        mmf(&L->ffn_gate, L->ffn_gate.data, s->xn, D, T, s->big, F);
        mmf(&L->ffn_up, L->ffn_up.data, s->xn, D, T, s->big2, F);
        for (int t = 0; t < T; t++) silu_mul(s->big + (size_t)t * F, s->big2 + (size_t)t * F, F);
        mmf(&L->ffn_down, L->ffn_down.data, s->big, F, T, out, D);
        return;
    }
    for (int t = 0; t < T; t++) q[t] = qset_set(&s->qa, t, s->xn + (size_t)t * D, D);
    mm_job jobs[2] = {{&L->ffn_gate, L->ffn_gate.data, s->big, F, F, q, T}, {&L->ffn_up, L->ffn_up.data, s->big2, F, F, q, T}};
    mm_multi(jobs, 2);
    for (int t = 0; t < T; t++) {
        silu_mul(s->big + (size_t)t * F, s->big2 + (size_t)t * F, F);
        q[t] = qset_set(&s->qb, t, s->big + (size_t)t * F, F);
    }
    mm1(&L->ffn_down, out, D, q, T);
}

// sigmoid router with selection bias; weights are the unbiased probabilities, normalized
static void route(const lfm_cfg *c, const lfm_layer *L, const float *x, int K, int *sel, float *w) {
    const int E = c->n_expert, D = c->n_embd;
    float prob[256], score[256];
    for (int e = 0; e < E; e++) {
        prob[e] = 1.0f / (1.0f + expf(-dotf(L->router + (size_t)e * D, x, D)));
        score[e] = prob[e] + L->probs_b[e];
    }
    float wsum = 0;
    for (int j = 0; j < K; j++) {
        int best = -1;
        for (int e = 0; e < E; e++) {
            int taken = 0;
            for (int q = 0; q < j; q++) taken |= sel[q] == e;
            if (!taken && (best < 0 || score[e] > score[best])) best = e;
        }
        sel[j] = best;
        w[j] = prob[best];
        wsum += w[j];
    }
    if (wsum < 6.103515625e-5f) wsum = 6.103515625e-5f;
    for (int j = 0; j < K; j++) w[j] = w[j] / wsum * c->w_scale;
}

// MoE for T tokens: XN in, OUT [T][D]. Experts are processed in groups of up to n_used
// (pinned while used), the ones already in RAM first.
static void moe(lfm_model *m, int l, scratch *s, int T, float *out) {
    const lfm_cfg *c = &m->c;
    const lfm_layer *L = &m->layers[l];
    const int D = c->n_embd, F = c->n_ff_exp, K = c->n_used, E = c->n_expert;
    double t0 = lnow();
    for (int t = 0; t < T; t++) route(c, L, s->xn + (size_t)t * D, K, s->sel + t * 16, s->w + t * 16);
    if (m->prefetch_on && T == 1 && l > c->n_dense) {
        for (int j = 0; j < K; j++)
            for (int q = 0; q < K; q++) m->pred_correct += m->pred[l * 16 + q] == s->sel[j];
        m->pred_total += K;
    }
    // experts in use, and which (token, slot) pairs use each
    int used[256], nu = 0, cnt[256] = {0};
    for (int t = 0; t < T; t++)
        for (int j = 0; j < K; j++) { const int e = s->sel[t * 16 + j]; if (!cnt[e]++) used[nu++] = e; }
    (void)E;
    lfm_prof[3] += lnow() - t0;
    t0 = lnow();
    qin *qx[TMAX];
    for (int t = 0; t < T; t++) qx[t] = qset_set(&s->qa, t, s->xn + (size_t)t * D, D);
    const int G = K;   // experts per pinned group
    for (int g0 = 0; g0 < nu; g0 += G) {
        const int ng = nu - g0 < G ? nu - g0 : G;
        const int *grp = used + g0;
        experts_begin_layer(m->experts, l, grp, ng);
        if (g0 + G < nu) experts_prefetch(m->experts, l, used + g0 + G, nu - g0 - G < G ? nu - g0 - G : G);
        else if (T == 1 && m->prefetch_on && l + 1 < c->n_layer && m->layers[l + 1].is_moe) {
            // guess the next layer's experts from this layer's normed input; prefetch the top 2
            float pw[16];
            route(c, &m->layers[l + 1], s->xn, K, m->pred + (l + 1) * 16, pw);
            static int pf_k = -1;
            if (pf_k < 0) pf_k = getenv("QUARRY_PF_K") ? atoi(getenv("QUARRY_PF_K")) : 2;
            experts_prefetch(m->experts, l + 1, m->pred + (l + 1) * 16, pf_k < K ? pf_k : K);
        }
        int done[16] = {0};
        for (int nd = 0; nd < ng;) {
            int bl[16], nb = 0;
            for (int k = 0; k < ng; k++) if (!done[k] && experts_ready(m->experts, l, grp[k])) bl[nb++] = k;
            if (!nb) {
                int rest[16], nr = 0;
                for (int k = 0; k < ng; k++) if (!done[k]) rest[nr++] = grp[k];
                experts_wait_any(m->experts, l, rest, nr);
                continue;
            }
            // all ready experts of the group together: gate+up in one region, down in another
            int tk[16][TMAX], sl[16][TMAX], cntb[16], off[17];
            const uint8_t *gw[16], *uw[16], *dw[16];
            qin *qi[16][TMAX];
            mm_job jobs[32];
            int nj = 0, big_e[16];
            off[0] = 0;
            for (int b = 0; b < nb; b++) {
                const int e = grp[bl[b]];
                experts_get(m->experts, l, e, &gw[b], &uw[b], &dw[b]);
                int n = 0;   // tokens routed to e, and their slot
                for (int t = 0; t < T; t++)
                    for (int j = 0; j < K; j++) if (s->sel[t * 16 + j] == e) { tk[b][n] = t; sl[b][n++] = j; }
                cntb[b] = n;
                off[b + 1] = off[b] + n;
                big_e[b] = mm_batched(n);
                if (big_e[b]) {   // many tokens: gather their inputs, batched matmul
                    for (int i = 0; i < n; i++) memcpy(s->gx + (size_t)i * D, s->xn + (size_t)tk[b][i] * D, D * 4);
                    mmf(&L->gate_exps, gw[b], s->gx, D, n, s->eg + (size_t)off[b] * F, F);
                    mmf(&L->up_exps, uw[b], s->gx, D, n, s->eh + (size_t)off[b] * F, F);
                    continue;
                }
                for (int i = 0; i < n; i++) qi[b][i] = qx[tk[b][i]];
                jobs[nj++] = (mm_job){&L->gate_exps, gw[b], s->eg + (size_t)off[b] * F, F, F, qi[b], n};
                jobs[nj++] = (mm_job){&L->up_exps, uw[b], s->eh + (size_t)off[b] * F, F, F, qi[b], n};
            }
            if (nj) mm_multi(jobs, nj);
            nj = 0;
            for (int b = 0; b < nb; b++) {
                for (int i = 0; i < cntb[b]; i++) {
                    float *g = s->eg + (size_t)(off[b] + i) * F;
                    silu_mul(g, s->eh + (size_t)(off[b] + i) * F, F);
                    if (!big_e[b]) qi[b][i] = qset_set(&s->qb, off[b] + i, g, F);
                }
                if (big_e[b]) mmf(&L->down_exps, dw[b], s->eg + (size_t)off[b] * F, F, cntb[b], s->eob + (size_t)off[b] * D, D);
                else jobs[nj++] = (mm_job){&L->down_exps, dw[b], s->eob + (size_t)off[b] * D, D, D, qi[b], cntb[b]};
            }
            if (nj) mm_multi(jobs, nj);
            for (int b = 0; b < nb; b++) {
                for (int i = 0; i < cntb[b]; i++)
                    memcpy(s->eo + ((size_t)tk[b][i] * K + sl[b][i]) * D, s->eob + (size_t)(off[b] + i) * D, D * 4);
                experts_release(m->experts, l, grp[bl[b]]);
                done[bl[b]] = 1;
                nd++;
            }
        }
    }
    // weighted sum in routing order (same result whatever order the experts arrived in)
    for (int t = 0; t < T; t++) {
        float *o = out + (size_t)t * D;
        memset(o, 0, D * 4);
        for (int j = 0; j < K; j++) {
            const float wj = s->w[t * 16 + j], *eo = s->eo + ((size_t)t * K + j) * D;
            for (int i = 0; i < D; i++) o[i] += wj * eo[i];
        }
    }
    lfm_prof[4] += lnow() - t0;
}

// logits from the 4-bit copy (w = d * (q - 8 + 0.5), x as u8 + 128 per 256 values), then the
// exact logits of the top candidates from the real rows
static void out_logits4(lfm_model *m, qin *q, float *lg) {
    const lfm_cfg *c = &m->c;
    const int D = c->n_embd, nr = c->n_vocab, nb = D / 64, rb = m->out4_rb;
    const qmat *O = &m->tok_embd;
    static uint8_t *xu;
    static float *dx, *xsum;
    if (!xu) { xu = xalloc(D); dx = xalloc(D / 256 * 4 + 64); xsum = xalloc(nb * 4 + 64); }
    quantize_row_u8(q->x, xu, dx, D);
    for (int b = 0; b < nb; b++) { int sacc = 0; for (int i = 0; i < 64; i++) sacc += xu[b * 64 + i] - 128; xsum[b] = sacc * 0.5f * dx[b / 4]; }
    const __m512i c128 = _mm512_set1_epi8((char)128), m4 = _mm512_set1_epi8(15), c8 = _mm512_set1_epi8(8);
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < nr; r++) {
        const uint8_t *row = m->out4 + (size_t)r * rb;
        __m512 acc = _mm512_setzero_ps();
        float off = 0;
        for (int b = 0; b < nb; b++) {
            float d; memcpy(&d, row + b * 36, 4);
            const __m256i p = _mm256_loadu_si256((const __m256i *)(row + b * 36 + 4));
            const __m512i pp = _mm512_inserti64x4(_mm512_castsi256_si512(p), _mm256_srli_epi16(p, 4), 1);
            const __m512i qv = _mm512_sub_epi8(_mm512_and_si512(pp, m4), c8);
            const __m512i corr = _mm512_dpbusd_epi32(_mm512_setzero_si512(), c128, qv);
            const __m512i xv = _mm512_loadu_si512((const void *)(xu + b * 64));
            const __m512i dp = _mm512_sub_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(), xv, qv), corr);
            acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(dp), _mm512_set1_ps(d * dx[b / 4]), acc);
            off += d * xsum[b];
        }
        lg[r] = _mm512_reduce_add_ps(acc) + off;
    }
    enum { NC = 64 };
    int cand[NC], nc = 0;
    for (int r = 0; r < nr; r++) {
        if (nc < NC || lg[r] > lg[cand[nc - 1]]) {
            int j = nc < NC ? nc++ : NC - 1;
            while (j > 0 && lg[cand[j - 1]] < lg[r]) { cand[j] = cand[j - 1]; j--; }
            cand[j] = r;
        }
    }
    qin_need(q, O->type);
    for (int j = 0; j < nc; j++) lg[cand[j]] = row_dot(O->type, O->data + (uint64_t)cand[j] * O->row_bytes, O->cols, q);
}

// ---------------- forward ----------------

// debug: QUARRY_DBG=prefix writes the last token's MoE inputs and layer outputs (prefix.norm-L.f32,
// prefix.l_out-L.f32) for comparison with llama.cpp (tests/ref_dump.c)
static const char *dbg_prefix(void) { static const char *p = (const char *)1; if (p == (const char *)1) p = getenv("QUARRY_DBG"); return p; }

// T tokens at positions pos0..; logits of the last one into m->logits when want_logits
static void forward_batch(lfm_model *m, const int *tokens, int T, int pos0, int want_logits) {
    const lfm_cfg *c = &m->c;
    scratch *s = scr(m);
    const int D = c->n_embd;
    const qmat *E = &m->tok_embd;
    for (int t = 0; t < T; t++) dequant_row(E->type, E->data + (uint64_t)tokens[t] * E->row_bytes, s->x + (size_t)t * D, D);
    for (int l = 0; l < c->n_layer; l++) {
        const lfm_layer *L = &m->layers[l];
        for (int t = 0; t < T; t++) rmsnorm(s->xn + (size_t)t * D, s->x + (size_t)t * D, L->attn_norm, D, c->eps);
        double t0 = lnow();
        if (L->is_attn) { attention(m, L, m->attn_idx[l], s, T, pos0, s->o); lfm_prof[1] += lnow() - t0; }
        else { shortconv(m, L, m->conv_idx[l], s, T, s->o); lfm_prof[0] += lnow() - t0; }
        for (size_t i = 0; i < (size_t)T * D; i++) s->x[i] += s->o[i];
        for (int t = 0; t < T; t++) rmsnorm(s->xn + (size_t)t * D, s->x + (size_t)t * D, L->ffn_norm, D, c->eps);
        if (want_logits && dbg_prefix()) {
            char path[512];
            snprintf(path, sizeof path, "%s.norm-%d.f32", dbg_prefix(), l);
            FILE *f = fopen(path, "wb");
            if (f) { fwrite(s->xn + (size_t)(T - 1) * D, 4, D, f); fclose(f); }
        }
        if (L->is_moe) moe(m, l, s, T, s->o);
        else { t0 = lnow(); dense_ffn(m, L, s, T, s->o); lfm_prof[2] += lnow() - t0; }
        for (size_t i = 0; i < (size_t)T * D; i++) s->x[i] += s->o[i];
        if (want_logits && dbg_prefix()) {   // debug: the last token's layer outputs
            char path[512];
            snprintf(path, sizeof path, "%s.l_out-%d.f32", dbg_prefix(), l);
            FILE *f = fopen(path, "wb");
            if (f) { fwrite(s->x + (size_t)(T - 1) * D, 4, D, f); fclose(f); }
        }
    }
    if (!want_logits) return;
    const double t0 = lnow();
    float *xn = s->xn;
    rmsnorm(xn, s->x + (size_t)(T - 1) * D, m->output_norm, D, c->eps);
    qin *q[1] = {qset_set(&s->qa, 0, xn, D)};
    if (m->out4) out_logits4(m, q[0], m->logits);
    else mm1(E, m->logits, 0, q, 1);
    lfm_prof[5] += lnow() - t0;
}

double lfm_total;
float *lfm_forward(lfm_model *m, int token, int pos) {
    const double t0 = lnow();
    forward_batch(m, &token, 1, pos, 1);
    lfm_total += lnow() - t0;
    return m->logits;
}

float *lfm_prefill(lfm_model *m, const int *tokens, int n, int pos0) {
    static int tb = 0;
    if (!tb) { tb = getenv("QUARRY_PF_T") ? atoi(getenv("QUARRY_PF_T")) : TMAX; if (tb < 1) tb = 1; if (tb > TMAX) tb = TMAX; }
    for (int i = 0; i < n; i += tb) {
        const int T = n - i < tb ? n - i : tb;
        forward_batch(m, tokens + i, T, pos0 + i, i + T == n);
    }
    return m->logits;
}
