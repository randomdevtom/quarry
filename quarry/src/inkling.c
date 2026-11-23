// Inkling 2 forward pass, one token at a time. Mirrors llama.cpp's
// src/models/inkling2.cpp and inkling2-loop.cpp (see inkling.h for the outline).
#define _GNU_SOURCE
#include "inkling.h"

#include <immintrin.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>

#include "quants.h"
#include <time.h>
double ink_prof[8];   // 0 attn+proj, 1 moe compute, 2 output, 3 mix/control
static double inow(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

static void *xalloc(size_t n) {
    // big buffers on 2MB huge pages: weights are streamed, 4KB pages cost a TLB walk each
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
    if (!t && required) { fprintf(stderr, "inkling: missing tensor %s\n", name); exit(1); }
    return t;
}

static float *f32(qg_file *g, const char *name, int required) {
    const qg_tensor *t = find(g, name, required);
    if (!t) return NULL;
    if (t->type != GT_F32) { fprintf(stderr, "inkling: %s is not F32\n", name); exit(1); }
    float *p = xalloc(t->nbytes);
    read_at(g, p, t->nbytes, t->offset);
    return p;
}

static qmat mat(qg_file *g, const char *name, int in_ram) {
    const qg_tensor *t = find(g, name, 1);
    qmat q = {t->type, (int)(t->ne[1] * t->ne[2]), (int)t->ne[0], qg_row_bytes(t->type, t->ne[0]), NULL, t->offset};
    if (in_ram) { q.data = xalloc(t->nbytes); read_at(g, q.data, t->nbytes, t->offset); }
    return q;
}

int ink_detect(const char *path) {
    qg_file g;
    if (qg_open(&g, path)) return 0;
    const char *a = qg_get_str(&g, "general.architecture");
    int r = a && !strcmp(a, "inkling2_loop");
    qg_close(&g);
    return r;
}

static int arr_i32(qg_file *g, const char *key, int i) {
    for (int k = 0; k < g->n_kv; k++)
        if (!strcmp(g->kv[k].key, key) && g->kv[k].type == GV_ARR) return qg_arr_int(&g->kv[k], i);
    return -1;
}

int ink_load(ink_model *m, const char *path, int n_ctx) {
    memset(m, 0, sizeof *m);
    if (qg_open(&m->g, path)) return -1;
    qg_file *g = &m->g;
    const char *arch = qg_get_str(g, "general.architecture");
    if (!arch || strcmp(arch, "inkling2_loop")) { fprintf(stderr, "inkling: unsupported arch %s\n", arch); return -1; }
    ink_cfg *c = &m->c;
#define K(s) "inkling2_loop." s
    c->n_phys = qg_get_int(g, K("block_count"), 0);
    c->n_embd = qg_get_int(g, K("embedding_length"), 0);
    c->n_head = arr_i32(g, K("attention.head_count"), 0);
    c->n_kv = arr_i32(g, K("attention.head_count_kv"), 0);
    c->hd = qg_get_int(g, K("attention.key_length"), 128);
    c->n_expert = qg_get_int(g, K("expert_count"), 0);
    c->n_used = qg_get_int(g, K("expert_used_count"), 0);
    if (getenv("QUARRY_TOPK") && atoi(getenv("QUARRY_TOPK")) > 0 && atoi(getenv("QUARRY_TOPK")) < c->n_used) c->n_used = atoi(getenv("QUARRY_TOPK"));   // lossy
    c->n_ff = qg_get_int(g, K("expert_feed_forward_length"), 0);
    c->n_vocab = qg_get_int(g, K("vocab_size_unpadded"), 0);
    c->n_vocab_pad = qg_get_int(g, K("vocab_size"), 0);
    c->loop_start = qg_get_int(g, K("loop.start"), 0);
    c->loop_end = qg_get_int(g, K("loop.end"), 0);
    c->n_passes = qg_get_int(g, K("loop.count"), 1);
    c->gate_rank = qg_get_int(g, K("loop.gate_rank"), 64);
    c->block = qg_get_int(g, K("attn_res.block_size"), 6);
    c->n_conv = qg_get_int(g, K("shortconv.l_cache"), 4);
    c->n_rel = qg_get_int(g, K("attention.relative_length"), 16);
    c->rel_extent = qg_get_int(g, K("attention.relative_extent"), 1024);
    c->n_swa = qg_get_int(g, K("attention.sliding_window"), 1024);
    c->log_floor = qg_get_int(g, K("attention.log_scale_floor"), 8192);
    c->eps = qg_get_float(g, K("attention.layer_norm_rms_epsilon"), 1e-6);
    c->qk_clip = qg_get_float(g, K("attention.qk_clip"), 0);
    c->rel_clip_thr = qg_get_float(g, K("attention.relative_clip"), 0);
    c->log_alpha = qg_get_float(g, K("attention.log_scale_alpha"), 0);
    c->w_scale = qg_get_float(g, K("expert_weights_scale"), 1);
    c->situ_beta = qg_get_float(g, K("activation.situ_beta"), 4);
    c->situ_lb = qg_get_float(g, K("activation.situ_linear_beta"), 0);
    c->softcap = qg_get_float(g, K("final_logit_softcapping"), 0);
    c->n_ctx = n_ctx;
    c->n_slots = c->n_phys + (c->n_passes - 1) * (c->loop_end - c->loop_start);
    if (c->n_slots > INK_MAX_SLOTS || c->n_conv != 4) { fprintf(stderr, "inkling: unsupported shape\n"); return -1; }

    m->tok_embd = mat(g, "token_embd.weight", 0);
    m->output = mat(g, "output.weight", 1);
    m->tok_norm = f32(g, "token_embd_norm.weight", 1);
    m->output_norm = f32(g, "output_norm.weight", 1);
    m->output_res_query = f32(g, "output_res_query.weight", 1);
    m->layers = calloc(c->n_phys, sizeof(ink_layer));
    char n[128];
#define NM(s) (snprintf(n, sizeof n, "blk.%d.%s", l, s), n)
    for (int l = 0; l < c->n_phys; l++) {
        ink_layer *L = &m->layers[l];
        L->is_swa = arr_i32(g, K("attention.sliding_window_pattern"), l) != 0;
        if (L->is_swa) L->attn_norm = f32(g, NM("attn_norm.weight"), 1);
        else {
            L->q_inp_norm = f32(g, NM("attn_q_inp_norm.weight"), 1);
            L->k_inp_norm = f32(g, NM("attn_k_inp_norm.weight"), 1);
            L->v_inp_norm = f32(g, NM("attn_v_inp_norm.weight"), 1);
            L->k_conv_proj = f32(g, NM("attn_k_conv_proj.weight"), 1);
            L->v_conv_proj = f32(g, NM("attn_v_conv_proj.weight"), 1);
        }
        L->wq = mat(g, NM("attn_q.weight"), 1);
        L->wk = mat(g, NM("attn_k.weight"), 1);
        L->wv = mat(g, NM("attn_v.weight"), 1);
        L->wo = mat(g, NM("attn_output.weight"), 1);
        L->wgate = mat(g, NM("attn_gate.weight"), 1);
        L->wrel = mat(g, NM("attn_rel.weight"), 1);
        L->q_clip = f32(g, NM("attn_q_clip.weight"), 1);
        L->sink_proj = f32(g, NM("attn_sink_proj.weight"), 1);
        L->rel_bias = f32(g, NM("attn_rel_bias.weight"), 1);
        L->rel_clip = f32(g, NM("attn_rel_clip.weight"), 1);
        L->k_conv = f32(g, NM("attn_k_conv.weight"), 1);
        L->v_conv = f32(g, NM("attn_v_conv.weight"), 1);
        L->post_conv = f32(g, NM("attn_post_conv.weight"), 1);
        L->ffn_post_conv = f32(g, NM("ffn_post_conv.weight"), 1);
        const qg_tensor *rq = find(g, NM("attn_res_query.weight"), 0);
        if (rq) { L->attn_res_query = f32(g, NM("attn_res_query.weight"), 1); L->n_res_q = (int)rq->ne[1]; }
        L->ffn_res_query = f32(g, NM("ffn_res_query.weight"), 1);
        L->ffn_norm = f32(g, NM("ffn_norm.weight"), 1);
        L->router = f32(g, NM("ffn_gate_inp.weight"), 1);
        L->probs_b = f32(g, NM("exp_probs_b.bias"), 0);
        L->gate_up = mat(g, NM("ffn_gate_up_exps.weight"), 0);
        L->down = mat(g, NM("ffn_down_exps.weight"), 0);
        L->gate_up.rows /= c->n_expert;   // rows per expert
        L->down.rows /= c->n_expert;
    }
    const int nl = c->loop_end - c->loop_start;
    m->loops = calloc(nl, sizeof(ink_loop));
    const char *cn[4] = {"q", "k", "v", "ffn"};
    for (int l = c->loop_start; l < c->loop_end; l++) {
        ink_loop *P = &m->loops[l - c->loop_start];
        P->msg_query = f32(g, NM("loop_msg_query.weight"), 1);
        for (int k = 0; k < 4; k++) {
            char s[64];
            snprintf(s, sizeof s, "loop_%s_ctrl_down.weight", cn[k]);   P->ctrl_down[k] = f32(g, NM(s), 1);
            snprintf(s, sizeof s, "loop_%s_ctrl_gain.weight", cn[k]);   P->ctrl_gain[k] = f32(g, NM(s), 1);
            snprintf(s, sizeof s, "loop_%s_ctrl_forget.weight", cn[k]); P->ctrl_forget[k] = f32(g, NM(s), 1);
        }
    }
#undef NM
#undef K
    // slots: KV caches (sliding-window layers keep only the window) and conv state
    const int kvd = c->n_kv * c->hd, conv_ch = 2 * kvd + 2 * c->n_embd;
    for (int s = 0; s < c->n_slots; s++) {
        const int p = s < c->n_phys ? s : c->loop_start + (s - c->n_phys) % nl;
        m->slot_phys[s] = p;
        int cap = m->layers[p].is_swa ? c->n_swa : n_ctx;
        if (cap > n_ctx) cap = n_ctx;
        m->slot_cap[s] = cap;
        m->kc[s] = xalloc((size_t)cap * kvd * 4);
        m->vc[s] = xalloc((size_t)cap * kvd * 4);
        m->conv[s] = xalloc((size_t)conv_ch * (c->n_conv - 1) * 4);
    }
    m->logits = xalloc((size_t)c->n_vocab_pad * 4);

    // experts: two parts, gate_up and down, whose types differ by layer
    uint64_t *off = malloc(c->n_phys * 2 * sizeof(uint64_t)), *bytes = malloc(c->n_phys * 2 * sizeof(uint64_t));
    for (int l = 0; l < c->n_phys; l++) {
        const ink_layer *L = &m->layers[l];
        off[l * 2] = L->gate_up.file_off;     bytes[l * 2] = L->gate_up.row_bytes * L->gate_up.rows;
        off[l * 2 + 1] = L->down.file_off;    bytes[l * 2 + 1] = L->down.row_bytes * L->down.rows;
    }
    expert_src src = {g->fd, g->file_size, c->n_phys, c->n_expert, 2, off, bytes,
                      &m->expert_bytes_read, &m->expert_loads, 2 * c->n_used};
    m->experts = experts_open_src(&src);
    free(off); free(bytes);
    m->prefetch_on = getenv("QUARRY_NO_PREFETCH") ? 0 : 1;
    for (int l = 0; l < c->loop_start; l++) m->seq[m->n_steps++] = l;
    for (int p = 0; p < c->n_passes; p++)
        for (int l = c->loop_start; l < c->loop_end; l++) m->seq[m->n_steps++] = l;
    for (int l = c->loop_end; l < c->n_phys; l++) m->seq[m->n_steps++] = l;
    return 0;
}

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

// o = x / rms(x) * w (w may be NULL)
static void rmsnorm(float *o, const float *x, const float *w, int n, float eps) {
    double ss = 0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    const float s = 1.0f / sqrtf((float)(ss / n) + eps);
    for (int i = 0; i < n; i++) o[i] = x[i] * s * (w ? w[i] : 1.0f);
}

// y[rows] = W x for an F32 matrix [rows][cols]
static void mv_f32(const float *W, int rows, int cols, const float *x, float *y) {
    #pragma omp parallel for schedule(static) if (rows * cols >= 65536)
    for (int r = 0; r < rows; r++) y[r] = dotf(W + (size_t)r * cols, x, cols);
}

// quantized activations of one input vector, in both layouts, made on demand
typedef struct {
    const float *x;
    int n, have_k, have_32;
    block_q8_K *qk;
    act32 a;
} qin;

static void qin_init(qin *q, const float *x, int n, block_q8_K *kbuf, int8_t *qs, float *d, int32_t *s) {
    q->x = x; q->n = n; q->have_k = q->have_32 = 0;
    q->qk = kbuf; q->a = (act32){qs, d, s};
}

static void qin_need(qin *q, int type) {
    if (type == GT_Q8_0 || type == GT_Q5_0) {
        if (!q->have_32) { quantize_row_a32(q->x, q->a, q->n); q->have_32 = 1; }
    } else if (!q->have_k) { quantize_row_q8_K(q->x, q->qk, q->n); q->have_k = 1; }
}

static inline float row_dot(int type, const uint8_t *row, int n, const qin *q) {
    switch (type) {
    case GT_Q8_0: return vec_dot_q8_0_a32(n, (const block_q8_0 *)row, q->a);
    case GT_Q5_0: return vec_dot_q5_0_a32(n, (const block_q5_0 *)row, q->a);
    case GT_Q3_K: return vec_dot_q3_K_q8_K(n, (const block_q3_K *)row, q->qk);
    case GT_Q4_K: return vec_dot_q4_K_q8_K(n, (const block_q4_K *)row, q->qk);
    case GT_Q5_K: return vec_dot_q5_K_q8_K(n, (const block_q5_K *)row, q->qk);
    case GT_Q6_K: return vec_dot_q6_K_q8_K(n, (const block_q6_K *)row, q->qk);
    }
    fprintf(stderr, "inkling: no dot for type %d\n", type);
    exit(1);
}

// y = W x for rows of a quantized matrix (data pointer, row_bytes)
static void mv_q(int type, const uint8_t *w, uint64_t rb, int rows, int cols, qin *q, float *y) {
    qin_need(q, type);
    #pragma omp parallel for schedule(static) if (rows >= 128)
    for (int r = 0; r < rows; r++) y[r] = row_dot(type, w + (uint64_t)r * rb, cols, q);
}

static float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }

// exp for |x| <= 87, ~1e-7 relative error
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
// tanh(x) = 1 - 2 / (exp(2x) + 1)
static inline __m512 tanh_ps(__m512 x) {
    const __m512 one = _mm512_set1_ps(1.0f);
    return _mm512_sub_ps(one, _mm512_div_ps(_mm512_set1_ps(2.0f), _mm512_add_ps(exp_ps(_mm512_add_ps(x, x)), one)));
}
// SITU: beta*tanh(g/beta)*sigmoid(g) * lb*tanh(u/lb), n a multiple of 16
static void situ(const float *g, const float *u, float *o, int n, float beta, float lb) {
    const __m512 vb = _mm512_set1_ps(beta), vib = _mm512_set1_ps(1.0f / beta), one = _mm512_set1_ps(1.0f);
    const __m512 vl = _mm512_set1_ps(lb), vil = _mm512_set1_ps(lb > 0 ? 1.0f / lb : 0.0f);
    for (int i = 0; i < n; i += 16) {
        const __m512 gv = _mm512_loadu_ps(g + i), uv = _mm512_loadu_ps(u + i);
        const __m512 sg = _mm512_div_ps(one, _mm512_add_ps(one, exp_ps(_mm512_sub_ps(_mm512_setzero_ps(), gv))));
        const __m512 a = _mm512_mul_ps(_mm512_mul_ps(vb, tanh_ps(_mm512_mul_ps(gv, vib))), sg);
        const __m512 b = lb > 0 ? _mm512_mul_ps(vl, tanh_ps(_mm512_mul_ps(uv, vil))) : uv;
        _mm512_storeu_ps(o + i, _mm512_mul_ps(a, b));
    }
}

// scale each head's vector down to L2 norm <= thr, after the optional per-channel scale
static void clip(float *v, const float *scale, int dim, int heads, float thr) {
    for (int h = 0; h < heads; h++) {
        float *x = v + h * dim;
        if (scale) for (int i = 0; i < dim; i++) x[i] *= scale[h * dim + i];
        double ss = 0;
        for (int i = 0; i < dim; i++) ss += (double)x[i] * x[i];
        const float norm = sqrtf((float)ss) + 1e-6f;
        float f = thr / norm;
        if (f > 1) f = 1;
        if (f < 0) f = 0;
        for (int i = 0; i < dim; i++) x[i] *= f;
    }
}

// attention over depth: states = blocks + [cur], rms-normed; out_q = softmax(states . query_q / sqrt(D)) . states
static void mix(int D, float *const *blocks, int nb, const float *cur, const float *queries, int nq, float *out) {
    float *st = malloc(sizeof(float) * D * (nb + 1));
    for (int i = 0; i < nb; i++) rmsnorm(st + (size_t)i * D, blocks[i], NULL, D, 1e-6f);
    rmsnorm(st + (size_t)nb * D, cur, NULL, D, 1e-6f);
    const float sc = 1.0f / sqrtf((float)D);
    for (int q = 0; q < nq; q++) {
        float w[16], mx = -1e30f, sum = 0;
        for (int i = 0; i <= nb; i++) { w[i] = dotf(st + (size_t)i * D, queries + (size_t)q * D, D) * sc; if (w[i] > mx) mx = w[i]; }
        for (int i = 0; i <= nb; i++) { w[i] = expf(w[i] - mx); sum += w[i]; }
        float *o = out + (size_t)q * D;
        for (int k = 0; k < D; k++) o[k] = 0;
        for (int i = 0; i <= nb; i++) {
            const float a = w[i] / sum, *s = st + (size_t)i * D;
            for (int k = 0; k < D; k++) o[k] += a * s[k];
        }
    }
    free(st);
}

// (1 - forget) * cur + gain * msg, gated on [rmsnorm(cur), rmsnorm(msg)]
static void control(const ink_cfg *c, float *cur, const float *msg, const ink_loop *P, int k) {
    const int D = c->n_embd, R = c->gate_rank;
    float *gate = malloc(sizeof(float) * 2 * D), z[256];
    float *gain = malloc(sizeof(float) * D), *forget = malloc(sizeof(float) * D);
    rmsnorm(gate, cur, NULL, D, 1e-6f);
    rmsnorm(gate + D, msg, NULL, D, 1e-6f);
    mv_f32(P->ctrl_down[k], R, 2 * D, gate, z);
    mv_f32(P->ctrl_gain[k], D, R, z, gain);
    mv_f32(P->ctrl_forget[k], D, R, z, forget);
    for (int i = 0; i < D; i++) cur[i] = cur[i] - tanhf(forget[i]) * cur[i] + tanhf(gain[i]) * msg[i];
    free(gate); free(gain); free(forget);
}

// causal short convolution with residual: state holds the previous n_conv-1 inputs
// per channel (oldest first). dyn (optional): per-head, per-lag input-dependent taps.
static void convolve(float *x, int C, const float *w, const float *dyn, int heads, float *state) {
    const int nc = 4;
    for (int ch = 0; ch < C; ch++) {
        float *p = state + (size_t)ch * (nc - 1);           // inputs t-3, t-2, t-1
        const float *wc = w + (size_t)ch * nc;
        const float cur = x[ch];
        float out = wc[0] * p[0] + wc[1] * p[1] + wc[2] * p[2] + wc[3] * cur;
        if (dyn) {
            const float *tp = dyn + (ch / (C / heads)) * nc;   // tap[lag] multiplies input t-lag
            out += tp[0] * cur + tp[1] * p[2] + tp[2] * p[1] + tp[3] * p[0];
        }
        p[0] = p[1]; p[1] = p[2]; p[2] = cur;
        x[ch] = (cur + out) / sqrtf((dyn ? 2.0f : 1.0f) * nc + 1.0f);
    }
}

// debug: QUARRY_DBG=prefix writes prefix.<name>-<slot>.f32 (overwritten each token)
static void dbg(const char *name, int slot, const float *x, int n) {
    static const char *pre = (const char *)1;
    if (pre == (const char *)1) pre = getenv("QUARRY_DBG");
    if (!pre) return;
    char path[512];
    snprintf(path, sizeof path, "%s.%s-%d.f32", pre, name, slot);
    FILE *f = fopen(path, "wb");
    if (f) { fwrite(x, 4, n, f); fclose(f); }
}

// ---------------- one layer ----------------

typedef struct {
    float *blocks[8];
    int nb;
} blockset;

typedef struct {
    float *in[3], *q, *k, *v, *r, *dk, *dv, *sink, *att, *gate, *o, *tab, *sc, *mixed, *h, *moe, *eo, *gu, *act;
    block_q8_K *kq[3];
    int8_t *aq[3]; float *ad[3]; int32_t *as[3];
    float *gub, *actb, *eob, *eok, *adb; block_q8_K *kqb; int8_t *aqb; int32_t *asb;   // batched experts
} ink_scratch;

static ink_scratch *scratch(const ink_cfg *c) {
    static ink_scratch S;
    static int init;
    if (init) return &S;
    init = 1;
    const int D = c->n_embd, qd = c->n_head * c->hd, kvd = c->n_kv * c->hd, big = qd > D ? qd : D;
    for (int j = 0; j < 3; j++) {
        S.in[j] = xalloc(D * 4);
        S.kq[j] = xalloc(sizeof(block_q8_K) * (big / QK_K + 1));
        S.aq[j] = xalloc(big); S.ad[j] = xalloc(big / 32 * 4); S.as[j] = xalloc(big / 32 * 4);
    }
    S.q = xalloc(qd * 4); S.k = xalloc(kvd * 4); S.v = xalloc(kvd * 4);
    S.r = xalloc(c->n_head * c->n_rel * 4); S.dk = xalloc(256 * 4); S.dv = xalloc(256 * 4);
    S.sink = xalloc(c->n_head * 4); S.att = xalloc(qd * 4); S.gate = xalloc(qd * 4); S.o = xalloc(D * 4);
    S.tab = xalloc((size_t)c->n_head * (c->rel_extent + 1) * 4);
    S.sc = xalloc((size_t)c->n_head * (c->n_ctx + 1) * 4);
    S.mixed = xalloc(3 * D * 4); S.h = xalloc(D * 4); S.moe = xalloc(D * 4); S.eo = xalloc(D * 4);
    S.gu = xalloc(2 * c->n_ff * 4); S.act = xalloc(c->n_ff * 4);
    const int F = c->n_ff;
    S.gub = xalloc((size_t)16 * 2 * F * 4); S.actb = xalloc((size_t)16 * F * 4); S.eob = xalloc((size_t)16 * D * 4); S.eok = xalloc((size_t)16 * D * 4);
    S.kqb = xalloc(sizeof(block_q8_K) * 16 * (F / QK_K + 1));
    S.aqb = xalloc((size_t)16 * F); S.adb = xalloc((size_t)16 * (F / 32) * 4); S.asb = xalloc((size_t)16 * (F / 32) * 4);
    return &S;
}

// sigmoid router with selection bias: top-k experts and their normalized, scaled weights
static void route(const ink_cfg *c, const ink_layer *L, const float *x, int *sel, float *w) {
    const int E = c->n_expert, K = c->n_used, D = c->n_embd;
    float logit[256], prob[256], score[256];
    mv_f32(L->router, E, D, x, logit);
    for (int e = 0; e < E; e++) {
        prob[e] = sigmoidf(logit[e]);
        score[e] = prob[e] + (L->probs_b ? L->probs_b[e] : 0);
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
    for (int j = 0; j < K; j++) w[j] = w[j] / (wsum + 1e-20f) * c->w_scale;
}

// x: normalized MoE input; resid: the layer's residual stream (used to guess the next step)
static void moe(ink_model *m, int phys, const float *x, const float *resid, float *out) {
    const ink_cfg *c = &m->c;
    const ink_layer *L = &m->layers[phys];
    ink_scratch *S = scratch(c);
    const int D = c->n_embd, F = c->n_ff;
    int K = c->n_used;
    int sel[16];
    float w[16];
    route(c, L, x, sel, w);
    {   // optional, lossy: skip an expert that is not in RAM when its weight is small (QUARRY_SKIP=0.05)
        static float skip = -1;
        if (skip < 0) skip = getenv("QUARRY_SKIP") ? atof(getenv("QUARRY_SKIP")) : 0;
        if (skip > 0) {
            int nk = 0;
            float tot = 0, kept = 0;
            for (int j = 0; j < K; j++) tot += w[j];
            for (int j = 0; j < K; j++)
                if (w[j] / tot >= skip || experts_ready(m->experts, phys, sel[j])) { sel[nk] = sel[j]; w[nk++] = w[j]; kept += w[j]; }
                else m->skipped++;
            for (int j = 0; j < nk; j++) w[j] *= tot / kept;   // keep the total weight
            K = nk;
        }
    }
    {   // QUARRY_TRACE=file: per step, the chosen experts and the prediction made for it
        static FILE *tf = (FILE *)1;
        if (tf == (FILE *)1) tf = getenv("QUARRY_TRACE") ? fopen(getenv("QUARRY_TRACE"), "w") : NULL;
        if (tf) {
            fprintf(tf, "%d %d", m->step, phys);
            for (int j = 0; j < K; j++) fprintf(tf, " %d", sel[j]);
            fprintf(tf, " |");
            for (int j = 0; j < K; j++) fprintf(tf, " %d", m->pred[m->step][j]);
            fprintf(tf, "\n");
            fflush(tf);
        }
    }
    experts_begin_layer(m->experts, phys, sel, K);
    if (m->prefetch_on) {
        const int st = m->step;
        if (st > 0) {   // how good was the guess made one step ago; demote the wrong ones
            int wrong[16], nw = 0;
            for (int q = 0; q < K; q++) {
                int hit = 0;
                for (int j = 0; j < K; j++) hit |= m->pred[st][q] == sel[j];
                m->pred_correct += hit;
                if (!hit) wrong[nw++] = m->pred[st][q];
            }
            m->pred_total += K;
            for (int q = 0; q < K; q++) for (int j = 0; j < K; j++) m->pred2_correct += m->pred2[q] == sel[j];
            (void)wrong; (void)nw;   // demoting them to the LRU tail was measured worse (re-prefetched later)
        }
        if (st + 1 < m->n_steps) {
            // guess the next step's experts: its router on this residual stream, normalized its way
            const int np = m->seq[st + 1];
            float *xn = S->eo, pw[16];
            rmsnorm(xn, resid, m->layers[np].ffn_norm, D, c->eps);
            route(c, &m->layers[np], xn, m->pred[st + 1], pw);
            static int pf_k = -1;
            // only the top few guesses: wrong prefetches cost SSD bandwidth (measured best: 2)
            if (pf_k < 0) pf_k = getenv("QUARRY_PF_K") ? atoi(getenv("QUARRY_PF_K")) : 2;
            experts_prefetch(m->experts, np, m->pred[st + 1], pf_k < K ? pf_k : K);
        }
        static int pf_a = -1;   // also guess two steps ahead (more lead time for the SSD)
        if (pf_a < 0) pf_a = getenv("QUARRY_PF_AHEAD") ? atoi(getenv("QUARRY_PF_AHEAD")) : 0;
        if (pf_a > 0 && st + 2 < m->n_steps) {
            const int np2 = m->seq[st + 2];
            float *xn = S->eo, pw[16];
            int g2[16];
            rmsnorm(xn, resid, m->layers[np2].ffn_norm, D, c->eps);
            route(c, &m->layers[np2], xn, g2, pw);
            experts_prefetch(m->experts, np2, g2, pf_a < K ? pf_a : K);
        }
    }
    qin qx;
    qin_init(&qx, x, D, S->kq[0], S->aq[0], S->ad[0], S->as[0]);
    qin_need(&qx, L->gate_up.type);
    memset(out, 0, D * 4);
    int done[16] = {0}, ndone = 0;
    const int gt = L->gate_up.type, dt = L->down.type;
    const uint64_t grb = L->gate_up.row_bytes, drb = L->down.row_bytes;
    while (ndone < K) {
        // every expert already in RAM as one batch; if none, wait for the first missing one
        int bl[16], nb = 0;
        for (int q = 0; q < K; q++) if (!done[q] && experts_ready(m->experts, phys, sel[q])) bl[nb++] = q;
        if (!nb) for (int q = 0; q < K && !nb; q++) if (!done[q]) bl[nb++] = q;
        const uint8_t *gu[16], *dn[16], *unused;
        for (int j = 0; j < nb; j++) experts_get(m->experts, phys, sel[bl[j]], &gu[j], &dn[j], &unused);
        float *GU = S->gub;                          // [nb][2F]
        #pragma omp parallel for schedule(static)
        for (int r = 0; r < nb * 2 * F; r++) {
            const int j = r / (2 * F), rr = r % (2 * F);
            GU[r] = row_dot(gt, gu[j] + (uint64_t)rr * grb, D, &qx);
        }
        qin qa[16];
        for (int j = 0; j < nb; j++) {
            float *act = S->actb + (size_t)j * F;
            situ(GU + (size_t)j * 2 * F, GU + (size_t)j * 2 * F + F, act, F, c->situ_beta, c->situ_lb);
            qin_init(&qa[j], act, F, S->kqb + (size_t)j * (F / QK_K + 1), S->aqb + (size_t)j * F,
                     S->adb + (size_t)j * (F / 32), S->asb + (size_t)j * (F / 32));
            qin_need(&qa[j], dt);
        }
        float *EO = S->eob;                          // [nb][D]
        #pragma omp parallel for schedule(static)
        for (int r = 0; r < nb * D; r++) {
            const int j = r / D, rr = r % D;
            EO[r] = row_dot(dt, dn[j] + (uint64_t)rr * drb, F, &qa[j]);
        }
        for (int j = 0; j < nb; j++) {
            experts_release(m->experts, phys, sel[bl[j]]);
            memcpy(S->eok + (size_t)bl[j] * D, EO + (size_t)j * D, D * 4);
            done[bl[j]] = 1;
            ndone++;
        }
    }
    // sum in routing order, whatever order the experts arrived in (same result every run)
    for (int q = 0; q < K; q++) {
        const float wq = w[q], *eq = S->eok + (size_t)q * D;
        for (int i = 0; i < D; i++) out[i] += wq * eq[i];
    }
}

// one layer at slot `slot` (physical layer `ip`), position pos. msgs: [4][D] or NULL.
static void layer(ink_model *m, int slot, int ip, float *cur, blockset *B, const float *msgs, const ink_loop *P,
                  int pos, float tau) {
    const ink_cfg *c = &m->c;
    const ink_layer *L = &m->layers[ip];
    ink_scratch *S = scratch(c);
    const int D = c->n_embd, HD = c->hd, NH = c->n_head, NKV = c->n_kv, kvd = NKV * HD, qd = NH * HD;
    const int NR = c->n_rel, gqa = NH / NKV;
    float *inpSA = malloc(D * 4);
    memcpy(inpSA, cur, D * 4);
    const double t_a = inow();
    static int pf2 = -1;
    if (pf2 < 0) pf2 = getenv("QUARRY_PF2_K") ? atoi(getenv("QUARRY_PF2_K")) : 0;
    if (m->prefetch_on && (pf2 > 0 || getenv("QUARRY_STATS"))) {
        // second guess, before attention (off by default: extra reads measured slower)
        // approximate the MoE input: the same depth mix, with the attention output not known yet
        float pw[16];
        if (ip % c->block == 0) { B->blocks[B->nb] = cur; mix(D, B->blocks, B->nb + 1, cur, L->ffn_res_query, 1, S->h); }
        else mix(D, B->blocks, B->nb, cur, L->ffn_res_query, 1, S->h);
        rmsnorm(S->h, S->h, L->ffn_norm, D, c->eps);
        route(c, L, S->h, m->pred2, pw);
        if (pf2 > 0) experts_prefetch(m->experts, ip, m->pred2, pf2 < c->n_used ? pf2 : c->n_used);
    }

    // attention inputs
    if (L->attn_res_query) {
        mix(D, B->blocks, B->nb, cur, L->attn_res_query, L->n_res_q, S->mixed);
        for (int j = 0; j < 3; j++) memcpy(S->in[j], S->mixed + (L->is_swa ? 0 : j) * D, D * 4);
    } else for (int j = 0; j < 3; j++) memcpy(S->in[j], cur, D * 4);
    double t_c = inow();
    if (msgs) for (int j = 0; j < 3; j++) control(c, S->in[j], msgs + j * D, P, j);
    ink_prof[4] += inow() - t_c;
    if (L->is_swa && msgs) for (int j = 0; j < 3; j++) rmsnorm(S->in[j], S->in[j], L->attn_norm, D, c->eps);
    else if (L->is_swa) { rmsnorm(S->in[0], S->in[0], L->attn_norm, D, c->eps); memcpy(S->in[1], S->in[0], D * 4); memcpy(S->in[2], S->in[0], D * 4); }
    else {
        rmsnorm(S->in[0], S->in[0], L->q_inp_norm, D, c->eps);
        rmsnorm(S->in[1], S->in[1], L->k_inp_norm, D, c->eps);
        rmsnorm(S->in[2], S->in[2], L->v_inp_norm, D, c->eps);
    }
    qin qi[3];
    for (int j = 0; j < 3; j++) qin_init(&qi[j], S->in[j], D, S->kq[j], S->aq[j], S->ad[j], S->as[j]);

    // projections
    double t_p = inow();
    {   // q, k, v, rel, gate in one parallel region
        const qmat *W[5] = {&L->wq, &L->wk, &L->wv, &L->wrel, &L->wgate};
        float *Y[5] = {S->q, S->k, S->v, S->r, S->gate};
        const int src[5] = {0, 1, 2, 0, 0};
        int start[6] = {0};
        for (int k = 0; k < 5; k++) { qin_need(&qi[src[k]], W[k]->type); start[k + 1] = start[k] + W[k]->rows; }
        #pragma omp parallel for schedule(static)
        for (int r = 0; r < start[5]; r++) {
            int k = 0;
            while (r >= start[k + 1]) k++;
            const int rr = r - start[k];
            Y[k][rr] = row_dot(W[k]->type, W[k]->data + (uint64_t)rr * W[k]->row_bytes, D, &qi[src[k]]);
        }
    }
    ink_prof[5] += inow() - t_p;
    t_p = inow();
    mv_f32(L->sink_proj, NH, D, S->in[0], S->sink);
    float *st = m->conv[slot];
    if (!L->is_swa) {
        mv_f32(L->k_conv_proj, NKV * 4, D, S->in[1], S->dk);
        mv_f32(L->v_conv_proj, NKV * 4, D, S->in[2], S->dv);
    }
    convolve(S->k, kvd, L->k_conv, L->is_swa ? NULL : S->dk, NKV, st);
    convolve(S->v, kvd, L->v_conv, L->is_swa ? NULL : S->dv, NKV, st + (size_t)kvd * 3);
    clip(S->q, L->q_clip, HD, NH, c->qk_clip);
    clip(S->k, NULL, HD, NKV, c->qk_clip);
    for (int h = 0; h < NKV; h++) rmsnorm(S->v + h * HD, S->v + h * HD, NULL, HD, c->eps);
    clip(S->r, L->rel_clip, NR, NH, c->rel_clip_thr);
    if (!L->is_swa) {
        for (int i = 0; i < qd; i++) S->q[i] *= tau;
        for (int i = 0; i < NH * NR; i++) S->r[i] *= tau;
    }
    dbg("Qcur", slot, S->q, qd); dbg("Kcur", slot, S->k, kvd); dbg("Vcur", slot, S->v, kvd); dbg("Rcur", slot, S->r, NH * NR);
    dbg("in0", slot, S->in[0], D);

    ink_prof[6] += inow() - t_p;
    t_p = inow();
    // KV cache (ring buffer of slot_cap positions)
    const int cap = m->slot_cap[slot];
    memcpy(m->kc[slot] + (size_t)(pos % cap) * kvd, S->k, kvd * 4);
    memcpy(m->vc[slot] + (size_t)(pos % cap) * kvd, S->v, kvd * 4);
    const int extent = L->is_swa ? c->n_swa : c->rel_extent;
    int s0 = 0;
    if (L->is_swa && pos - (c->n_swa - 1) > 0) s0 = pos - (c->n_swa - 1);
    if (pos - s0 + 1 > cap) s0 = pos - cap + 1;
    const float scale = 1.0f / sqrtf((float)HD);
    dbg("kcache", slot, m->kc[slot], (pos + 1 < cap ? pos + 1 : cap) * kvd);
    dbg("vcache", slot, m->vc[slot], (pos + 1 < cap ? pos + 1 : cap) * kvd);
    dbg("sink", slot, S->sink, NH);

    #pragma omp parallel for schedule(static)
    for (int h = 0; h < NH; h++) {
        // relative bias table for this head: tab[idx] = rel_bias[idx] . r_h - sink_h, tab[extent] = -sink_h
        float *tab = S->tab + (size_t)h * (c->rel_extent + 1);
        const float *rh = S->r + h * NR;
        const int nt = pos + 1 < extent ? pos + 1 : extent;
        for (int i = 0; i < nt; i++) tab[i] = dotf(L->rel_bias + (size_t)i * NR, rh, NR) - S->sink[h];
        tab[extent] = -S->sink[h];
        const float *qh = S->q + h * HD;
        const int kvh = h / gqa;
        float *sc = S->sc + (size_t)h * (c->n_ctx + 1);
        float mx = 0;   // the sink logit is 0
        for (int s = s0; s <= pos; s++) {
            const int dist = pos - s, idx = dist < extent ? dist : extent;
            const float *ks = m->kc[slot] + (size_t)(s % cap) * kvd + kvh * HD;
            const float l = dotf(qh, ks, HD) * scale + tab[idx];   // bias and sink are not scaled
            sc[s - s0] = l;
            if (l > mx) mx = l;
        }
        float sum = expf(0 - mx);
        for (int s = s0; s <= pos; s++) { sc[s - s0] = expf(sc[s - s0] - mx); sum += sc[s - s0]; }
        if (h == NH - 1 && getenv("QUARRY_DBG")) {   // debug: probabilities of the last head, with the sink
            float pr[1100]; int np = 0;
            for (int s = s0; s <= pos && np < 1100; s++) pr[np++] = sc[s - s0] / sum;
            dbg("probs_last_head", slot, pr, np);
        }
        float *o = S->att + h * HD;
        for (int i = 0; i < HD; i++) o[i] = 0;
        for (int s = s0; s <= pos; s++) {
            const float a = sc[s - s0] / sum;
            const float *vs = m->vc[slot] + (size_t)(s % cap) * kvd + kvh * HD;
            for (int i = 0; i < HD; i++) o[i] += a * vs[i];
        }
    }
    ink_prof[7] += inow() - t_p;
    dbg("kqv_out", slot, S->att, qd);
    for (int i = 0; i < qd; i++) S->att[i] *= sigmoidf(S->gate[i]);
    qin qa;
    qin_init(&qa, S->att, qd, S->kq[0], S->aq[0], S->ad[0], S->as[0]);
    mv_q(L->wo.type, L->wo.data, L->wo.row_bytes, D, qd, &qa, S->o);
    dbg("kqv_gated", slot, S->att, qd);
    convolve(S->o, D, L->post_conv, NULL, 1, st + (size_t)kvd * 6);
    dbg("attn_out", slot, S->o, D);

    ink_prof[0] += inow() - t_a;
    // block residuals: the first layer of a block starts a new stream
    if (ip % c->block == 0) {
        B->blocks[B->nb++] = inpSA;
        memcpy(cur, S->o, D * 4);
    } else {
        for (int i = 0; i < D; i++) cur[i] = inpSA[i] + S->o[i];
        free(inpSA);
    }

    dbg("ffn_inp", slot, cur, D);
    // MoE
    double t_m = inow();
    mix(D, B->blocks, B->nb, cur, L->ffn_res_query, 1, S->h);
    if (msgs) control(c, S->h, msgs + 3 * D, P, 3);
    rmsnorm(S->h, S->h, L->ffn_norm, D, c->eps);
    ink_prof[3] += inow() - t_m;
    t_m = inow();
    moe(m, ip, S->h, cur, S->moe);
    m->step++;
    ink_prof[1] += inow() - t_m;
    convolve(S->moe, D, L->ffn_post_conv, NULL, 1, st + (size_t)kvd * 6 + (size_t)D * 3);
    dbg("ffn_out", slot, S->moe, D);
    for (int i = 0; i < D; i++) cur[i] += S->moe[i];
    if (m->dbg_lout) memcpy(m->dbg_lout + (size_t)slot * D, cur, D * 4);
}

static void bs_copy(blockset *dst, const blockset *src, int D) {
    dst->nb = src->nb;
    for (int i = 0; i < src->nb; i++) { dst->blocks[i] = malloc(D * 4); memcpy(dst->blocks[i], src->blocks[i], D * 4); }
}
static void bs_free(blockset *b) { for (int i = 0; i < b->nb; i++) free(b->blocks[i]); b->nb = 0; }

float *ink_forward(ink_model *m, int token, int pos) {
    const ink_cfg *c = &m->c;
    const int D = c->n_embd, start = c->loop_start, end = c->loop_end, nl = end - start;
    float *cur = xalloc(D * 4), *prelude = xalloc(D * 4), *bank_out = xalloc(D * 4), *msgs = xalloc(4 * D * 4);
    {   // embedding
        uint8_t *row = xalloc(m->tok_embd.row_bytes);
        read_at(&m->g, row, m->tok_embd.row_bytes, m->tok_embd.file_off + (uint64_t)token * m->tok_embd.row_bytes);
        dequant_row(m->tok_embd.type, row, cur, D);
        free(row);
        rmsnorm(cur, cur, m->tok_norm, D, c->eps);
        dbg("inp_norm", 0, cur, D);
    }
    float tau = (pos + 1.0f) / c->log_floor;
    tau = 1.0f + c->log_alpha * logf(tau > 1 ? tau : 1);

    blockset B = {{0}, 0}, pre = {{0}, 0}, bank = {{0}, 0};
    m->step = 0;
    for (int il = 0; il < start; il++) layer(m, il, il, cur, &B, NULL, NULL, pos, tau);
    memcpy(prelude, cur, D * 4);
    bs_copy(&pre, &B, D);
    for (int pass = 0; pass < c->n_passes; pass++) {
        if (pass > 0) {
            // the previous pass: its blocks after the first core block, and its output
            bs_free(&bank);
            for (int i = pre.nb + 1; i < B.nb; i++) bank.blocks[bank.nb++] = B.blocks[i];
            for (int i = pre.nb; i <= pre.nb && i < B.nb; i++) free(B.blocks[i]);
            for (int i = 0; i < pre.nb; i++) free(B.blocks[i]);
            B.nb = 0;
            memcpy(bank_out, cur, D * 4);
            memcpy(cur, prelude, D * 4);
            bs_copy(&B, &pre, D);
        }
        for (int ip = start; ip < end; ip++) {
            if (pass == 0) { layer(m, ip, ip, cur, &B, NULL, NULL, pos, tau); continue; }
            const ink_loop *P = &m->loops[ip - start];
            mix(D, bank.blocks, bank.nb, bank_out, P->msg_query, 4, msgs);
            const int slot = c->n_phys + (pass - 1) * nl + (ip - start);
            layer(m, slot, ip, cur, &B, msgs, P, pos, tau);
        }
    }
    for (int il = end; il < c->n_phys; il++) layer(m, il, il, cur, &B, NULL, NULL, pos, tau);

    // output
    float *h = xalloc(D * 4);
    const double t_o = inow();
    mix(D, B.blocks, B.nb, cur, m->output_res_query, 1, h);
    rmsnorm(h, h, m->output_norm, D, c->eps);
    ink_scratch *S = scratch(c);
    qin q;
    qin_init(&q, h, D, S->kq[0], S->aq[0], S->ad[0], S->as[0]);
    mv_q(m->output.type, m->output.data, m->output.row_bytes, c->n_vocab, D, &q, m->logits);
    if (c->softcap > 0)
        for (int i = 0; i < c->n_vocab; i++) m->logits[i] = c->softcap * tanhf(m->logits[i] / c->softcap);
    ink_prof[2] += inow() - t_o;
    bs_free(&B); bs_free(&pre); bs_free(&bank);
    free(cur); free(prelude); free(bank_out); free(msgs); free(h);
    return m->logits;
}
