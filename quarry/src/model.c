#define _GNU_SOURCE
#include "model.h"

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>

#include "quants.h"
#include <immintrin.h>
#include <time.h>
double qt_dbg[32];
double qt_prof[8];   // 0 attn matvec, 1 attn core, 2 router+pred, 3 expert compute, 4 output, 5 norms/rope/misc
static double qt_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
#define PT(i, stmt) do { double _t = qt_now(); stmt; qt_prof[i] += qt_now() - _t; } while (0)

static void *xalloc(size_t n) {
    // big buffers on 2MB huge pages: weights are streamed, 4KB pages cost a TLB walk each
    const size_t al = n >= (1u << 21) ? (1u << 21) : 64, sz = (n + al - 1) / al * al;
    void *p = aligned_alloc(al, sz);
    if (!p) { fprintf(stderr, "out of memory (%zu bytes)\n", n); exit(1); }
    if (al > 64) madvise(p, sz, MADV_HUGEPAGE);
    return p;
}

static const qg_tensor *need(qg_file *g, const char *name) {
    const qg_tensor *t = qg_find_tensor(g, name);
    if (!t) { fprintf(stderr, "missing tensor %s\n", name); exit(1); }
    return t;
}

static void read_at(qg_file *g, void *dst, uint64_t n, uint64_t off) {
    uint8_t *p = dst;
    while (n) {
        ssize_t r = pread(g->fd, p, n > (1u << 30) ? (1u << 30) : n, off);
        if (r <= 0) { perror("pread"); exit(1); }
        p += r; off += r; n -= r;
    }
}

static float *load_f32(qg_file *g, const char *name) {
    const qg_tensor *t = need(g, name);
    if (t->type != GT_F32) { fprintf(stderr, "%s: expected F32\n", name); exit(1); }
    float *p = xalloc(t->nbytes);
    read_at(g, p, t->nbytes, t->offset);
    return p;
}

static qmat load_mat(qg_file *g, const char *name, int in_ram) {
    const qg_tensor *t = need(g, name);
    qmat q = {t->type, (int)(t->ne[1] * t->ne[2]), (int)t->ne[0], qg_row_bytes(t->type, t->ne[0]), NULL, t->offset};
    if (in_ram) {
        q.data = xalloc(t->nbytes);
        read_at(g, q.data, t->nbytes, t->offset);
    }
    return q;
}

int model_load(qmodel *m, const char *path, int n_ctx) {
    memset(m, 0, sizeof *m);
    if (qg_open(&m->g, path)) return -1;
    qg_file *g = &m->g;
    const char *arch = qg_get_str(g, "general.architecture");
    if (!arch || strcmp(arch, "qwen3moe")) { fprintf(stderr, "unsupported arch %s\n", arch); return -1; }
    qcfg *c = &m->c;
    c->n_layer = qg_get_int(g, "qwen3moe.block_count", 0);
    c->n_embd = qg_get_int(g, "qwen3moe.embedding_length", 0);
    c->n_head = qg_get_int(g, "qwen3moe.attention.head_count", 0);
    c->n_head_kv = qg_get_int(g, "qwen3moe.attention.head_count_kv", 0);
    c->head_dim = qg_get_int(g, "qwen3moe.attention.key_length", 128);
    c->n_expert = qg_get_int(g, "qwen3moe.expert_count", 0);
    c->n_expert_used = qg_get_int(g, "qwen3moe.expert_used_count", 0);
    // QUARRY_TOPK: use fewer experts per token than trained (lossy, faster: fewer bytes and SSD reads)
    if (getenv("QUARRY_TOPK") && atoi(getenv("QUARRY_TOPK")) > 0 && atoi(getenv("QUARRY_TOPK")) < c->n_expert_used) c->n_expert_used = atoi(getenv("QUARRY_TOPK"));
    c->n_ff_exp = qg_get_int(g, "qwen3moe.expert_feed_forward_length", 0);
    c->rope_base = qg_get_float(g, "qwen3moe.rope.freq_base", 1e6);
    c->rms_eps = qg_get_float(g, "qwen3moe.attention.layer_norm_rms_epsilon", 1e-6);
    c->n_ctx = n_ctx;

    m->tok_embd = load_mat(g, "token_embd.weight", 0);  // rows read on demand
    c->n_vocab = m->tok_embd.rows;
    m->output = load_mat(g, "output.weight", 1);
    m->output_norm = load_f32(g, "output_norm.weight");
    m->layers = calloc(c->n_layer, sizeof(qlayer));
    char n[128];
    for (int l = 0; l < c->n_layer; l++) {
        qlayer *L = &m->layers[l];
#define NM(s) (snprintf(n, sizeof n, "blk.%d.%s", l, s), n)
        L->attn_norm = load_f32(g, NM("attn_norm.weight"));
        L->ffn_norm = load_f32(g, NM("ffn_norm.weight"));
        L->q_norm = load_f32(g, NM("attn_q_norm.weight"));
        L->k_norm = load_f32(g, NM("attn_k_norm.weight"));
        {   // keep the router as bf16: the stored F32 values are exact bf16 (checked), so this
            // halves its per-token reads with bit-identical results
            float *rf = load_f32(g, NM("ffn_gate_inp.weight"));
            const size_t nr = (size_t)c->n_expert * c->n_embd;
            L->router = xalloc(nr * 2);
            for (size_t i = 0; i < nr; i++) {
                uint32_t u; memcpy(&u, rf + i, 4);
                if (u & 0xffff) { fprintf(stderr, "router weight is not bf16\n"); exit(1); }
                L->router[i] = (uint16_t)(u >> 16);
            }
            free(rf);
        }
        L->wq = load_mat(g, NM("attn_q.weight"), 1);
        L->wk = load_mat(g, NM("attn_k.weight"), 1);
        L->wv = load_mat(g, NM("attn_v.weight"), 1);
        L->wo = load_mat(g, NM("attn_output.weight"), 1);
        L->gate_exps = load_mat(g, NM("ffn_gate_exps.weight"), 0);
        L->up_exps = load_mat(g, NM("ffn_up_exps.weight"), 0);
        L->down_exps = load_mat(g, NM("ffn_down_exps.weight"), 0);
#undef NM
    }
    const int kv_dim = c->n_head_kv * c->head_dim, q_dim = c->n_head * c->head_dim;
    size_t kv = (size_t)c->n_layer * n_ctx * kv_dim * sizeof(uint16_t);   // fp16 KV, like llama.cpp's default
    m->k_cache = xalloc(kv);
    m->v_cache = xalloc(kv);
    m->x = xalloc(c->n_embd * 4);   m->xb = xalloc(q_dim * 4);  m->xb2 = xalloc(c->n_embd * 4);
    m->q = xalloc(q_dim * 4);       m->k = xalloc(kv_dim * 4);  m->v = xalloc(kv_dim * 4);
    m->att = xalloc((size_t)c->n_head * n_ctx * 4);
    m->hb = xalloc(c->n_ff_exp * 4); m->hb2 = xalloc(c->n_ff_exp * 4);
    m->logits = xalloc((size_t)c->n_vocab * 4);
    m->router_logits = xalloc(c->n_expert * 4);
    m->xq = xalloc(sizeof(block_q8_K) * (q_dim / QK_K + 1));
    m->experts = experts_open(m);
    m->eh = xalloc((size_t)16 * c->n_ff_exp * 4);
    m->eg = xalloc((size_t)16 * c->n_ff_exp * 4);
    m->eo = xalloc((size_t)16 * c->n_embd * 4);
    m->ehq = xalloc(sizeof(block_q8_K) * 16 * (c->n_ff_exp / QK_K));
    m->pred = calloc((size_t)c->n_layer * c->n_expert_used, sizeof(int));
    m->prefetch_on = getenv("QUARRY_NO_PREFETCH") ? 0 : 1;
    return 0;
}

// ---------------- ops ----------------

static void rmsnorm(float *o, const float *x, const float *w, int n, float eps) {
    double ss = 0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    float s = 1.0f / sqrtf((float)(ss / n) + eps);
    for (int i = 0; i < n; i++) o[i] = x[i] * s * w[i];
}

// y[rows] = W (rows x cols, quantized) * x ; x given already quantized (q8_K)
static void matvec_q(const uint8_t *w, int type, uint64_t row_bytes, int rows, int cols,
                     const block_q8_K *xq, float *y) {
    #pragma omp parallel for schedule(static) if (rows >= 256)
    for (int r = 0; r < rows; r++) {
        const uint8_t *row = w + (uint64_t)r * row_bytes;
        y[r] = type == GT_Q3_K ? vec_dot_q3_K_q8_K(cols, (const block_q3_K *)row, xq)
                               : vec_dot_q6_K_q8_K(cols, (const block_q6_K *)row, xq);
    }
}

static void matvec(const qmat *W, const float *x, float *y, block_q8_K *xq) {
    quantize_row_q8_K(x, xq, W->cols);
    matvec_q(W->data, W->type, W->row_bytes, W->rows, W->cols, xq, y);
}

// NEOX-style RoPE (rotate halves), as used by qwen3 in llama.cpp
// cos/sin table for one position (same values as rope_neox computes), shared by all heads/layers
static void rope_table(float *cs, int head_dim, int pos, float base) {
    const int half = head_dim / 2;
    for (int i = 0; i < half; i++) {
        float theta = pos * powf(base, -2.0f * i / head_dim);
        cs[i] = cosf(theta);
        cs[half + i] = sinf(theta);
    }
}
static void rope_apply(float *v, int head_dim, const float *cs) {
    const int half = head_dim / 2;
    for (int i = 0; i < half; i++) {
        float c = cs[i], s = cs[half + i];
        float a = v[i], b = v[i + half];
        v[i] = a * c - b * s;
        v[i + half] = a * s + b * c;
    }
}

static float silu(float x) { return x / (1.0f + expf(-x)); }

// router logits -> top-k experts (softmax weights, renormalized)
static inline float dot_f32(const float *a, const float *b, int n) {
    __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), acc0);
        acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 16), _mm512_loadu_ps(b + i + 16), acc1);
    }
    float s = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}

// bf16 row . f32 vector, same accumulation order as dot_f32 (bit-identical to the f32 rows)
static inline __m512 bf16x16(const uint16_t *p) {
    return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *)p)), 16));
}
static inline float dot_bf16_f32(const uint16_t *a, const float *b, int n) {
    __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        acc0 = _mm512_fmadd_ps(bf16x16(a + i), _mm512_loadu_ps(b + i), acc0);
        acc1 = _mm512_fmadd_ps(bf16x16(a + i + 16), _mm512_loadu_ps(b + i + 16), acc1);
    }
    float s = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
    for (; i < n; i++) { uint32_t u = (uint32_t)a[i] << 16; float f; memcpy(&f, &u, 4); s += f * b[i]; }
    return s;
}

// fp16 KV cache helpers (n multiple of 16)
static inline void f32_to_f16(uint16_t *dst, const float *src, int n) {
    for (int i = 0; i < n; i += 16)
        _mm256_storeu_si256((__m256i *)(dst + i), _mm512_cvtps_ph(_mm512_loadu_ps(src + i), _MM_FROUND_TO_NEAREST_INT));
}
static inline __m512 f16x16(const uint16_t *p) { return _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)p)); }
static inline float dot_f32_f16(const float *a, const uint16_t *b, int n) {
    __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
    for (int i = 0; i < n; i += 32) {
        acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), f16x16(b + i), acc0);
        acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 16), f16x16(b + i + 16), acc1);
    }
    return _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
}

static void route(const uint16_t *router, const float *xn, int n_expert, int D, int k, float *rl,
                  int *sel, float *wsel) {
    #pragma omp parallel for schedule(static)
    for (int e = 0; e < n_expert; e++) rl[e] = dot_bf16_f32(router + (size_t)e * D, xn, D);
    float mx = rl[0];
    for (int e = 1; e < n_expert; e++) if (rl[e] > mx) mx = rl[e];
    float tot = 0;
    for (int e = 0; e < n_expert; e++) { rl[e] = expf(rl[e] - mx); tot += rl[e]; }
    float wsum = 0;
    for (int j = 0; j < k; j++) {
        int best = -1;
        for (int e = 0; e < n_expert; e++) {
            int taken = 0;
            for (int q = 0; q < j; q++) taken |= sel[q] == e;
            if (!taken && (best < 0 || rl[e] > rl[best])) best = e;
        }
        sel[j] = best;
        wsel[j] = rl[best] / tot;
        wsum += wsel[j];
    }
    for (int j = 0; j < k; j++) wsel[j] /= wsum;
}

static void embed(qmodel *m, int token, float *out) {
    const qmat *E = &m->tok_embd;
    uint8_t *row = xalloc(E->row_bytes);
    read_at(&m->g, row, E->row_bytes, E->file_off + (uint64_t)token * E->row_bytes);
    if (E->type == GT_Q3_K) dequant_row_q3_K((block_q3_K *)row, out, E->cols);
    else dequant_row_q6_K((block_q6_K *)row, out, E->cols);
    free(row);
}

static void softmax512(float *row, int n);
static inline __m512 exp512(__m512 x);

float *model_forward(qmodel *m, int token, int pos) {
    const qcfg *c = &m->c;
    const int D = c->n_embd, HD = c->head_dim, NH = c->n_head, NKV = c->n_head_kv;
    const int kv_dim = NKV * HD, q_dim = NH * HD, gqa = NH / NKV;
    block_q8_K *xq = m->xq;
    embed(m, token, m->x);
    float rcs[512];
    rope_table(rcs, HD, pos, c->rope_base);

    for (int l = 0; l < c->n_layer; l++) {
        qlayer *L = &m->layers[l];
        // ---- attention ----
        rmsnorm(m->xb, m->x, L->attn_norm, D, c->rms_eps);
        quantize_row_q8_K(m->xb, xq, D);
        {
            double _t = qt_now();
            const qmat *W3[3] = {&L->wq, &L->wk, &L->wv};
            float *Y3[3] = {m->q, m->k, m->v};
            const int nq = L->wq.rows, nk = L->wk.rows, total = nq + nk + L->wv.rows;
            #pragma omp parallel for schedule(static)
            for (int r = 0; r < total; r++) {
                int w = r < nq ? 0 : r < nq + nk ? 1 : 2;
                int rr = r - (w == 0 ? 0 : w == 1 ? nq : nq + nk);
                const qmat *W = W3[w];
                const uint8_t *row = W->data + (uint64_t)rr * W->row_bytes;
                Y3[w][rr] = W->type == GT_Q3_K ? vec_dot_q3_K_q8_K(D, (const block_q3_K *)row, xq)
                                               : vec_dot_q6_K_q8_K(D, (const block_q6_K *)row, xq);
            }
            qt_prof[0] += qt_now() - _t;
        }
        for (int h = 0; h < NH; h++) {
            rmsnorm(m->q + h * HD, m->q + h * HD, L->q_norm, HD, c->rms_eps);
            rope_apply(m->q + h * HD, HD, rcs);
        }
        for (int h = 0; h < NKV; h++) {
            rmsnorm(m->k + h * HD, m->k + h * HD, L->k_norm, HD, c->rms_eps);
            rope_apply(m->k + h * HD, HD, rcs);
        }
        uint16_t *kc = m->k_cache + ((size_t)l * c->n_ctx) * kv_dim;
        uint16_t *vc = m->v_cache + ((size_t)l * c->n_ctx) * kv_dim;
        f32_to_f16(kc + (size_t)pos * kv_dim, m->k, kv_dim);
        f32_to_f16(vc + (size_t)pos * kv_dim, m->v, kv_dim);
        const float scale = 1.0f / sqrtf((float)HD);
        double _ta = qt_now();
        #pragma omp parallel for schedule(static)
        for (int h = 0; h < NH; h++) {
            const float *qh = m->q + h * HD;
            float *att = m->att + (size_t)h * c->n_ctx;
            const int kvh = h / gqa;
            for (int t = 0; t <= pos; t++) att[t] = dot_f32_f16(qh, kc + (size_t)t * kv_dim + kvh * HD, HD) * scale;
            softmax512(att, pos + 1);
            __m512 o[8];
            for (int i = 0; i < 8; i++) o[i] = _mm512_setzero_ps();
            for (int t = 0; t <= pos; t++) {
                const __m512 a = _mm512_set1_ps(att[t]);
                const uint16_t *vt = vc + (size_t)t * kv_dim + kvh * HD;
                for (int i = 0; i < 8; i++) o[i] = _mm512_fmadd_ps(a, f16x16(vt + 16 * i), o[i]);
            }
            float *out = m->xb + h * HD;
            for (int i = 0; i < 8; i++) _mm512_storeu_ps(out + 16 * i, o[i]);
        }
        qt_prof[1] += qt_now() - _ta;
        quantize_row_q8_K(m->xb, xq, q_dim);
        PT(0, matvec_q(L->wo.data, L->wo.type, L->wo.row_bytes, L->wo.rows, q_dim, xq, m->xb2));
        for (int i = 0; i < D; i++) m->x[i] += m->xb2[i];

        // ---- MoE FFN ----
        rmsnorm(m->xb, m->x, L->ffn_norm, D, c->rms_eps);
        const int K = c->n_expert_used;
        int sel[16];
        float wsel[16];
        double _tr = qt_now();
        route(L->router, m->xb, c->n_expert, D, K, m->router_logits, sel, wsel);
        // prediction accuracy of the previous layer's guess for this layer
        if (m->prefetch_on && l > 0) {
            for (int j = 0; j < K; j++)
                for (int q = 0; q < K; q++) m->pred_correct += m->pred[l * K + q] == sel[j];
            m->pred_total += K;
        }
        // optional (lossy) expert skipping: drop low-weight experts, either all of them
        // (QUARRY_SKIP) or only those not already in RAM (QUARRY_SKIP_MISS), whose
        // SSD read would stall the layer
        static float skip_all = -2, skip_miss = -1;
        if (skip_all == -2) {
            skip_all = getenv("QUARRY_SKIP") ? (float)atof(getenv("QUARRY_SKIP")) : -1;
            skip_miss = getenv("QUARRY_SKIP_MISS") ? (float)atof(getenv("QUARRY_SKIP_MISS")) : -1;
        }
        int Ke = K;
        if (skip_all > 0 || skip_miss > 0) {
            Ke = 0;
            for (int j = 0; j < K; j++) {
                const int drop = wsel[j] < skip_all || (wsel[j] < skip_miss && !experts_ready(m->experts, l, sel[j]));
                if (drop) { qt_dbg[20] += 1; continue; }
                sel[Ke] = sel[j]; wsel[Ke] = wsel[j]; Ke++;
            }
        }
        experts_begin_layer(m->experts, l, sel, Ke);
        // predict next layer's experts from the current hidden state and prefetch
        if (m->prefetch_on && l + 1 < c->n_layer) {
            const qlayer *N = &m->layers[l + 1];
            float *xn = m->att;  // free scratch here (n_head * n_ctx >= n_embd)
            rmsnorm(xn, m->x, N->ffn_norm, D, c->rms_eps);
            float pw[16];
            route(N->router, xn, c->n_expert, D, K, m->router_logits, m->pred + (l + 1) * K, pw);
            experts_prefetch(m->experts, l + 1, m->pred + (l + 1) * K, K);
        }
        qt_prof[2] += qt_now() - _tr;
        double _te = qt_now();
        quantize_row_q8_K(m->xb, xq, D);
        memset(m->xb2, 0, D * 4);
        int done[16] = {0};
        for (int n_done = 0; n_done < Ke;) {      // batch every expert in RAM; then wait for any of the rest
            int bl[16], nb = 0;
            for (int k = 0; k < Ke; k++)
                if (!done[k] && experts_ready(m->experts, l, sel[k])) bl[nb++] = k;
            if (!nb) {
                int rest[16], nr = 0;
                for (int k = 0; k < Ke; k++) if (!done[k]) rest[nr++] = sel[k];
                experts_wait_any(m->experts, l, rest, nr);
                continue;
            }
            n_done += nb;
            const uint8_t *gw[16], *uw[16], *dw[16];
            for (int j = 0; j < nb; j++) experts_get(m->experts, l, sel[bl[j]], &gw[j], &uw[j], &dw[j]);
            const int F = c->n_ff_exp;
            const uint64_t grb = L->gate_exps.row_bytes, urb = L->up_exps.row_bytes, drb = L->down_exps.row_bytes;
            float *H = m->eh, *G = m->eg;            // [nb][F]
            double _tc = qt_now();
            qt_dbg[8 + (nb > 8 ? 8 : nb)] += 1;
            #pragma omp parallel for schedule(static)
            for (int r = 0; r < nb * 2 * F; r++) {
                int j = r / (2 * F), rr = r % (2 * F);
                if (rr < F) G[j * F + rr] = vec_dot_q3_K_q8_K(D, (const block_q3_K *)(gw[j] + rr * grb), xq);
                else H[j * F + rr - F] = vec_dot_q3_K_q8_K(D, (const block_q3_K *)(uw[j] + (rr - F) * urb), xq);
            }
            double _tg = qt_now(); qt_dbg[0] += _tg - _tc;
            block_q8_K *hq = m->ehq;                 // [nb][F/QK_K]
            const int nbq = F / QK_K;
            for (int j = 0; j < nb; j++) {
                float *g = G + j * F, *hh = H + j * F;
                for (int i = 0; i < F; i += 16) {   // silu(g) * h = g * h / (1 + exp(-g))
                    const __m512 gv = _mm512_loadu_ps(g + i);
                    const __m512 e = exp512(_mm512_sub_ps(_mm512_setzero_ps(), gv));
                    _mm512_storeu_ps(g + i, _mm512_div_ps(_mm512_mul_ps(gv, _mm512_loadu_ps(hh + i)),
                                                          _mm512_add_ps(_mm512_set1_ps(1.0f), e)));
                }
                quantize_row_q8_K(g, hq + j * nbq, F);
            }
            double _ts = qt_now(); qt_dbg[1] += _ts - _tg;
            float *O = m->eo;                        // [nb][D]
            #pragma omp parallel for schedule(static)
            for (int r = 0; r < nb * D; r++) {
                int j = r / D, rr = r % D;
                O[r] = vec_dot_q3_K_q8_K(F, (const block_q3_K *)(dw[j] + rr * drb), hq + j * nbq);
            }
            qt_dbg[2] += qt_now() - _ts;
            if (nb == 8) { qt_dbg[21] += qt_now() - _tc; qt_dbg[22] += 8; } else if (nb == 1) { qt_dbg[23] += qt_now() - _tc; qt_dbg[24] += 1; }
            qt_prof[6] += qt_now() - _tc;
            qt_prof[7] += nb;
            for (int j = 0; j < nb; j++) {
                experts_release(m->experts, l, sel[bl[j]]);
                const float w = wsel[bl[j]];
                for (int i = 0; i < D; i++) m->xb2[i] += w * O[j * D + i];
                done[bl[j]] = 1;
            }
        }
        qt_prof[3] += qt_now() - _te;
        for (int i = 0; i < D; i++) m->x[i] += m->xb2[i];
    }
    rmsnorm(m->x, m->x, m->output_norm, D, c->rms_eps);
    PT(4, matvec(&m->output, m->x, m->logits, xq));
    return m->logits;
}

// ---------------- batched prefill ----------------
// Each weight row is repacked once (q3k_repack_row) and reused for every token
// in the batch; tokens are grouped by expert so each expert is read once per
// batch. Experts not in the cache are streamed into staging slots instead of
// evicting the cache, so generation stays warm after a long prompt.

#define PF_T 512
#define PF_TT 8        // tokens per attention work item
#define NH_MAX 8       // query heads per kv head handled per item (gqa)
double qt_pf[8];

typedef struct {
    float *X, *XN, *Q, *K, *V, *A, *XO, *cosb, *sinb, *rl, *G, *H, *O, *wsel, *tokw, *DX, *dxe, *dh, *sc;
    float *KT;                 // [kv head][head_dim][n_ctx]: keys transposed for the score tiles
    uint8_t *XU, *Xe, *HU;
    int *sel, *cnt, *off, *tok;
    q3k_s8 *pk;
    int nthr, maxb;
} prefill_buf;

static prefill_buf *pf_get(qmodel *m) {
    if (m->pf) return m->pf;
    const qcfg *c = &m->c;
    const int T = PF_T, D = c->n_embd, qd = c->n_head * c->head_dim, kvd = c->n_head_kv * c->head_dim;
    const int F = c->n_ff_exp, K = c->n_expert_used, maxc = qd > D ? qd : D;
    prefill_buf *b = calloc(1, sizeof *b);
    b->X = xalloc((size_t)T * D * 4);   b->XN = xalloc((size_t)T * D * 4);
    b->Q = xalloc((size_t)T * qd * 4);  b->K = xalloc((size_t)T * kvd * 4); b->V = xalloc((size_t)T * kvd * 4);
    b->A = xalloc((size_t)T * qd * 4);  b->XO = xalloc((size_t)T * D * 4);
    b->cosb = xalloc((size_t)T * c->head_dim / 2 * 4); b->sinb = xalloc((size_t)T * c->head_dim / 2 * 4);
    b->rl = xalloc((size_t)T * c->n_expert * 4);
    b->G = xalloc((size_t)T * F * 4);   b->H = xalloc((size_t)T * F * 4); b->O = xalloc((size_t)T * D * 4);
    b->wsel = xalloc((size_t)T * K * 4); b->tokw = xalloc((size_t)T * K * 4);
    b->sel = xalloc((size_t)T * K * 4); b->tok = xalloc((size_t)T * K * 4);
    b->cnt = xalloc(c->n_expert * 4);   b->off = xalloc((c->n_expert + 1) * 4);
    b->XU = xalloc((size_t)T * maxc);   b->DX = xalloc((size_t)T * (maxc / QK_K) * 4);
    b->Xe = xalloc((size_t)T * D);      b->dxe = xalloc((size_t)T * (D / QK_K) * 4);
    b->HU = xalloc((size_t)T * F);      b->dh = xalloc((size_t)T * (F / QK_K) * 4);
    b->nthr = omp_get_max_threads();
    b->maxb = maxc / QK_K;
    b->pk = xalloc(sizeof(q3k_s8) * b->maxb * b->nthr * 16);
    b->sc = xalloc((size_t)b->nthr * PF_TT * (NH_MAX) * (c->n_ctx + 64) * 4);
    b->KT = xalloc((size_t)c->n_head_kv * c->head_dim * (c->n_ctx + 64) * 4);
    m->pf = b;
    return b;
}

// out[t][rows] = W * x_t for T tokens; xu/dx hold the quantized tokens with stride W->cols.
// Rows are repacked in blocks of MM_RB, then each group of 4 tokens is swept across the
// block, so a token's activations are reused from L1 for MM_RB rows.
#define MM_RB 16
static void mm_q3k(prefill_buf *b, const uint8_t *w, uint64_t row_bytes, int rows, int cols, int T,
                   const uint8_t *xu, const float *dx, float *out, int out_stride) {
    const int nbk = cols / QK_K, nblk = (rows + MM_RB - 1) / MM_RB;
    #pragma omp parallel for schedule(static)
    for (int rb = 0; rb < nblk; rb++) {
        q3k_s8 *pk = b->pk + (size_t)omp_get_thread_num() * b->maxb * MM_RB;
        const int r0 = rb * MM_RB, nr = rows - r0 < MM_RB ? rows - r0 : MM_RB;
        for (int r = 0; r < nr; r++)
            q3k_repack_row((const block_q3_K *)(w + (uint64_t)(r0 + r) * row_bytes), cols, pk + r * nbk);
        for (int t = 0; t < T; t += 4) {
            const int nt = T - t < 4 ? T - t : 4;
            for (int r = 0; r < nr; r++)
                dot_q3k_s8_multi(cols, pk + r * nbk, nt, xu + (size_t)t * cols, dx + (size_t)t * nbk,
                                 out + (size_t)t * out_stride + r0 + r, out_stride);
        }
    }
}

static void quantize_rows_u8(const float *x, int T, int n, uint8_t *xu, float *dx) {
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < T; t++) quantize_row_u8(x + (size_t)t * n, xu + (size_t)t * n, dx + (size_t)t * (n / QK_K), n);
}

// one expert over its tokens: Y[tok] += w * down(silu(gate x) * up x)
static void pf_expert(qmodel *m, prefill_buf *b, const qlayer *L, int e, const uint8_t *gw,
                      const uint8_t *uw, const uint8_t *dw) {
    const qcfg *c = &m->c;
    const int D = c->n_embd, F = c->n_ff_exp, nbD = D / QK_K, nbF = F / QK_K;
    const int n0 = b->off[e], ne = b->off[e + 1] - n0;
    for (int j = 0; j < ne; j++) {
        const int t = b->tok[n0 + j];
        memcpy(b->Xe + (size_t)j * D, b->XU + (size_t)t * D, D);
        memcpy(b->dxe + (size_t)j * nbD, b->DX + (size_t)t * nbD, nbD * 4);
    }
    const uint64_t grb = L->gate_exps.row_bytes, urb = L->up_exps.row_bytes, drb = L->down_exps.row_bytes;
    double _t1 = qt_now();
    mm_q3k(b, gw, grb, F, D, ne, b->Xe, b->dxe, b->G, F);
    mm_q3k(b, uw, urb, F, D, ne, b->Xe, b->dxe, b->H, F);
    qt_pf[5] += qt_now() - _t1;
    for (int j = 0; j < ne; j++) {
        float *g = b->G + (size_t)j * F, *h = b->H + (size_t)j * F;
        for (int i = 0; i < F; i++) g[i] = silu(g[i]) * h[i];
        quantize_row_u8(g, b->HU + (size_t)j * F, b->dh + (size_t)j * nbF, F);
    }
    _t1 = qt_now();
    mm_q3k(b, dw, drb, D, F, ne, b->HU, b->dh, b->O, D);
    qt_pf[6] += qt_now() - _t1;
    for (int j = 0; j < ne; j++) {
        const float w = b->tokw[n0 + j];
        float *y = b->XO + (size_t)b->tok[n0 + j] * D, *o = b->O + (size_t)j * D;
        for (int i = 0; i < D; i++) y[i] += w * o[i];
    }
}

// exp(x) for x <= 0 (softmax inputs), ~1e-7 relative error: 2^n * poly(r), x = n*ln2 + r
static inline __m512 exp512(__m512 x) {
    x = _mm512_max_ps(x, _mm512_set1_ps(-87.0f));
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

// in place: row[0..n) = softmax(row[0..n))
static void softmax512(float *row, int n) {
    __m512 mx = _mm512_set1_ps(-1e30f);
    int i = 0;
    for (; i + 16 <= n; i += 16) mx = _mm512_max_ps(mx, _mm512_loadu_ps(row + i));
    if (i < n) mx = _mm512_max_ps(mx, _mm512_mask_loadu_ps(_mm512_set1_ps(-1e30f), (__mmask16)((1u << (n - i)) - 1), row + i));
    const __m512 m = _mm512_set1_ps(_mm512_reduce_max_ps(mx));
    __m512 sum = _mm512_setzero_ps();
    for (i = 0; i + 16 <= n; i += 16) {
        __m512 e = exp512(_mm512_sub_ps(_mm512_loadu_ps(row + i), m));
        _mm512_storeu_ps(row + i, e);
        sum = _mm512_add_ps(sum, e);
    }
    if (i < n) {
        const __mmask16 k = (__mmask16)((1u << (n - i)) - 1);
        __m512 e = exp512(_mm512_sub_ps(_mm512_maskz_loadu_ps(k, row + i), m));
        _mm512_mask_storeu_ps(row + i, k, e);
        sum = _mm512_add_ps(sum, _mm512_maskz_mov_ps(k, e));
    }
    const __m512 r = _mm512_set1_ps(1.0f / _mm512_reduce_add_ps(sum));
    for (i = 0; i + 16 <= n; i += 16) _mm512_storeu_ps(row + i, _mm512_mul_ps(_mm512_loadu_ps(row + i), r));
    if (i < n) {
        const __mmask16 k = (__mmask16)((1u << (n - i)) - 1);
        _mm512_mask_storeu_ps(row + i, k, _mm512_mul_ps(_mm512_maskz_loadu_ps(k, row + i), r));
    }
}

static void pf_batch(qmodel *m, const int *tokens, int T, int pos0) {
    const qcfg *c = &m->c;
    prefill_buf *b = pf_get(m);
    const int D = c->n_embd, HD = c->head_dim, NH = c->n_head, NKV = c->n_head_kv, K = c->n_expert_used;
    const int kvd = NKV * HD, qd = NH * HD, gqa = NH / NKV, half = HD / 2, NE = c->n_expert;
    double t0;
    for (int t = 0; t < T; t++) embed(m, tokens[t], b->X + (size_t)t * D);
    // rope tables for this batch (shared by every head and layer)
    for (int t = 0; t < T; t++)
        for (int i = 0; i < half; i++) {
            float th = (pos0 + t) * powf(c->rope_base, -2.0f * i / HD);
            b->cosb[t * half + i] = cosf(th);
            b->sinb[t * half + i] = sinf(th);
        }
    for (int l = 0; l < c->n_layer; l++) {
        const qlayer *L = &m->layers[l];
        // ---- attention ----
        t0 = qt_now();
        #pragma omp parallel for schedule(static)
        for (int t = 0; t < T; t++) rmsnorm(b->XN + (size_t)t * D, b->X + (size_t)t * D, L->attn_norm, D, c->rms_eps);
        quantize_rows_u8(b->XN, T, D, b->XU, b->DX);
        mm_q3k(b, L->wq.data, L->wq.row_bytes, L->wq.rows, D, T, b->XU, b->DX, b->Q, qd);
        mm_q3k(b, L->wk.data, L->wk.row_bytes, L->wk.rows, D, T, b->XU, b->DX, b->K, kvd);
        mm_q3k(b, L->wv.data, L->wv.row_bytes, L->wv.rows, D, T, b->XU, b->DX, b->V, kvd);
        qt_pf[0] += qt_now() - t0;
        t0 = qt_now();
        uint16_t *kc = m->k_cache + ((size_t)l * c->n_ctx) * kvd;
        uint16_t *vc = m->v_cache + ((size_t)l * c->n_ctx) * kvd;
        #pragma omp parallel for schedule(static)
        for (int t = 0; t < T; t++) {
            const float *cs = b->cosb + t * half, *sn = b->sinb + t * half;
            for (int h = 0; h < NH + NKV; h++) {
                float *v = h < NH ? b->Q + (size_t)t * qd + h * HD : b->K + (size_t)t * kvd + (h - NH) * HD;
                rmsnorm(v, v, h < NH ? L->q_norm : L->k_norm, HD, c->rms_eps);
                for (int i = 0; i < half; i++) {
                    float a = v[i], bb = v[i + half];
                    v[i] = a * cs[i] - bb * sn[i];
                    v[i + half] = a * sn[i] + bb * cs[i];
                }
            }
            f32_to_f16(kc + (size_t)(pos0 + t) * kvd, b->K + (size_t)t * kvd, kvd);
            f32_to_f16(vc + (size_t)(pos0 + t) * kvd, b->V + (size_t)t * kvd, kvd);
        }
        // keys transposed: KT[kvh][d][s], so 16 keys' scores come out of one FMA chain
        const int S = pos0 + T, ldk = c->n_ctx + 64;
        #pragma omp parallel for schedule(static)
        for (int hd = 0; hd < NKV * HD; hd++) {
            const int kvh = hd / HD, d = hd % HD;
            float *kt = b->KT + (size_t)hd * ldk;
            const uint16_t *src = kc + kvh * HD + d;
            for (int s = 0; s < S; s++) kt[s] = _cvtsh_ss(src[(size_t)s * kvd]);
        }
        const float scale = 1.0f / sqrtf((float)HD);
        const int ntile = (T + PF_TT - 1) / PF_TT;
        #pragma omp parallel for schedule(dynamic, 1)
        for (int item = 0; item < NKV * ntile; item++) {
            const int kvh = item % NKV, tt0 = (ntile - 1 - item / NKV) * PF_TT;   // longest rows first
            const int tt1 = tt0 + PF_TT < T ? tt0 + PF_TT : T, nq = (tt1 - tt0) * gqa;
            float *Sc = b->sc + (size_t)omp_get_thread_num() * PF_TT * NH_MAX * ldk;
            const float *KT = b->KT + (size_t)kvh * HD * ldk;
            // key tile outer, queries inner: each 64-key tile stays in L1/L2 for all queries
            float qs_[PF_TT * NH_MAX * 128] __attribute__((aligned(64)));
            int pq[PF_TT * NH_MAX];
            for (int qi = 0; qi < nq; qi++) {
                const int t = tt0 + qi / gqa, h = kvh * gqa + qi % gqa;
                pq[qi] = pos0 + t;
                const float *qv = b->Q + (size_t)t * qd + h * HD;
                for (int d = 0; d < HD; d++) qs_[qi * HD + d] = qv[d] * scale;
            }
            const int pmax = pq[nq - 1];
            for (int s0 = 0; s0 <= pmax; s0 += 64) {
                const float *k0 = KT + s0;
                for (int qi = 0; qi < nq; qi++) {
                    if (s0 > pq[qi]) continue;
                    const float *qv = qs_ + qi * HD;
                    __m512 a0 = _mm512_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
                    for (int d = 0; d < HD; d++) {
                        const __m512 qd_ = _mm512_set1_ps(qv[d]);
                        const float *kd = k0 + (size_t)d * ldk;
                        a0 = _mm512_fmadd_ps(qd_, _mm512_loadu_ps(kd), a0);
                        a1 = _mm512_fmadd_ps(qd_, _mm512_loadu_ps(kd + 16), a1);
                        a2 = _mm512_fmadd_ps(qd_, _mm512_loadu_ps(kd + 32), a2);
                        a3 = _mm512_fmadd_ps(qd_, _mm512_loadu_ps(kd + 48), a3);
                    }
                    float *row = Sc + (size_t)qi * ldk + s0;
                    _mm512_storeu_ps(row, a0); _mm512_storeu_ps(row + 16, a1);
                    _mm512_storeu_ps(row + 32, a2); _mm512_storeu_ps(row + 48, a3);
                }
            }
            for (int qi = 0; qi < nq; qi++) softmax512(Sc + (size_t)qi * ldk, pq[qi] + 1);
            // out[q] = sum_s a[q][s] * V[s], same tiling; V rows are contiguous in d
            __m512 oacc[PF_TT * NH_MAX][8];
            for (int qi = 0; qi < nq; qi++) for (int i = 0; i < 8; i++) oacc[qi][i] = _mm512_setzero_ps();
            const uint16_t *vb = vc + kvh * HD;
            for (int s0 = 0; s0 <= pmax; s0 += 64) {
                for (int qi = 0; qi < nq; qi++) {
                    const int p = pq[qi];
                    if (s0 > p) continue;
                    const int s1 = s0 + 64 <= p + 1 ? s0 + 64 : p + 1;
                    const float *row = Sc + (size_t)qi * ldk;
                    __m512 o0 = oacc[qi][0], o1 = oacc[qi][1], o2 = oacc[qi][2], o3 = oacc[qi][3];
                    __m512 o4 = oacc[qi][4], o5 = oacc[qi][5], o6 = oacc[qi][6], o7 = oacc[qi][7];
                    for (int s2 = s0; s2 < s1; s2++) {
                        const __m512 a = _mm512_set1_ps(row[s2]);
                        const uint16_t *vt = vb + (size_t)s2 * kvd;
                        o0 = _mm512_fmadd_ps(a, f16x16(vt), o0);
                        o1 = _mm512_fmadd_ps(a, f16x16(vt + 16), o1);
                        o2 = _mm512_fmadd_ps(a, f16x16(vt + 32), o2);
                        o3 = _mm512_fmadd_ps(a, f16x16(vt + 48), o3);
                        o4 = _mm512_fmadd_ps(a, f16x16(vt + 64), o4);
                        o5 = _mm512_fmadd_ps(a, f16x16(vt + 80), o5);
                        o6 = _mm512_fmadd_ps(a, f16x16(vt + 96), o6);
                        o7 = _mm512_fmadd_ps(a, f16x16(vt + 112), o7);
                    }
                    oacc[qi][0] = o0; oacc[qi][1] = o1; oacc[qi][2] = o2; oacc[qi][3] = o3;
                    oacc[qi][4] = o4; oacc[qi][5] = o5; oacc[qi][6] = o6; oacc[qi][7] = o7;
                }
            }
            for (int qi = 0; qi < nq; qi++) {
                const int t = tt0 + qi / gqa, h = kvh * gqa + qi % gqa;
                float *out = b->A + (size_t)t * qd + h * HD;
                for (int i = 0; i < 8; i++) _mm512_storeu_ps(out + 16 * i, oacc[qi][i]);
            }
        }
        qt_pf[1] += qt_now() - t0;
        t0 = qt_now();
        quantize_rows_u8(b->A, T, qd, b->XU, b->DX);
        mm_q3k(b, L->wo.data, L->wo.row_bytes, L->wo.rows, qd, T, b->XU, b->DX, b->XO, D);
        for (size_t i = 0; i < (size_t)T * D; i++) b->X[i] += b->XO[i];
        qt_pf[0] += qt_now() - t0;

        // ---- routing ----
        t0 = qt_now();
        #pragma omp parallel for schedule(static)
        for (int t = 0; t < T; t++) {
            float *xn = b->XN + (size_t)t * D, *rl = b->rl + (size_t)t * NE;
            rmsnorm(xn, b->X + (size_t)t * D, L->ffn_norm, D, c->rms_eps);
            for (int e = 0; e < NE; e++) rl[e] = dot_bf16_f32(L->router + (size_t)e * D, xn, D);
            float mx = rl[0];
            for (int e = 1; e < NE; e++) if (rl[e] > mx) mx = rl[e];
            for (int e = 0; e < NE; e++) rl[e] = expf(rl[e] - mx);
            int *sel = b->sel + t * K;
            float *ws = b->wsel + t * K, wsum = 0;
            for (int j = 0; j < K; j++) {
                int best = -1;
                for (int e = 0; e < NE; e++) {
                    int taken = 0;
                    for (int q = 0; q < j; q++) taken |= sel[q] == e;
                    if (!taken && (best < 0 || rl[e] > rl[best])) best = e;
                }
                sel[j] = best;
                ws[j] = rl[best];
                wsum += ws[j];
            }
            for (int j = 0; j < K; j++) ws[j] /= wsum;
        }
        quantize_rows_u8(b->XN, T, D, b->XU, b->DX);
        memset(b->cnt, 0, NE * 4);
        for (int i = 0; i < T * K; i++) b->cnt[b->sel[i]]++;
        b->off[0] = 0;
        for (int e = 0; e < NE; e++) b->off[e + 1] = b->off[e] + b->cnt[e];
        memset(b->cnt, 0, NE * 4);
        for (int i = 0; i < T * K; i++) {
            const int e = b->sel[i], at = b->off[e] + b->cnt[e]++;
            b->tok[at] = i / K;
            b->tokw[at] = b->wsel[i];
        }
        qt_pf[2] += qt_now() - t0;

        // ---- experts: cached ones first while the misses stream in ----
        t0 = qt_now();
        double w0 = experts_wait_seconds(m->experts);
        memset(b->XO, 0, (size_t)T * D * 4);
        int miss[256], nm = 0, hq[256], issued = 0;
        const uint8_t *gw, *uw, *dw;
        for (int e = 0; e < NE; e++)
            if (b->cnt[e] && !experts_try_pin(m->experts, l, e, &gw, &uw, &dw)) miss[nm++] = e;
#define TOPUP() while (issued < nm) { int h_ = experts_stream(m->experts, l, miss[issued]); if (h_ < 0) break; hq[issued++] = h_; }
        TOPUP();
        for (int e = 0, mi = 0; e < NE; e++) {
            if (!b->cnt[e]) continue;
            if (mi < nm && miss[mi] == e) { mi++; continue; }
            experts_try_pin(m->experts, l, e, &gw, &uw, &dw);   // second pin: pointers again
            experts_release(m->experts, l, e);
            pf_expert(m, b, L, e, gw, uw, dw);
            experts_release(m->experts, l, e);
            TOPUP();
        }
        for (int i = 0; i < nm; i++) {
            TOPUP();
            experts_stream_get(m->experts, hq[i], &gw, &uw, &dw);
            pf_expert(m, b, L, miss[i], gw, uw, dw);
            experts_stream_done(m->experts, hq[i]);
        }
#undef TOPUP
        m->expert_loads += nm;
        qt_pf[4] += experts_wait_seconds(m->experts) - w0;
        for (size_t i = 0; i < (size_t)T * D; i++) b->X[i] += b->XO[i];
        qt_pf[3] += qt_now() - t0;
    }
}

float *model_prefill(qmodel *m, const int *tokens, int n, int pos0) {
    const qcfg *c = &m->c;
    int ok = 1;   // batched kernels cover Q3_K only; otherwise fall back
    for (int l = 0; l < c->n_layer && ok; l++) {
        const qlayer *L = &m->layers[l];
        ok = L->wq.type == GT_Q3_K && L->wk.type == GT_Q3_K && L->wv.type == GT_Q3_K && L->wo.type == GT_Q3_K &&
             L->gate_exps.type == GT_Q3_K && L->up_exps.type == GT_Q3_K && L->down_exps.type == GT_Q3_K;
    }
    if (!ok || n < 16 || getenv("QUARRY_PER_TOKEN")) {
        float *lg = NULL;
        for (int i = 0; i < n; i++) lg = model_forward(m, tokens[i], pos0 + i);
        return lg;
    }
    // all but the last token in batches; the last one through the normal path for its logits
    int done = 0;
    while (done < n - 1) {
        int T = n - 1 - done;
        if (T > PF_T) T = PF_T;
        pf_batch(m, tokens + done, T, pos0 + done);
        done += T;
    }
    return model_forward(m, tokens[n - 1], pos0 + n - 1);
}
