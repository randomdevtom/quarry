#define _GNU_SOURCE
#include "model.h"

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "quants.h"
#include <immintrin.h>
#include <time.h>
double qt_prof[8];   // 0 attn matvec, 1 attn core, 2 router+pred, 3 expert compute, 4 output, 5 norms/rope/misc
static double qt_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
#define PT(i, stmt) do { double _t = qt_now(); stmt; qt_prof[i] += qt_now() - _t; } while (0)

static void *xalloc(size_t n) {
    void *p = aligned_alloc(64, (n + 63) / 64 * 64);
    if (!p) { fprintf(stderr, "out of memory (%zu bytes)\n", n); exit(1); }
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
        L->router = load_f32(g, NM("ffn_gate_inp.weight"));
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
    size_t kv = (size_t)c->n_layer * n_ctx * kv_dim * sizeof(float);
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
static void rope_neox(float *v, int head_dim, int pos, float base) {
    const int half = head_dim / 2;
    for (int i = 0; i < half; i++) {
        float theta = pos * powf(base, -2.0f * i / head_dim);
        float c = cosf(theta), s = sinf(theta);
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

static void route(const float *router, const float *xn, int n_expert, int D, int k, float *rl,
                  int *sel, float *wsel) {
    #pragma omp parallel for schedule(static)
    for (int e = 0; e < n_expert; e++) rl[e] = dot_f32(router + (size_t)e * D, xn, D);
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

float *model_forward(qmodel *m, int token, int pos) {
    const qcfg *c = &m->c;
    const int D = c->n_embd, HD = c->head_dim, NH = c->n_head, NKV = c->n_head_kv;
    const int kv_dim = NKV * HD, q_dim = NH * HD, gqa = NH / NKV;
    block_q8_K *xq = m->xq;
    embed(m, token, m->x);

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
            rope_neox(m->q + h * HD, HD, pos, c->rope_base);
        }
        for (int h = 0; h < NKV; h++) {
            rmsnorm(m->k + h * HD, m->k + h * HD, L->k_norm, HD, c->rms_eps);
            rope_neox(m->k + h * HD, HD, pos, c->rope_base);
        }
        float *kc = m->k_cache + ((size_t)l * c->n_ctx) * kv_dim;
        float *vc = m->v_cache + ((size_t)l * c->n_ctx) * kv_dim;
        memcpy(kc + (size_t)pos * kv_dim, m->k, kv_dim * 4);
        memcpy(vc + (size_t)pos * kv_dim, m->v, kv_dim * 4);
        const float scale = 1.0f / sqrtf((float)HD);
        double _ta = qt_now();
        #pragma omp parallel for schedule(static)
        for (int h = 0; h < NH; h++) {
            const float *qh = m->q + h * HD;
            float *att = m->att + (size_t)h * c->n_ctx;
            const int kvh = h / gqa;
            float mx = -1e30f;
            for (int t = 0; t <= pos; t++) {
                const float *kt = kc + (size_t)t * kv_dim + kvh * HD;
                float s = 0;
                for (int i = 0; i < HD; i++) s += qh[i] * kt[i];
                att[t] = s * scale;
                if (att[t] > mx) mx = att[t];
            }
            float sum = 0;
            for (int t = 0; t <= pos; t++) { att[t] = expf(att[t] - mx); sum += att[t]; }
            float *out = m->xb + h * HD;
            for (int i = 0; i < HD; i++) out[i] = 0;
            for (int t = 0; t <= pos; t++) {
                const float a = att[t] / sum;
                const float *vt = vc + (size_t)t * kv_dim + kvh * HD;
                for (int i = 0; i < HD; i++) out[i] += a * vt[i];
            }
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
        experts_begin_layer(m->experts, l, sel, K);
        // prediction accuracy of the previous layer's guess for this layer
        if (m->prefetch_on && l > 0) {
            for (int j = 0; j < K; j++)
                for (int q = 0; q < K; q++) m->pred_correct += m->pred[l * K + q] == sel[j];
            m->pred_total += K;
        }
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
        for (int pass = 0; pass < 2; pass++) {       // pass 0: experts already in RAM, as one batch
            int bl[16], nb = 0;
            for (int k = 0; k < K; k++) {
                if (done[k]) continue;
                if (pass == 0 && !experts_ready(m->experts, l, sel[k])) continue;
                bl[nb++] = k;
                if (pass == 1) break;                // pass 1: one at a time as they arrive
            }
            if (!nb) { if (pass == 1) break; continue; }
            if (pass == 1) pass = 0;                 // loop until every expert is done
            const uint8_t *gw[16], *uw[16], *dw[16];
            for (int j = 0; j < nb; j++) experts_get(m->experts, l, sel[bl[j]], &gw[j], &uw[j], &dw[j]);
            const int F = c->n_ff_exp;
            const uint64_t grb = L->gate_exps.row_bytes, urb = L->up_exps.row_bytes, drb = L->down_exps.row_bytes;
            float *H = m->eh, *G = m->eg;            // [nb][F]
            #pragma omp parallel for schedule(static)
            for (int r = 0; r < nb * 2 * F; r++) {
                int j = r / (2 * F), rr = r % (2 * F);
                if (rr < F) G[j * F + rr] = vec_dot_q3_K_q8_K(D, (const block_q3_K *)(gw[j] + rr * grb), xq);
                else H[j * F + rr - F] = vec_dot_q3_K_q8_K(D, (const block_q3_K *)(uw[j] + (rr - F) * urb), xq);
            }
            block_q8_K *hq = m->ehq;                 // [nb][F/QK_K]
            const int nbq = F / QK_K;
            for (int j = 0; j < nb; j++) {
                for (int i = 0; i < F; i++) G[j * F + i] = silu(G[j * F + i]) * H[j * F + i];
                quantize_row_q8_K(G + j * F, hq + j * nbq, F);
            }
            float *O = m->eo;                        // [nb][D]
            #pragma omp parallel for schedule(static)
            for (int r = 0; r < nb * D; r++) {
                int j = r / D, rr = r % D;
                O[r] = vec_dot_q3_K_q8_K(F, (const block_q3_K *)(dw[j] + rr * drb), hq + j * nbq);
            }
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
