// Qwen3.5 / Qwen3.6 MoE forward pass, one token at a time (see qwen35.h).
#define _GNU_SOURCE
#include "qwen35.h"

#include <immintrin.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/syscall.h>

#include "quants.h"

double q35_prof[8], q35_mp[8];   // q35_mp (prefill moe): 0 shared, 1 route, 2 cached experts, 3 streamed experts, 4 stream wait   // 0 delta-net, 1 attention, 2 router, 3 experts, 4 shared expert, 5 output
static double qnow(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

static void *xalloc(size_t n) {
    const size_t al = n >= (1u << 21) ? (1u << 21) : 64, sz = n ? (n + al - 1) / al * al : al;
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
    if (!t && required) { fprintf(stderr, "qwen35: missing tensor %s\n", name); exit(1); }
    return t;
}
static float *f32(qg_file *g, const char *name) {
    const qg_tensor *t = find(g, name, 1);
    if (t->type != GT_F32) { fprintf(stderr, "qwen35: %s is not F32\n", name); exit(1); }
    float *p = xalloc(t->nbytes);
    read_at(g, p, t->nbytes, t->offset);
    return p;
}
static qmat mat(qg_file *g, const char *name, int in_ram, int n_split) {
    const qg_tensor *t = find(g, name, 1);
    qmat q = {t->type, (int)(t->ne[1] * t->ne[2] / n_split), (int)t->ne[0], qg_row_bytes(t->type, t->ne[0]), NULL, t->offset};
    if (in_ram) { q.data = xalloc(t->nbytes); read_at(g, q.data, t->nbytes, t->offset); }
    return q;
}

int q35_detect(const char *path) {
    qg_file g;
    if (qg_open(&g, path)) return 0;
    const char *a = qg_get_str(&g, "general.architecture");
    int r = a && (!strcmp(a, "qwen35moe") || !strcmp(a, "mellum"));
    qg_close(&g);
    return r;
}

static int arr_int(qg_file *g, const char *key, int i, int def) {
    const qg_kv *kv = qg_find_kv(g, key);
    return kv && kv->type == GV_ARR && (uint64_t)i < kv->arr_n ? (int)qg_arr_int(kv, i) : def;
}

int q35_load(q35_model *m, const char *path, int n_ctx) {
    memset(m, 0, sizeof *m);
    if (qg_open(&m->g, path)) return -1;
    qg_file *g = &m->g;
    q35_cfg *c = &m->c;
    const char *arch = qg_get_str(g, "general.architecture");
    c->mellum = arch && !strcmp(arch, "mellum");
    c->gated = !c->mellum;
    char kbuf[128];
#define K(s) (snprintf(kbuf, sizeof kbuf, "%s.%s", c->mellum ? "mellum" : "qwen35moe", s), kbuf)
    // block_count includes the MTP layer(s) at the end
    c->n_layer = qg_get_int(g, K("block_count"), 0) - qg_get_int(g, K("nextn_predict_layers"), 0);
    {
        char t[64];
        snprintf(t, sizeof t, "blk.%d.nextn.eh_proj.weight", c->n_layer);
        m->has_mtp = qg_get_int(g, K("nextn_predict_layers"), 0) >= 1 && qg_find_tensor(g, t) && !getenv("QUARRY_NO_MTP");
    }
    c->n_embd = qg_get_int(g, K("embedding_length"), 0);
    c->n_head = qg_get_int(g, K("attention.head_count"), 0);
    c->n_head_kv = qg_get_int(g, K("attention.head_count_kv"), 0);
    c->hd = qg_get_int(g, K("attention.key_length"), 0);
    c->n_rot = qg_get_int(g, K("rope.dimension_count"), c->hd);
    for (int i = 0; i < 4; i++) c->sect[i] = arr_int(g, K("rope.dimension_sections"), i, 0);
    c->rope_base = qg_get_float(g, K("rope.freq_base"), 1e7);
    c->eps = qg_get_float(g, K("attention.layer_norm_rms_epsilon"), 1e-6);
    c->n_expert = qg_get_int(g, K("expert_count"), 0);
    c->n_used = qg_get_int(g, K("expert_used_count"), 0);
    c->n_ff = qg_get_int(g, K("expert_feed_forward_length"), 0);
    c->n_ff_sh = qg_get_int(g, K("expert_shared_feed_forward_length"), 0);
    c->n_conv = qg_get_int(g, K("ssm.conv_kernel"), 4);
    c->dk = qg_get_int(g, K("ssm.state_size"), 128);
    c->n_kh = qg_get_int(g, K("ssm.group_count"), 0);
    c->n_vh = qg_get_int(g, K("ssm.time_step_rank"), 0);
    c->value_dim = qg_get_int(g, K("ssm.inner_size"), 0);
    c->dv = c->n_vh ? c->value_dim / c->n_vh : 0;
    if (c->mellum) {
        c->n_swa = qg_get_int(g, K("attention.sliding_window"), 0);
        c->rope_base_swa = qg_get_float(g, K("rope.freq_base_swa"), c->rope_base);
        // YaRN on the full-attention layers, as ggml's rope_yarn computes it (llama.cpp
        // context defaults: ext_factor 1, attn_factor cancelled so mscale = 1 + 0.1 ln(factor))
        const char *st = qg_get_str(g, K("rope.scaling.type"));
        const float factor = qg_get_float(g, K("rope.scaling.factor"), 1.0f);
        if (st && !strcmp(st, "yarn") && factor > 1.0f) {
            c->yarn_freq_scale = 1.0f / factor;
            c->yarn_mscale = 1.0f + 0.1f * logf(factor);
            const float orig = (float)qg_get_int(g, K("rope.scaling.original_context_length"), 8192);
            const float bf = qg_get_float(g, K("rope.scaling.yarn_beta_fast"), 32.0f), bs = qg_get_float(g, K("rope.scaling.yarn_beta_slow"), 1.0f);
            const float nd = (float)c->hd;
            const float lo = nd * logf(orig / (bf * 2 * (float)M_PI)) / (2 * logf(c->rope_base));
            const float hi = nd * logf(orig / (bs * 2 * (float)M_PI)) / (2 * logf(c->rope_base));
            c->yarn_corr[0] = floorf(lo) > 0 ? floorf(lo) : 0;
            c->yarn_corr[1] = ceilf(hi) < nd - 1 ? ceilf(hi) : nd - 1;
        }
    }
    c->key_dim = c->dk * c->n_kh;
    c->conv_dim = 2 * c->key_dim + c->value_dim;
    c->n_ctx = n_ctx;
    const int interval = qg_get_int(g, K("full_attention_interval"), 4);
    // QUARRY_TOPK: fewer experts per token than trained (lossy, faster)
    if (getenv("QUARRY_TOPK") && atoi(getenv("QUARRY_TOPK")) > 0 && atoi(getenv("QUARRY_TOPK")) < c->n_used) c->n_used = atoi(getenv("QUARRY_TOPK"));
    const qg_tensor *emb = find(g, "token_embd.weight", 1);
    c->n_vocab = emb->ne[1];
    if ((!c->mellum && c->dk != c->dv) || c->hd > 512 || c->hd % 128 || c->n_head / c->n_head_kv > 8 || c->n_used > 16 || c->n_conv > 8 || c->conv_dim % 16 || c->dv % 16) { fprintf(stderr, "qwen35: unsupported shape\n"); return -1; }

    m->tok_embd = mat(g, "token_embd.weight", 0, 1);
    m->output_norm = f32(g, "output_norm.weight");
    {
        const char *on = qg_find_tensor(g, "output.weight") ? "output.weight" : "token_embd.weight";
        // default: a 4-bit copy in RAM finds the top candidates, their exact logits come from the
        // file's rows (read from the SSD). QUARRY_EXACT_OUTPUT=1 keeps the whole matrix in RAM.
        const int exact = getenv("QUARRY_EXACT_OUTPUT") || c->n_embd % 256;
        m->output = mat(g, on, exact, 1);
        if (!exact) {
            const qmat *O = &m->output;
            const int nb = O->cols / 64;
            m->out4_rb = nb * 36;
            m->out4 = xalloc((size_t)O->rows * m->out4_rb);
            enum { CH = 2048 };
            uint8_t *raw = malloc((size_t)CH * O->row_bytes);
            for (int r0 = 0; r0 < O->rows; r0 += CH) {
                const int nr = O->rows - r0 < CH ? O->rows - r0 : CH;
                read_at(g, raw, (uint64_t)nr * O->row_bytes, O->file_off + (uint64_t)r0 * O->row_bytes);
                #pragma omp parallel
                {
                    float *f = malloc((size_t)O->cols * 4);
                    #pragma omp for schedule(static)
                    for (int r = 0; r < nr; r++) {
                        dequant_row(O->type, raw + (size_t)r * O->row_bytes, f, O->cols);
                        uint8_t *dst = m->out4 + (size_t)(r0 + r) * m->out4_rb;
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
            free(raw);
        }
    }
    m->layers = calloc(c->n_layer + 1, sizeof(q35_layer));
    m->attn_idx = malloc(c->n_layer * sizeof(int));
    m->rec_idx = malloc(c->n_layer * sizeof(int));
    char nm[128];
#define NM(s) (snprintf(nm, sizeof nm, "blk.%d." s, l), nm)
    for (int l = 0; l < c->n_layer + m->has_mtp; l++) {
        q35_layer *L = &m->layers[l];
        if (l == c->n_layer) {   // MTP block: full attention, its own KV
            L->is_attn = 1;
            m->eh_proj = mat(g, NM("nextn.eh_proj.weight"), 1, 1);
            m->mtp_enorm = f32(g, NM("nextn.enorm.weight"));
            m->mtp_hnorm = f32(g, NM("nextn.hnorm.weight"));
            m->mtp_head_norm = qg_find_tensor(g, NM("nextn.shared_head_norm.weight")) ? f32(g, NM("nextn.shared_head_norm.weight")) : m->output_norm;
        } else {
            const int rec = c->mellum ? 0 : arr_int(g, K("attention.recurrent_layers"), l, (l + 1) % interval != 0);
            L->is_attn = !rec;
            if (c->mellum) L->is_swa = c->n_swa > 0 && arr_int(g, K("attention.sliding_window_pattern"), l, 0);
            m->attn_idx[l] = L->is_attn ? m->n_attn++ : -1;
            m->rec_idx[l] = L->is_attn ? -1 : m->n_rec++;
        }
        L->attn_norm = f32(g, NM("attn_norm.weight"));
        L->post_norm = f32(g, c->mellum ? NM("ffn_norm.weight") : NM("post_attention_norm.weight"));
        if (L->is_attn) {
            L->wq = mat(g, NM("attn_q.weight"), 1, 1);
            L->wk = mat(g, NM("attn_k.weight"), 1, 1);
            L->wv = mat(g, NM("attn_v.weight"), 1, 1);
            L->wo = mat(g, NM("attn_output.weight"), 1, 1);
            L->q_norm = f32(g, NM("attn_q_norm.weight"));
            L->k_norm = f32(g, NM("attn_k_norm.weight"));
        } else {
            L->wqkv = mat(g, NM("attn_qkv.weight"), 1, 1);
            L->wz = mat(g, NM("attn_gate.weight"), 1, 1);
            L->walpha = mat(g, NM("ssm_alpha.weight"), 1, 1);
            L->wbeta = mat(g, NM("ssm_beta.weight"), 1, 1);
            L->wout = mat(g, NM("ssm_out.weight"), 1, 1);
            L->conv_w = f32(g, NM("ssm_conv1d.weight"));
            L->conv_wt = xalloc((size_t)c->conv_dim * c->n_conv * 4);
            for (int ch = 0; ch < c->conv_dim; ch++)
                for (int k = 0; k < c->n_conv; k++) L->conv_wt[(size_t)k * c->conv_dim + ch] = L->conv_w[(size_t)ch * c->n_conv + k];
            L->dt_bias = f32(g, NM("ssm_dt.bias"));
            L->ssm_a = f32(g, NM("ssm_a"));
            L->ssm_norm = f32(g, NM("ssm_norm.weight"));
        }
        {   // router as bf16 when the F32 values are exact bf16 (they are for Qwen checkpoints)
            float *rf = f32(g, NM("ffn_gate_inp.weight"));
            const size_t nr = (size_t)c->n_expert * c->n_embd;
            int exact = 1;
            for (size_t i = 0; i < nr && exact; i++) { uint32_t u; memcpy(&u, rf + i, 4); exact = !(u & 0xffff); }
            if (exact) {
                L->router = xalloc(nr * 2);
                for (size_t i = 0; i < nr; i++) { uint32_t u; memcpy(&u, rf + i, 4); L->router[i] = (uint16_t)(u >> 16); }
                free(rf);
            } else L->router_f32 = rf;
        }
        if (find(g, NM("ffn_gate_shexp.weight"), 0)) {   // shared expert (Qwen; Mellum has none)
            L->shexp_gate = f32(g, NM("ffn_gate_inp_shexp.weight"));
            L->sh_gate = mat(g, NM("ffn_gate_shexp.weight"), 1, 1);
            L->sh_up = mat(g, NM("ffn_up_shexp.weight"), 1, 1);
            L->sh_down = mat(g, NM("ffn_down_shexp.weight"), 1, 1);
        }
        L->gate_exps = mat(g, NM("ffn_gate_exps.weight"), 0, c->n_expert);
        L->up_exps = mat(g, NM("ffn_up_exps.weight"), 0, c->n_expert);
        L->down_exps = mat(g, NM("ffn_down_exps.weight"), 0, c->n_expert);
    }
#undef NM
#undef K
    const int kvd = c->n_head_kv * c->hd;
    m->kc = xalloc((size_t)m->n_attn * n_ctx * kvd * 2);
    m->vc = xalloc((size_t)m->n_attn * n_ctx * kvd * 2);
    const size_t hsz = (size_t)c->n_vh * c->dv * c->dk;
    m->S = xalloc((size_t)m->n_rec * hsz * 4);
    m->sp = malloc(m->n_rec * sizeof(float *));
    m->sp_alt = malloc(m->n_rec * sizeof(float *));
    for (int r = 0; r < m->n_rec; r++) m->sp[r] = m->S + (size_t)r * hsz;
    if (m->has_mtp) {
        m->S2 = xalloc((size_t)m->n_rec * hsz * 4);
        for (int r = 0; r < m->n_rec; r++) m->sp_alt[r] = m->S2 + (size_t)r * hsz;
        m->conv_bk = xalloc((size_t)m->n_rec * (c->n_conv - 1) * c->conv_dim * 4);
        m->mtp_kc = xalloc((size_t)n_ctx * kvd * 2);
        m->mtp_vc = xalloc((size_t)n_ctx * kvd * 2);
        for (int t = 0; t < 2; t++) { m->h2[t] = xalloc(c->n_embd * 4); m->logits2[t] = xalloc((size_t)c->n_vocab * 4); }
    }
    m->hlast = xalloc(c->n_embd * 4);
    m->conv = xalloc((size_t)m->n_rec * (c->n_conv - 1) * c->conv_dim * 4);
    m->logits = xalloc((size_t)c->n_vocab * 4);
    m->pred = calloc((size_t)(c->n_layer + 1) * 16, sizeof(int));

    // experts: three parts (gate, up, down); their types differ by layer
    const int nl = c->n_layer + m->has_mtp;
    uint64_t *off = malloc(nl * 3 * sizeof(uint64_t)), *bytes = malloc(nl * 3 * sizeof(uint64_t));
    for (int l = 0; l < nl; l++) {
        const q35_layer *L = &m->layers[l];
        const qmat *P[3] = {&L->gate_exps, &L->up_exps, &L->down_exps};
        for (int p = 0; p < 3; p++) { off[l * 3 + p] = P[p]->file_off; bytes[l * 3 + p] = P[p]->row_bytes * P[p]->rows; }
    }
    expert_src src = {g->fd, g->file_size, nl, c->n_expert, 3, off, bytes,
                      &m->expert_bytes_read, &m->expert_loads, 2 * c->n_used + 4};
    // <model>.experts: experts repacked so each one is a single contiguous read. Faster in a
    // raw read test (1.45 vs 1.07 GB/s) but slower inside the engine on our laptop (I/O wait
    // 21s vs 14s), so opt-in: QUARRY_WRITE_PACKED=1 creates it, QUARRY_PACKED=1 uses it.
    char packed[4096];
    snprintf(packed, sizeof packed, "%s.experts", path);
    if (getenv("QUARRY_WRITE_PACKED") && experts_write_packed(&src, packed)) fprintf(stderr, "quarry: packing failed\n");
    if (getenv("QUARRY_PACKED") && access(packed, R_OK) == 0) src.packed = packed;
    m->experts = experts_open_src(&src);
    free(off); free(bytes);
    m->prefetch_on = getenv("QUARRY_NO_PREFETCH") ? 0 : 1;
    return 0;
}

size_t q35_state_bytes(const q35_model *m) {
    const q35_cfg *c = &m->c;
    return (size_t)m->n_rec * c->n_vh * c->dv * c->dk * 4 + (size_t)m->n_rec * (c->n_conv - 1) * c->conv_dim * 4;
}
void q35_state_save(const q35_model *m, void *dst) {
    const q35_cfg *c = &m->c;
    const size_t h = (size_t)c->n_vh * c->dv * c->dk * 4;
    for (int r = 0; r < m->n_rec; r++) memcpy((char *)dst + r * h, m->sp[r], h);
    memcpy((char *)dst + m->n_rec * h, m->conv, (size_t)m->n_rec * (c->n_conv - 1) * c->conv_dim * 4);
}
void q35_state_load(q35_model *m, const void *src) {
    const q35_cfg *c = &m->c;
    const size_t h = (size_t)c->n_vh * c->dv * c->dk * 4;
    for (int r = 0; r < m->n_rec; r++) memcpy(m->sp[r], (const char *)src + r * h, h);
    memcpy(m->conv, (const char *)src + m->n_rec * h, (size_t)m->n_rec * (c->n_conv - 1) * c->conv_dim * 4);
    m->mtp_n = 0;
}
void q35_state_reset(q35_model *m) {
    const q35_cfg *c = &m->c;
    const size_t h = (size_t)c->n_vh * c->dv * c->dk * 4;
    for (int r = 0; r < m->n_rec; r++) memset(m->sp[r], 0, h);
    memset(m->conv, 0, (size_t)m->n_rec * (c->n_conv - 1) * c->conv_dim * 4);
    m->mtp_n = 0;
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
static inline __m512 bf16x16(const uint16_t *p) {
    return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *)p)), 16));
}
static inline float dot_bf16(const uint16_t *a, const float *b, int n) {
    __m512 s0 = _mm512_setzero_ps(), s1 = s0;
    for (int i = 0; i < n; i += 32) {
        s0 = _mm512_fmadd_ps(bf16x16(a + i), _mm512_loadu_ps(b + i), s0);
        s1 = _mm512_fmadd_ps(bf16x16(a + i + 16), _mm512_loadu_ps(b + i + 16), s1);
    }
    return _mm512_reduce_add_ps(_mm512_add_ps(s0, s1));
}
static void rmsnorm(float *o, const float *x, const float *w, int n, float eps) {
    double ss = 0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    const float s = 1.0f / sqrtf((float)(ss / n) + eps);
    for (int i = 0; i < n; i++) o[i] = x[i] * s * (w ? w[i] : 1.0f);
}
static float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }
static float siluf(float x) { return x / (1.0f + expf(-x)); }
static float softplusf(float x) { return x > 20.0f ? x : log1pf(expf(x)); }

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
static inline __m512 silu_ps(__m512 x) {
    const __m512 one = _mm512_set1_ps(1.0f);
    return _mm512_div_ps(x, _mm512_add_ps(one, exp_ps(_mm512_sub_ps(_mm512_setzero_ps(), x))));
}
// g[i] = silu(g[i]) * h[i]
static void silu_mul(float *g, const float *h, size_t n) {
    size_t i = 0;
    for (; i + 16 <= n; i += 16) _mm512_storeu_ps(g + i, _mm512_mul_ps(silu_ps(_mm512_loadu_ps(g + i)), _mm512_loadu_ps(h + i)));
    for (; i < n; i++) g[i] = siluf(g[i]) * h[i];
}

static inline void f32_to_f16(uint16_t *dst, const float *src, int n) {
    for (int i = 0; i < n; i += 16)
        _mm256_storeu_si256((__m256i *)(dst + i), _mm512_cvtps_ph(_mm512_loadu_ps(src + i), _MM_FROUND_TO_NEAREST_INT));
}
static inline __m512 f16x16(const uint16_t *p) { return _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)p)); }

// quantized activations for one input vector, both layouts, made on demand
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
    if (type == GT_Q8_0 || type == GT_Q5_0 || type == GT_Q4_0 || type == GT_Q4_1) {
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
    case GT_Q5_K: return vec_dot_q5_K_q8_K(n, (const block_q5_K *)row, q->qk);
    case GT_Q6_K: return vec_dot_q6_K_q8_K(n, (const block_q6_K *)row, q->qk);
    }
    fprintf(stderr, "qwen35: no dot for type %d\n", type);
    exit(1);
}

// scratch shared by one forward call
typedef struct {
    float *x, *xn, *t1, *t2, *t3, *big, *att, *eg, *eh, *eo, *moe;
    block_q8_K *qk, *qk2;
    int8_t *a_qs, *a_qs2; float *a_d, *a_d2; int32_t *a_s, *a_s2;
} scratch;
static scratch *scr_init(q35_model *m, scratch *sp, int *init);
static scratch *scr(q35_model *m) { static scratch s; static int init; return scr_init(m, &s, &init); }
static scratch *scr1(q35_model *m) { static scratch s; static int init; return scr_init(m, &s, &init); }
static scratch *scr_init(q35_model *m, scratch *sp, int *initp) {
    if (*initp) return sp;
    scratch s;
    const q35_cfg *c = &m->c;
    int mx = c->conv_dim;
    if (c->n_head * c->hd * 2 > mx) mx = c->n_head * c->hd * 2;
    if (c->value_dim > mx) mx = c->value_dim;
    s.x = xalloc(c->n_embd * 4); s.xn = xalloc(c->n_embd * 4);
    s.t1 = xalloc(mx * 4); s.t2 = xalloc(mx * 4); s.t3 = xalloc(mx * 4);
    s.big = xalloc(mx * 4);
    s.att = xalloc((size_t)c->n_head * c->n_ctx * 4);
    s.eg = xalloc(16 * c->n_ff * 4); s.eh = xalloc(16 * c->n_ff * 4);
    s.eo = xalloc(16 * c->n_embd * 4); s.moe = xalloc(c->n_embd * 4);
    s.qk = xalloc((mx / QK_K + 16) * sizeof(block_q8_K) * 16);
    s.qk2 = xalloc((mx / QK_K + 16) * sizeof(block_q8_K) * 16);
    s.a_qs = xalloc(mx * 16); s.a_d = xalloc(mx / 32 * 4 * 16 + 64); s.a_s = xalloc(mx / 32 * 4 * 16 + 64);
    s.a_qs2 = xalloc(mx * 16); s.a_d2 = xalloc(mx / 32 * 4 * 16 + 64); s.a_s2 = xalloc(mx / 32 * 4 * 16 + 64);
    *sp = s;
    *initp = 1;
    return sp;
}

// y[rows] = W x, several matrices sharing one input, in one parallel region
typedef struct { const qmat *W; float *y; } mv_job;
static void mv_multi(const mv_job *jobs, int nj, qin *q) {
    int start[8], total = 0;
    for (int j = 0; j < nj; j++) { qin_need(q, jobs[j].W->type); start[j] = total; total += jobs[j].W->rows; }
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < total; r++) {
        int j = nj - 1;
        while (r < start[j]) j--;
        const qmat *W = jobs[j].W;
        const int rr = r - start[j];
        jobs[j].y[rr] = row_dot(W->type, W->data + (uint64_t)rr * W->row_bytes, W->cols, q);
    }
}

static void embed(q35_model *m, int token, float *out) {
    const qmat *E = &m->tok_embd;
    uint8_t *row = malloc(E->row_bytes);
    read_at(&m->g, row, E->row_bytes, E->file_off + (uint64_t)token * E->row_bytes);
    dequant_row(E->type, row, out, E->cols);
    free(row);
}

// softmax over all experts, top-k, renormalized (qwen35moe: norm_w = true)
static void route(const q35_layer *L, const float *xn, const q35_cfg *c, int k, int *sel, float *w) {
    float rl[512];
    #pragma omp parallel for schedule(static)
    for (int e = 0; e < c->n_expert; e++)
        rl[e] = L->router ? dot_bf16(L->router + (size_t)e * c->n_embd, xn, c->n_embd)
                          : dotf(L->router_f32 + (size_t)e * c->n_embd, xn, c->n_embd);
    float mx = rl[0];
    for (int e = 1; e < c->n_expert; e++) if (rl[e] > mx) mx = rl[e];
    for (int e = 0; e < c->n_expert; e++) rl[e] = expf(rl[e] - mx);
    float ws = 0;
    for (int j = 0; j < k; j++) {
        int best = -1;
        for (int e = 0; e < c->n_expert; e++) {
            int taken = 0;
            for (int q = 0; q < j; q++) taken |= sel[q] == e;
            if (!taken && (best < 0 || rl[e] > rl[best])) best = e;
        }
        sel[j] = best;
        w[j] = rl[best];
        ws += w[j];
    }
    for (int j = 0; j < k; j++) w[j] /= ws;
}

// ---------------- gated delta net (one token) ----------------
static void delta_net(q35_model *m, const q35_layer *L, int ri, const float *xn, float *out, scratch *s) {
    const q35_cfg *c = &m->c;
    qin q;
    qin_init(&q, xn, c->n_embd, s->qk, s->a_qs, s->a_d, s->a_s);
    float *qkv = s->t1, *z = s->t2, *ab = s->t3;     // ab: alpha [n_vh] then beta [n_vh]
    double td = qnow();
    mv_job jobs[4] = {{&L->wqkv, qkv}, {&L->wz, z}, {&L->walpha, ab}, {&L->wbeta, ab + c->n_vh}};
    mv_multi(jobs, 4, &q);
    q35_prof[6] += qnow() - td; td = qnow();
    // short causal conv over [3 previous inputs, current], then SiLU
    float *cs = m->conv + (size_t)ri * (c->n_conv - 1) * c->conv_dim;   // [tap][channel], oldest first
    float *cv = s->big;
    const int nc = c->n_conv;
    for (int ch = 0; ch < c->conv_dim; ch++) {
        const float *w = L->conv_w + (size_t)ch * nc;
        float acc = w[nc - 1] * qkv[ch];
        for (int t = 0; t < nc - 1; t++) acc += w[t] * cs[(size_t)t * c->conv_dim + ch];
        for (int t = 0; t < nc - 2; t++) cs[(size_t)t * c->conv_dim + ch] = cs[(size_t)(t + 1) * c->conv_dim + ch];
        cs[(size_t)(nc - 2) * c->conv_dim + ch] = qkv[ch];
        cv[ch] = siluf(acc);
    }
    float *qv = cv, *kv = cv + c->key_dim, *vv = cv + 2 * c->key_dim;
    // L2-normalize q and k per head: x / sqrt(sum x^2 + eps)
    for (int h = 0; h < 2 * c->n_kh; h++) {
        float *p = cv + (size_t)h * c->dk;
        float ss = 0;
        for (int i = 0; i < c->dk; i++) ss += p[i] * p[i];
        const float r = 1.0f / sqrtf(ss + c->eps);
        for (int i = 0; i < c->dk; i++) p[i] *= r;
    }
    float *S = m->sp[ri];
    const float scale = 1.0f / sqrtf((float)c->dv);
    float *o = s->t1;   // qkv no longer needed
    const int D = c->dk;
    #pragma omp parallel for schedule(static)
    for (int h = 0; h < c->n_vh; h++) {
        const float g = expf(softplusf(ab[h] + L->dt_bias[h]) * L->ssm_a[h]);
        const float beta = sigmoidf(ab[c->n_vh + h]);
        const float *kh = kv + (size_t)(h % c->n_kh) * D, *qh = qv + (size_t)(h % c->n_kh) * D;
        const float *vh = vv + (size_t)h * c->dv;
        float *M = S + (size_t)h * c->dv * D;
        // row j of M = column j of the state; one pass per row:
        // decay, delta = (v - M k) beta, M += delta k, out = M q
        for (int j = 0; j < c->dv; j++) {
            float *row = M + (size_t)j * D;
            __m512 acc = _mm512_setzero_ps();
            const __m512 gv = _mm512_set1_ps(g);
            for (int i = 0; i < D; i += 16) {
                __m512 r = _mm512_mul_ps(_mm512_loadu_ps(row + i), gv);
                _mm512_storeu_ps(row + i, r);
                acc = _mm512_fmadd_ps(r, _mm512_loadu_ps(kh + i), acc);
            }
            const float delta = (vh[j] - _mm512_reduce_add_ps(acc)) * beta;
            const __m512 dv = _mm512_set1_ps(delta);
            __m512 oa = _mm512_setzero_ps();
            for (int i = 0; i < D; i += 16) {
                __m512 r = _mm512_fmadd_ps(dv, _mm512_loadu_ps(kh + i), _mm512_loadu_ps(row + i));
                _mm512_storeu_ps(row + i, r);
                oa = _mm512_fmadd_ps(r, _mm512_loadu_ps(qh + i), oa);
            }
            o[(size_t)h * c->dv + j] = _mm512_reduce_add_ps(oa) * scale;
        }
        // gated norm: rmsnorm(o_h) * w * silu(z_h)
        float *oh = o + (size_t)h * c->dv;
        const float *zh = z + (size_t)h * c->dv;
        float ss = 0;
        for (int j = 0; j < c->dv; j++) ss += oh[j] * oh[j];
        const float r = 1.0f / sqrtf(ss / c->dv + c->eps);
        for (int j = 0; j < c->dv; j++) oh[j] = oh[j] * r * L->ssm_norm[j] * siluf(zh[j]);
    }
    q35_prof[7] += qnow() - td; td = qnow();
    qin q2;
    qin_init(&q2, o, c->value_dim, s->qk2, s->a_qs2, s->a_d2, s->a_s2);
    mv_job j2 = {&L->wout, out};
    mv_multi(&j2, 1, &q2);
    q35_prof[6] += qnow() - td;
}

// ---------------- gated full attention (one token) ----------------
static void softmax_inplace(float *x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float s = 0;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); s += x[i]; }
    const float r = 1.0f / s;
    for (int i = 0; i < n; i++) x[i] *= r;
}
// RoPE cos/sin for one position (NEOX pairs i, i + n_rot/2). Mellum's full-attention layers
// use YaRN (ggml rope_yarn: interpolated/extrapolated mix per dim, cos/sin scaled by mscale).
static void rope_cs(const q35_cfg *c, const q35_layer *L, int pos, int py, int px, float *cs, float *sn) {
    const int half = c->n_rot / 2;
    const int nsect = c->sect[0] + c->sect[1] + c->sect[2] + c->sect[3];
    const int yarn = c->mellum && !L->is_swa && c->yarn_freq_scale > 0;
    const float base = c->mellum && L->is_swa ? c->rope_base_swa : c->rope_base;
    for (int i = 0; i < half; i++) {
        // interleaved M-RoPE (ggml IMROPE): dims cycle t, y, x within their section limits;
        // text tokens have t = y = x, which makes this plain RoPE
        int p = pos;
        if (nsect) {
            const int sec = i % nsect;
            if (sec % 3 == 1 && sec < 3 * c->sect[1]) p = py;
            else if (sec % 3 == 2 && sec < 3 * c->sect[2]) p = px;
        }
        const float ext = p * powf(base, -2.0f * i / c->n_rot);
        float th = ext, ms = 1.0f;
        if (yarn) {
            const float interp = c->yarn_freq_scale * ext;
            const float y = (i - c->yarn_corr[0]) / fmaxf(0.001f, c->yarn_corr[1] - c->yarn_corr[0]);
            const float mix = 1.0f - fminf(1.0f, fmaxf(0.0f, y));
            th = interp * (1 - mix) + ext * mix;
            ms = c->yarn_mscale;
        }
        cs[i] = cosf(th) * ms; sn[i] = sinf(th) * ms;
    }
}
// first cache row a query at row S - 1 may attend to (sliding window), else 0
static inline int attn_lo(const q35_cfg *c, const q35_layer *L, int S) {
    return L->is_swa && S > c->n_swa ? S - c->n_swa : 0;
}

// per head: [q | gate]; q and k get an RMS norm, then RoPE (NEOX) on the first n_rot dims.
// q goes to qh_all; k and v are written to the cache at row widx.
static void attn_prep(const q35_model *m, const q35_layer *L, const float *qg, float *k, const float *v, float *qh_all,
                      int rpos, uint16_t *kc, uint16_t *vc, int widx) {
    const q35_cfg *c = &m->c;
    const int HD = c->hd, NH = c->n_head, NKV = c->n_head_kv, kvd = NKV * HD;
    const int half = c->n_rot / 2, qs = c->gated ? 2 * HD : HD;
    float cs[256], sn[256];
    const int rp = rpos + m->rope_delta;
    rope_cs(c, L, rp, rp, rp, cs, sn);
    for (int h = 0; h < NH + NKV; h++) {
        float *p = h < NH ? qh_all + (size_t)h * HD : k + (size_t)(h - NH) * HD;
        if (h < NH) memcpy(p, qg + (size_t)h * qs, HD * 4);
        rmsnorm(p, p, h < NH ? L->q_norm : L->k_norm, HD, c->eps);
        for (int i = 0; i < half; i++) {
            const float a = p[i], b = p[i + half];
            p[i] = a * cs[i] - b * sn[i];
            p[i + half] = a * sn[i] + b * cs[i];
        }
    }
    f32_to_f16(kc + (size_t)widx * kvd, k, kvd);
    f32_to_f16(vc + (size_t)widx * kvd, v, kvd);
}

// attention of the heads in qh_all over cache rows [0, S), gated by sigmoid(gate) -> ao [NH * HD]
static void attn_core(const q35_model *m, const uint16_t *kc, const uint16_t *vc, int S, const float *qh_all,
                      const float *qg, float *ao, scratch *s, int lo) {
    const q35_cfg *c = &m->c;
    const int HD = c->hd, NH = c->n_head, NKV = c->n_head_kv, kvd = NKV * HD, gqa = NH / NKV;
    const float scale = 1.0f / sqrtf((float)HD);
    // sliding window: attend over rows [lo, S) only (shift the bases, keep the row math)
    kc += (size_t)lo * kvd; vc += (size_t)lo * kvd; S -= lo;
    // time-blocked: each K/V row is read once for all heads that share it
    const int NBLK = S < 256 ? 1 : (S / 256 < 32 ? S / 256 : 32), bs = (S + NBLK - 1) / NBLK;
    static float *part;
    static size_t part_cap;
    if ((size_t)NBLK * NH * HD > part_cap) { free(part); part_cap = (size_t)32 * NH * HD; part = aligned_alloc(64, part_cap * 4); }
    #pragma omp parallel for schedule(static)
    for (int bk = 0; bk < NBLK; bk++) {
        const int t0 = bk * bs, t1 = t0 + bs < S ? t0 + bs : S;
        for (int t = t0; t < t1; t++)
            for (int kvh = 0; kvh < NKV; kvh++) {
                const uint16_t *kt = kc + (size_t)t * kvd + kvh * HD;
                for (int g = 0; g < gqa; g++) {
                    const int h = kvh * gqa + g;
                    const float *qh = qh_all + (size_t)h * HD;
                    __m512 a0 = _mm512_setzero_ps(), a1 = a0;
                    for (int i = 0; i < HD; i += 32) {
                        a0 = _mm512_fmadd_ps(_mm512_loadu_ps(qh + i), f16x16(kt + i), a0);
                        a1 = _mm512_fmadd_ps(_mm512_loadu_ps(qh + i + 16), f16x16(kt + i + 16), a1);
                    }
                    s->att[(size_t)h * c->n_ctx + t] = _mm512_reduce_add_ps(_mm512_add_ps(a0, a1)) * scale;
                }
            }
    }
    #pragma omp parallel for schedule(static)
    for (int h = 0; h < NH; h++) softmax_inplace(s->att + (size_t)h * c->n_ctx, S);
    #pragma omp parallel for schedule(static)
    for (int bk = 0; bk < NBLK; bk++) {
        const int t0 = bk * bs, t1 = t0 + bs < S ? t0 + bs : S;
        float *pp = part + (size_t)bk * NH * HD;
        memset(pp, 0, (size_t)NH * HD * 4);
        for (int t = t0; t < t1; t++)
            for (int kvh = 0; kvh < NKV; kvh++) {
                const uint16_t *vt = vc + (size_t)t * kvd + kvh * HD;
                for (int i = 0; i < HD; i += 16) {
                    const __m512 v = f16x16(vt + i);
                    for (int g = 0; g < gqa; g++) {
                        const int h = kvh * gqa + g;
                        float *o = pp + (size_t)h * HD + i;
                        _mm512_storeu_ps(o, _mm512_fmadd_ps(_mm512_set1_ps(s->att[(size_t)h * c->n_ctx + t]), v, _mm512_loadu_ps(o)));
                    }
                }
            }
    }
    #pragma omp parallel for schedule(static)
    for (int h = 0; h < NH; h++) {
        float *o = ao + (size_t)h * HD;
        for (int i = 0; i < HD; i += 16) {
            __m512 acc = _mm512_setzero_ps();
            for (int bk = 0; bk < NBLK; bk++) acc = _mm512_add_ps(acc, _mm512_loadu_ps(part + ((size_t)bk * NH + h) * HD + i));
            _mm512_storeu_ps(o + i, acc);
        }
        if (c->gated) {
            const float *gate = qg + (size_t)h * 2 * HD + HD;
            for (int i = 0; i < HD; i++) o[i] *= sigmoidf(gate[i]);
        }
    }
}

// one token: projections, write K/V at row widx (RoPE position rpos), attend over rows [0, S)
static void attention(q35_model *m, const q35_layer *L, uint16_t *kc, uint16_t *vc, int widx, int rpos, int S,
                      const float *xn, float *out, scratch *s) {
    const q35_cfg *c = &m->c;
    const int kvd = c->n_head_kv * c->hd;
    qin q;
    qin_init(&q, xn, c->n_embd, s->qk, s->a_qs, s->a_d, s->a_s);
    float *qg = s->t1, *k = s->t2, *v = s->t3;
    mv_job jobs[3] = {{&L->wq, qg}, {&L->wk, k}, {&L->wv, v}};
    mv_multi(jobs, 3, &q);
    attn_prep(m, L, qg, k, v, s->big, rpos, kc, vc, widx);
    float *ao = s->t2 + kvd;   // after k: NH*HD floats fit (t2 has max(conv_dim, 2*NH*HD))
    attn_core(m, kc, vc, S, s->big, qg, ao, s, attn_lo(c, L, S));
    qin q2;
    qin_init(&q2, ao, c->n_head * c->hd, s->qk2, s->a_qs2, s->a_d2, s->a_s2);
    mv_job j2 = {&L->wo, out};
    mv_multi(&j2, 1, &q2);
}
static uint16_t *kc_layer(q35_model *m, int ai) { return m->kc + (size_t)ai * m->c.n_ctx * m->c.n_head_kv * m->c.hd; }
static uint16_t *vc_layer(q35_model *m, int ai) { return m->vc + (size_t)ai * m->c.n_ctx * m->c.n_head_kv * m->c.hd; }

// ---------------- MoE: routed experts + shared expert ----------------
static void moe(q35_model *m, int l, const float *xn, const float *xn_next, float *out, scratch *s) {
    const q35_cfg *c = &m->c;
    const q35_layer *L = &m->layers[l];
    const int K = c->n_used, F = c->n_ff, D = c->n_embd;
    int sel[16];
    float w[16];
    double t0 = qnow();
    route(L, xn, c, K, sel, w);
    if (m->prefetch_on && l > 0 && l < c->n_layer) {
        for (int j = 0; j < K; j++)
            for (int q = 0; q < (m->pred_n ? m->pred_n : K); q++) m->pred_correct += m->pred[l * 16 + q] == sel[j];
        m->pred_total += K;
    }
    experts_begin_layer(m->experts, l, sel, K);
    if (m->prefetch_on && xn_next && l + 1 < c->n_layer) {   // guess the next layer's experts
        // QUARRY_PF_EXTRA: also prefetch the next few most likely ones (more SSD reads, fewer misses)
        static int extra = -1;
        if (extra < 0) extra = getenv("QUARRY_PF_EXTRA") ? atoi(getenv("QUARRY_PF_EXTRA")) : 0;
        const int kp = K + extra > 16 ? 16 : K + extra;
        float pw[16];
        route(&m->layers[l + 1], xn_next, c, kp, m->pred + (l + 1) * 16, pw);
        // QUARRY_PF_MIN: skip guesses whose (renormalized) router weight is below this
        static float pmin = -1;
        if (pmin < 0) pmin = getenv("QUARRY_PF_MIN") ? atof(getenv("QUARRY_PF_MIN")) : 0;
        int np = kp;
        if (pmin > 0) { np = 0; for (int j = 0; j < kp; j++) if (pw[j] >= pmin) np++; }   // sorted by weight
        experts_prefetch(m->experts, l + 1, m->pred + (l + 1) * 16, np);
        m->pred_n = kp;
    }
    q35_prof[2] += qnow() - t0;

    // shared expert while the routed ones load
    t0 = qnow();
    qin q;
    qin_init(&q, xn, D, s->qk, s->a_qs, s->a_d, s->a_s);
    if (!L->sh_gate.data) memset(out, 0, D * 4);
    else {
    float *sg = s->t1, *su = s->t2;
    mv_job jobs[2] = {{&L->sh_gate, sg}, {&L->sh_up, su}};
    mv_multi(jobs, 2, &q);
    for (int i = 0; i < c->n_ff_sh; i++) sg[i] = siluf(sg[i]) * su[i];
    qin q2;
    qin_init(&q2, sg, c->n_ff_sh, s->qk2, s->a_qs2, s->a_d2, s->a_s2);
    mv_job jd = {&L->sh_down, out};
    mv_multi(&jd, 1, &q2);
    const float shg = sigmoidf(dotf(L->shexp_gate, xn, D));
    for (int i = 0; i < D; i++) out[i] *= shg;
    }
    q35_prof[4] += qnow() - t0;

    // routed experts: everything already in RAM as one batch, then the rest as they arrive
    t0 = qnow();
    qin_need(&q, L->gate_exps.type);
    qin_need(&q, L->up_exps.type);
    const int gt = L->gate_exps.type, ut = L->up_exps.type, dt = L->down_exps.type;
    const uint64_t grb = L->gate_exps.row_bytes, urb = L->up_exps.row_bytes, drb = L->down_exps.row_bytes;
    int done[16] = {0};
    for (int n_done = 0; n_done < K;) {
        int bl[16], nb = 0;
        for (int k = 0; k < K; k++) if (!done[k] && experts_ready(m->experts, l, sel[k])) bl[nb++] = k;
        if (!nb) {
            int rest[16], nr = 0;
            for (int k = 0; k < K; k++) if (!done[k]) rest[nr++] = sel[k];
            experts_wait_any(m->experts, l, rest, nr);
            continue;
        }
        n_done += nb;
        const uint8_t *gw[16], *uw[16], *dw[16];
        for (int j = 0; j < nb; j++) experts_get(m->experts, l, sel[bl[j]], &gw[j], &uw[j], &dw[j]);
        float *G = s->eg, *H = s->eh;
        #pragma omp parallel for schedule(static)
        for (int r = 0; r < nb * 2 * F; r++) {
            const int j = r / (2 * F), rr = r % (2 * F);
            if (rr < F) G[j * F + rr] = row_dot(gt, gw[j] + (uint64_t)rr * grb, D, &q);
            else H[j * F + rr - F] = row_dot(ut, uw[j] + (uint64_t)(rr - F) * urb, D, &q);
        }
        qin hq[16];
        for (int j = 0; j < nb; j++) {
            float *gg = G + j * F;
            for (int i = 0; i < F; i++) gg[i] = siluf(gg[i]) * H[j * F + i];
            const int nbk = (F + QK_K - 1) / QK_K + 1;
            qin_init(&hq[j], gg, F, s->qk2 + j * nbk, s->a_qs2 + j * F, s->a_d2 + j * (F / 32 + 1), s->a_s2 + j * (F / 32 + 1));
            qin_need(&hq[j], dt);
        }
        float *O = s->eo;
        #pragma omp parallel for schedule(static)
        for (int r = 0; r < nb * D; r++) {
            const int j = r / D, rr = r % D;
            O[r] = row_dot(dt, dw[j] + (uint64_t)rr * drb, F, &hq[j]);
        }
        for (int j = 0; j < nb; j++) {
            experts_release(m->experts, l, sel[bl[j]]);
            const float wj = w[bl[j]];
            for (int i = 0; i < D; i++) out[i] += wj * O[j * D + i];
            done[bl[j]] = 1;
        }
    }
    q35_prof[3] += qnow() - t0;
}

// logits for T (1 or 2) normed hidden states. With the 4-bit copy: approximate logits for all
// rows (each row read once for all tokens), then, if `exact`, the exact logits of the top 64
// per token from the file's rows.
static void out_logits(q35_model *m, const float *const *xn, int T, float *const *lg, int exact, scratch **ss) {
    const q35_cfg *c = &m->c;
    const int D = c->n_embd;
    const qmat *O = &m->output;
    qin q[2];
    for (int t = 0; t < T; t++) qin_init(&q[t], xn[t], D, ss[t]->qk, ss[t]->a_qs, ss[t]->a_d, ss[t]->a_s);
    if (!m->out4) {
        for (int t = 0; t < T; t++) qin_need(&q[t], O->type);
        #pragma omp parallel for schedule(static)
        for (int r = 0; r < O->rows; r++)
            for (int t = 0; t < T; t++) lg[t][r] = row_dot(O->type, O->data + (uint64_t)r * O->row_bytes, O->cols, &q[t]);
        return;
    }
    // approximate logits from the 4-bit copy: w = d * (q - 8 + 0.5), x as u8 (+128) per 256
    static uint8_t xu[2][8192] __attribute__((aligned(64)));
    static float dx[2][32], xsum[2][128];
    const int nb = D / 64, rb = m->out4_rb;
    for (int t = 0; t < T; t++) {
        quantize_row_u8(xn[t], xu[t], dx[t], D);
        // sum over each 64-block of (xu - 128), for the +0.5 offset
        for (int b = 0; b < nb; b++) { int sacc = 0; for (int i = 0; i < 64; i++) sacc += xu[t][b * 64 + i] - 128; xsum[t][b] = sacc * 0.5f * dx[t][b / 4]; }
    }
    const __m512i c128 = _mm512_set1_epi8((char)128), m4 = _mm512_set1_epi8(15), c8 = _mm512_set1_epi8(8);
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < O->rows; r++) {
        const uint8_t *row = m->out4 + (size_t)r * rb;
        __m512 acc[2] = {_mm512_setzero_ps(), _mm512_setzero_ps()};
        float off[2] = {0, 0};
        for (int b = 0; b < nb; b++) {
            float d; memcpy(&d, row + b * 36, 4);
            const __m256i p = _mm256_loadu_si256((const __m256i *)(row + b * 36 + 4));
            const __m512i pp = _mm512_inserti64x4(_mm512_castsi256_si512(p), _mm256_srli_epi16(p, 4), 1);
            const __m512i qv = _mm512_sub_epi8(_mm512_and_si512(pp, m4), c8);
            const __m512i corr = _mm512_dpbusd_epi32(_mm512_setzero_si512(), c128, qv);
            for (int t = 0; t < T; t++) {
                const __m512i xv = _mm512_loadu_si512((const void *)(xu[t] + b * 64));
                const __m512i dp = _mm512_sub_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(), xv, qv), corr);
                acc[t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(dp), _mm512_set1_ps(d * dx[t][b / 4]), acc[t]);
                off[t] += d * xsum[t][b];
            }
        }
        for (int t = 0; t < T; t++) lg[t][r] = _mm512_reduce_add_ps(acc[t]) + off[t];
    }
    if (!exact) return;
    // exact logits for the top candidates, rows read from the file
    enum { NC = 64 };
    static uint8_t *rows;
    if (!rows) rows = malloc((size_t)NC * O->row_bytes);
    for (int t = 0; t < T; t++) {
        float *L = lg[t];
        int cand[NC], nc = 0;
        for (int r = 0; r < O->rows; r++) {
            if (nc < NC || L[r] > L[cand[nc - 1]]) {
                int j = nc < NC ? nc++ : NC - 1;
                while (j > 0 && L[cand[j - 1]] < L[r]) { cand[j] = cand[j - 1]; j--; }
                cand[j] = r;
            }
        }
        for (int j = 0; j < nc; j++) read_at(&m->g, rows + (size_t)j * O->row_bytes, O->row_bytes, O->file_off + (uint64_t)cand[j] * O->row_bytes);
        qin_need(&q[t], O->type);
        for (int j = 0; j < nc; j++) L[cand[j]] = row_dot(O->type, rows + (size_t)j * O->row_bytes, O->cols, &q[t]);
    }
}

float *q35_forward(q35_model *m, int token, int pos) {
    const q35_cfg *c = &m->c;
    scratch *s = scr(m);
    const int D = c->n_embd;
    float *x = s->x, *xn = s->xn, *res = s->moe;
    embed(m, token, x);
    for (int l = 0; l < c->n_layer; l++) {
        const q35_layer *L = &m->layers[l];
        rmsnorm(xn, x, L->attn_norm, D, c->eps);
        double t0 = qnow();
        if (L->is_attn) {
            const int ai = m->attn_idx[l];
            attention(m, L, kc_layer(m, ai), vc_layer(m, ai), pos, pos, pos + 1, xn, res, s);
            q35_prof[1] += qnow() - t0;
        }
        else { delta_net(m, L, m->rec_idx[l], xn, res, s); q35_prof[0] += qnow() - t0; }
        for (int i = 0; i < D; i++) x[i] += res[i];
        rmsnorm(xn, x, L->post_norm, D, c->eps);
        // next layer's router sees (approximately) this layer's post-attention input;
        // use the current normed state as the guess input for prefetch
        moe(m, l, xn, xn, res, s);
        for (int i = 0; i < D; i++) x[i] += res[i];
    }
    rmsnorm(xn, x, m->output_norm, D, c->eps);
    memcpy(m->hlast, xn, D * 4);
    double t0 = qnow();
    const float *xs[1] = {xn};
    float *ls[1] = {m->logits};
    scratch *ss[1] = {s};
    out_logits(m, xs, 1, ls, 1, ss);
    q35_prof[5] += qnow() - t0;
    return m->logits;
}


// ======================= speculative decoding: MTP draft + 2-token verify =======================

static inline void row_dot2(int type, const uint8_t *row, int n, const qin *q0, const qin *q1, float *r0, float *r1) {
    switch (type) {
    case GT_Q3_K: vec_dot2_q3_K_q8_K(n, (const block_q3_K *)row, q0->qk, q1->qk, r0, r1); return;
    case GT_Q4_K: vec_dot2_q4_K_q8_K(n, (const block_q4_K *)row, q0->qk, q1->qk, r0, r1); return;
    case GT_Q5_K: vec_dot2_q5_K_q8_K(n, (const block_q5_K *)row, q0->qk, q1->qk, r0, r1); return;
    }
    *r0 = row_dot(type, row, n, q0);
    *r1 = row_dot(type, row, n, q1);
}

typedef struct { const qmat *W; float *y0, *y1; } mv2_job;
// y0 = W x0, y1 = W x1: each weight row is read once for both tokens
static void mv_multi2(const mv2_job *jobs, int nj, qin *q0, qin *q1) {
    int start[8], total = 0;
    for (int j = 0; j < nj; j++) { qin_need(q0, jobs[j].W->type); qin_need(q1, jobs[j].W->type); start[j] = total; total += jobs[j].W->rows; }
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < total; r++) {
        int j = nj - 1;
        while (r < start[j]) j--;
        const qmat *W = jobs[j].W;
        const int rr = r - start[j];
        const uint8_t *row = W->data + (uint64_t)rr * W->row_bytes;
        row_dot2(W->type, row, W->cols, q0, q1, &jobs[j].y0[rr], &jobs[j].y1[rr]);
    }
}

// causal conv for one token (state cs: [n_conv - 1][conv_dim], oldest first), then SiLU
static void conv_step(const q35_cfg *c, const q35_layer *L, float *cs, const float *qkv, float *cv) {
    const int CD = c->conv_dim, nc = c->n_conv;
    for (int ch = 0; ch < CD; ch += 16) {
        const __m512 cur = _mm512_loadu_ps(qkv + ch);
        __m512 acc = _mm512_mul_ps(_mm512_loadu_ps(L->conv_wt + (size_t)(nc - 1) * CD + ch), cur);
        for (int k = 0; k < nc - 1; k++)
            acc = _mm512_fmadd_ps(_mm512_loadu_ps(L->conv_wt + (size_t)k * CD + ch), _mm512_loadu_ps(cs + (size_t)k * CD + ch), acc);
        for (int k = 0; k < nc - 2; k++) _mm512_storeu_ps(cs + (size_t)k * CD + ch, _mm512_loadu_ps(cs + (size_t)(k + 1) * CD + ch));
        _mm512_storeu_ps(cs + (size_t)(nc - 2) * CD + ch, cur);
        _mm512_storeu_ps(cv + ch, silu_ps(acc));
    }
    for (int h = 0; h < 2 * c->n_kh; h++) {   // L2 norm of the q and k heads
        float *p = cv + (size_t)h * c->dk;
        const float r = 1.0f / sqrtf(dotf(p, p, c->dk) + c->eps);
        for (int i = 0; i < c->dk; i++) p[i] *= r;
    }
}

static void delta_net2(q35_model *m, const q35_layer *L, int ri, const float *xn0, const float *xn1,
                       float *out0, float *out1, scratch *s0, scratch *s1) {
    const q35_cfg *c = &m->c;
    const int NV = c->n_vh, DK = c->dk, DV = c->dv, CD = c->conv_dim;
    qin q0, q1;
    qin_init(&q0, xn0, c->n_embd, s0->qk, s0->a_qs, s0->a_d, s0->a_s);
    qin_init(&q1, xn1, c->n_embd, s1->qk, s1->a_qs, s1->a_d, s1->a_s);
    double td = qnow();
    mv2_job jobs[4] = {{&L->wqkv, s0->t1, s1->t1}, {&L->wz, s0->t2, s1->t2},
                       {&L->walpha, s0->t3, s1->t3}, {&L->wbeta, s0->t3 + NV, s1->t3 + NV}};
    mv_multi2(jobs, 4, &q0, &q1);
    q35_prof[6] += qnow() - td; td = qnow();
    float *cs = m->conv + (size_t)ri * (c->n_conv - 1) * CD;
    conv_step(c, L, cs, s0->t1, s0->big);
    memcpy(m->conv_bk + (size_t)ri * (c->n_conv - 1) * CD, cs, (size_t)(c->n_conv - 1) * CD * 4);   // for rollback
    conv_step(c, L, cs, s1->t1, s1->big);
    const float *cv[2] = {s0->big, s1->big}, *ab[2] = {s0->t3, s1->t3}, *z[2] = {s0->t2, s1->t2};
    float *o[2] = {s0->t1, s1->t1};
    float *S = m->sp[ri], *S2 = m->sp_alt[ri];
    const float scale = 1.0f / sqrtf((float)DV);
    const int NR = DK / 16;
    #pragma omp parallel for schedule(static)
    for (int h = 0; h < NV; h++) {
        float g[2], beta[2];
        const float *kh[2], *qh[2], *vh[2];
        for (int t = 0; t < 2; t++) {
            g[t] = expf(softplusf(ab[t][h] + L->dt_bias[h]) * L->ssm_a[h]);
            beta[t] = sigmoidf(ab[t][NV + h]);
            kh[t] = cv[t] + c->key_dim + (size_t)(h % c->n_kh) * DK;
            qh[t] = cv[t] + (size_t)(h % c->n_kh) * DK;
            vh[t] = cv[t] + 2 * c->key_dim + (size_t)h * DV;
        }
        float *M = S + (size_t)h * DV * DK, *M2 = S2 + (size_t)h * DV * DK;
        for (int j = 0; j < DV; j++) {
            __m512 r[16];
            float *row = M + (size_t)j * DK;
            for (int i = 0; i < NR; i++) r[i] = _mm512_loadu_ps(row + 16 * i);
            for (int t = 0; t < 2; t++) {
                const __m512 gv = _mm512_set1_ps(g[t]);
                __m512 acc = _mm512_setzero_ps();
                for (int i = 0; i < NR; i++) { r[i] = _mm512_mul_ps(r[i], gv); acc = _mm512_fmadd_ps(r[i], _mm512_loadu_ps(kh[t] + 16 * i), acc); }
                const __m512 dv = _mm512_set1_ps((vh[t][j] - _mm512_reduce_add_ps(acc)) * beta[t]);
                __m512 oa = _mm512_setzero_ps();
                for (int i = 0; i < NR; i++) {
                    r[i] = _mm512_fmadd_ps(dv, _mm512_loadu_ps(kh[t] + 16 * i), r[i]);
                    oa = _mm512_fmadd_ps(r[i], _mm512_loadu_ps(qh[t] + 16 * i), oa);
                }
                float *dst = t == 0 ? row : M2 + (size_t)j * DK;   // state after token 0 in place, after token 1 in the alternate buffer
                for (int i = 0; i < NR; i++) _mm512_storeu_ps(dst + 16 * i, r[i]);
                o[t][(size_t)h * DV + j] = _mm512_reduce_add_ps(oa) * scale;
            }
        }
        for (int t = 0; t < 2; t++) {
            float *oh = o[t] + (size_t)h * DV;
            const float *zh = z[t] + (size_t)h * DV;
            const __m512 rv = _mm512_set1_ps(1.0f / sqrtf(dotf(oh, oh, DV) / DV + c->eps));
            for (int j = 0; j < DV; j += 16)
                _mm512_storeu_ps(oh + j, _mm512_mul_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_loadu_ps(oh + j), rv),
                                 _mm512_loadu_ps(L->ssm_norm + j)), silu_ps(_mm512_loadu_ps(zh + j))));
        }
    }
    q35_prof[7] += qnow() - td; td = qnow();
    qin p0, p1;
    qin_init(&p0, o[0], c->value_dim, s0->qk2, s0->a_qs2, s0->a_d2, s0->a_s2);
    qin_init(&p1, o[1], c->value_dim, s1->qk2, s1->a_qs2, s1->a_d2, s1->a_s2);
    mv2_job j2 = {&L->wout, out0, out1};
    mv_multi2(&j2, 1, &p0, &p1);
    q35_prof[6] += qnow() - td;
}

static void attention2(q35_model *m, const q35_layer *L, int ai, const float *xn0, const float *xn1, int pos,
                       float *out0, float *out1, scratch *s0, scratch *s1) {
    const q35_cfg *c = &m->c;
    const int kvd = c->n_head_kv * c->hd;
    qin q0, q1;
    qin_init(&q0, xn0, c->n_embd, s0->qk, s0->a_qs, s0->a_d, s0->a_s);
    qin_init(&q1, xn1, c->n_embd, s1->qk, s1->a_qs, s1->a_d, s1->a_s);
    mv2_job jobs[3] = {{&L->wq, s0->t1, s1->t1}, {&L->wk, s0->t2, s1->t2}, {&L->wv, s0->t3, s1->t3}};
    mv_multi2(jobs, 3, &q0, &q1);
    uint16_t *kc = kc_layer(m, ai), *vc = vc_layer(m, ai);
    scratch *ss[2] = {s0, s1};
    for (int t = 0; t < 2; t++) attn_prep(m, L, ss[t]->t1, ss[t]->t2, ss[t]->t3, ss[t]->big, pos + t, kc, vc, pos + t);
    for (int t = 0; t < 2; t++) attn_core(m, kc, vc, pos + t + 1, ss[t]->big, ss[t]->t1, ss[t]->t2 + kvd, s0, attn_lo(&m->c, L, pos + t + 1));
    qin p0, p1;
    qin_init(&p0, s0->t2 + kvd, c->n_head * c->hd, s0->qk2, s0->a_qs2, s0->a_d2, s0->a_s2);
    qin_init(&p1, s1->t2 + kvd, c->n_head * c->hd, s1->qk2, s1->a_qs2, s1->a_d2, s1->a_s2);
    mv2_job j2 = {&L->wo, out0, out1};
    mv_multi2(&j2, 1, &p0, &p1);
}

// routed experts (union of both tokens' choices) + shared expert, for two tokens
static void moe2(q35_model *m, int l, const float *xn0, const float *xn1, float *out0, float *out1, scratch *s0, scratch *s1) {
    const q35_cfg *c = &m->c;
    const q35_layer *L = &m->layers[l];
    const int K = c->n_used, F = c->n_ff, D = c->n_embd;
    int sel[2][16], U[32], nu = 0;
    float w[2][16], uw[32][2];
    double t0 = qnow();
    route(L, xn0, c, K, sel[0], w[0]);
    route(L, xn1, c, K, sel[1], w[1]);
    for (int t = 0; t < 2; t++)
        for (int j = 0; j < K; j++) {
            int u = 0;
            while (u < nu && U[u] != sel[t][j]) u++;
            if (u == nu) { U[nu] = sel[t][j]; uw[nu][0] = uw[nu][1] = 0; nu++; }
            uw[u][t] = w[t][j];
        }
    if (m->prefetch_on && l > 0 && l < c->n_layer) {
        for (int j = 0; j < K; j++)
            for (int q = 0; q < (m->pred_n ? m->pred_n : K); q++) m->pred_correct += m->pred[l * 16 + q] == sel[0][j];
        m->pred_total += K;
    }
    experts_begin_layer(m->experts, l, U, nu);
    if (m->prefetch_on && l + 1 < c->n_layer) {   // next layer: union of both tokens' guesses
        int p[2][16], pu[16], npu = 0;
        float pw[16];
        route(&m->layers[l + 1], xn0, c, K, p[0], pw);
        route(&m->layers[l + 1], xn1, c, K, p[1], pw);
        for (int j = 0; j < K; j++)
            for (int t = 0; t < 2; t++) {   // interleaved: most likely first
                int u = 0;
                while (u < npu && pu[u] != p[t][j]) u++;
                if (u == npu && npu < 16) pu[npu++] = p[t][j];
            }
        memcpy(m->pred + (l + 1) * 16, pu, sizeof pu);
        m->pred_n = npu;
        experts_prefetch(m->experts, l + 1, pu, npu);
    }
    q35_prof[2] += qnow() - t0;

    t0 = qnow();
    qin q0, q1;
    qin_init(&q0, xn0, D, s0->qk, s0->a_qs, s0->a_d, s0->a_s);
    qin_init(&q1, xn1, D, s1->qk, s1->a_qs, s1->a_d, s1->a_s);
    mv2_job jobs[2] = {{&L->sh_gate, s0->t1, s1->t1}, {&L->sh_up, s0->t2, s1->t2}};
    mv_multi2(jobs, 2, &q0, &q1);
    silu_mul(s0->t1, s0->t2, c->n_ff_sh);
    silu_mul(s1->t1, s1->t2, c->n_ff_sh);
    qin p0, p1;
    qin_init(&p0, s0->t1, c->n_ff_sh, s0->qk2, s0->a_qs2, s0->a_d2, s0->a_s2);
    qin_init(&p1, s1->t1, c->n_ff_sh, s1->qk2, s1->a_qs2, s1->a_d2, s1->a_s2);
    mv2_job jd = {&L->sh_down, out0, out1};
    mv_multi2(&jd, 1, &p0, &p1);
    const float g0 = sigmoidf(dotf(L->shexp_gate, xn0, D)), g1 = sigmoidf(dotf(L->shexp_gate, xn1, D));
    for (int i = 0; i < D; i++) { out0[i] *= g0; out1[i] *= g1; }
    q35_prof[4] += qnow() - t0;

    t0 = qnow();
    qin *qq[2] = {&q0, &q1};
    for (int t = 0; t < 2; t++) { qin_need(qq[t], L->gate_exps.type); qin_need(qq[t], L->up_exps.type); }
    const int gt = L->gate_exps.type, ut = L->up_exps.type, dt = L->down_exps.type;
    const uint64_t grb = L->gate_exps.row_bytes, urb = L->up_exps.row_bytes, drb = L->down_exps.row_bytes;
    float *outs[2] = {out0, out1};
    int done[32] = {0};
    for (int n_done = 0; n_done < nu;) {
        int bl[32], nb = 0;
        for (int k = 0; k < nu; k++) if (!done[k] && experts_ready(m->experts, l, U[k])) bl[nb++] = k;
        if (!nb) {
            int rest[32], nr = 0;
            for (int k = 0; k < nu; k++) if (!done[k]) rest[nr++] = U[k];
            experts_wait_any(m->experts, l, rest, nr);
            continue;
        }
        n_done += nb;
        // (expert, token) pairs
        int pe[32], pt[32], np = 0;
        const uint8_t *gw[32], *uwp[32], *dw[32];
        for (int j = 0; j < nb; j++) {
            const uint8_t *a, *b, *d;
            experts_get(m->experts, l, U[bl[j]], &a, &b, &d);
            for (int t = 0; t < 2; t++)
                if (uw[bl[j]][t] > 0 && np < 16) { pe[np] = bl[j]; pt[np] = t; gw[np] = a; uwp[np] = b; dw[np] = d; np++; }
        }
        // entries: one per expert, with its pair index for each token (-1 if unused); an expert
        // used by both tokens is unpacked once per row (row_dot2)
        int ep[16][2], ne = 0;
        for (int j = 0; j < np; j++) {
            if (ne && pe[ep[ne - 1][0] >= 0 ? ep[ne - 1][0] : ep[ne - 1][1]] == pe[j]) ep[ne - 1][pt[j]] = j;
            else { ep[ne][0] = ep[ne][1] = -1; ep[ne][pt[j]] = j; ne++; }
        }
        float *G = s0->eg, *H = s0->eh;
        #pragma omp parallel for schedule(static)
        for (int r = 0; r < ne * 2 * F; r++) {
            const int e = r / (2 * F), rr = r % (2 * F), a = ep[e][0], b = ep[e][1], j = a >= 0 ? a : b;
            const int up = rr >= F, ri = up ? rr - F : rr, ty = up ? ut : gt;
            const uint8_t *row = (up ? uwp[j] + (uint64_t)ri * urb : gw[j] + (uint64_t)ri * grb);
            float *Y = up ? H : G;
            if (a >= 0 && b >= 0) row_dot2(ty, row, D, qq[0], qq[1], &Y[a * F + ri], &Y[b * F + ri]);
            else Y[j * F + ri] = row_dot(ty, row, D, qq[pt[j]]);
        }
        qin hq[16];
        for (int j = 0; j < np; j++) {
            float *gg = G + j * F;
            silu_mul(gg, H + j * F, F);
            const int nbk = (F + QK_K - 1) / QK_K + 1;
            qin_init(&hq[j], gg, F, s0->qk2 + j * nbk, s0->a_qs2 + j * F, s0->a_d2 + j * (F / 32 + 1), s0->a_s2 + j * (F / 32 + 1));
            qin_need(&hq[j], dt);
        }
        float *O = s0->eo;
        #pragma omp parallel for schedule(static)
        for (int r = 0; r < ne * D; r++) {
            const int e = r / D, rr = r % D, a = ep[e][0], b = ep[e][1], j = a >= 0 ? a : b;
            const uint8_t *row = dw[j] + (uint64_t)rr * drb;
            if (a >= 0 && b >= 0) row_dot2(dt, row, F, &hq[a], &hq[b], &O[a * D + rr], &O[b * D + rr]);
            else O[j * D + rr] = row_dot(dt, row, F, &hq[j]);
        }
        for (int j = 0; j < np; j++) {
            const float wj = uw[pe[j]][pt[j]];
            float *o = outs[pt[j]];
            for (int i = 0; i < D; i++) o[i] += wj * O[j * D + i];
        }
        for (int j = 0; j < nb; j++) { experts_release(m->experts, l, U[bl[j]]); done[bl[j]] = 1; }
    }
    q35_prof[3] += qnow() - t0;
}

void q35_forward2(q35_model *m, int tok0, int tok1, int pos) {
    const q35_cfg *c = &m->c;
    scratch *s0 = scr(m), *s1 = scr1(m);
    const int D = c->n_embd;
    float *x0 = s0->x, *x1 = s1->x, *n0 = s0->xn, *n1 = s1->xn, *r0 = s0->moe, *r1 = s1->moe;
    embed(m, tok0, x0);
    embed(m, tok1, x1);
    for (int l = 0; l < c->n_layer; l++) {
        const q35_layer *L = &m->layers[l];
        rmsnorm(n0, x0, L->attn_norm, D, c->eps);
        rmsnorm(n1, x1, L->attn_norm, D, c->eps);
        double t0 = qnow();
        if (L->is_attn) { attention2(m, L, m->attn_idx[l], n0, n1, pos, r0, r1, s0, s1); q35_prof[1] += qnow() - t0; }
        else { delta_net2(m, L, m->rec_idx[l], n0, n1, r0, r1, s0, s1); q35_prof[0] += qnow() - t0; }
        for (int i = 0; i < D; i++) { x0[i] += r0[i]; x1[i] += r1[i]; }
        rmsnorm(n0, x0, L->post_norm, D, c->eps);
        rmsnorm(n1, x1, L->post_norm, D, c->eps);
        moe2(m, l, n0, n1, r0, r1, s0, s1);
        for (int i = 0; i < D; i++) { x0[i] += r0[i]; x1[i] += r1[i]; }
    }
    rmsnorm(m->h2[0], x0, m->output_norm, D, c->eps);
    rmsnorm(m->h2[1], x1, m->output_norm, D, c->eps);
    double t0 = qnow();
    const float *xs[2] = {m->h2[0], m->h2[1]};
    scratch *ss[2] = {s0, s1};
    out_logits(m, xs, 2, m->logits2, 1, ss);
    q35_prof[5] += qnow() - t0;
}

void q35_commit(q35_model *m, int keep_both) {
    const q35_cfg *c = &m->c;
    m->spec_steps++;
    if (keep_both) {
        m->spec_accepted++;
        for (int r = 0; r < m->n_rec; r++) { float *t = m->sp[r]; m->sp[r] = m->sp_alt[r]; m->sp_alt[r] = t; }
        memcpy(m->hlast, m->h2[1], c->n_embd * 4);
    } else {
        memcpy(m->conv, m->conv_bk, (size_t)m->n_rec * (c->n_conv - 1) * c->conv_dim * 4);
        memcpy(m->hlast, m->h2[0], c->n_embd * 4);
    }
}

// MTP: x = eh_proj([enorm(embed(tok)); hnorm(hlast)]), one attention + MoE block, then the LM head
int q35_draft(q35_model *m, int tok, int pos) {
    const q35_cfg *c = &m->c;
    scratch *s = scr(m);
    const int D = c->n_embd;
    const q35_layer *L = &m->layers[c->n_layer];
    float *cat = s->big, *x = s->x, *xn = s->xn, *res = s->moe;
    double t0 = qnow();
    embed(m, tok, s->t3);
    rmsnorm(cat, s->t3, m->mtp_enorm, D, c->eps);
    rmsnorm(cat + D, m->hlast, m->mtp_hnorm, D, c->eps);
    qin q;
    qin_init(&q, cat, 2 * D, s->qk, s->a_qs, s->a_d, s->a_s);
    mv_job j = {&m->eh_proj, x};
    mv_multi(&j, 1, &q);
    if (m->mtp_n >= c->n_ctx) m->mtp_n = 0;
    rmsnorm(xn, x, L->attn_norm, D, c->eps);
    attention(m, L, m->mtp_kc, m->mtp_vc, m->mtp_n, pos, m->mtp_n + 1, xn, res, s);
    m->mtp_n++;
    for (int i = 0; i < D; i++) x[i] += res[i];
    rmsnorm(xn, x, L->post_norm, D, c->eps);
    {   // QUARRY_MTP_K: experts per token in the draft layer (its Q8_0 experts are big; fewer
        // costs a little acceptance, never correctness)
        static int mk = -1;
        if (mk < 0) mk = getenv("QUARRY_MTP_K") ? atoi(getenv("QUARRY_MTP_K")) : 0;
        q35_cfg *cw = &m->c;
        const int keep = cw->n_used;
        if (mk > 0 && mk < keep) cw->n_used = mk;
        moe(m, c->n_layer, xn, NULL, res, s);
        cw->n_used = keep;
    }
    for (int i = 0; i < D; i++) x[i] += res[i];
    rmsnorm(xn, x, m->mtp_head_norm, D, c->eps);
    const float *xs[1] = {xn};
    float *ls[1] = {m->logits2[1]};
    scratch *ss[1] = {s};
    out_logits(m, xs, 1, ls, 0, ss);
    int best = 0;
    for (int i = 1; i < c->n_vocab; i++) if (ls[0][i] > ls[0][best]) best = i;
    m->draft_seconds += qnow() - t0;
    return best;
}

// ======================= batched prompt processing =======================
//
// A chunk of T tokens goes through each layer together. Every weight row is dequantized
// to f32 once per chunk and multiplied with all T activation rows (AVX-512 FMA), so the
// weights are read once per chunk instead of once per token. Experts are grouped by the
// tokens that picked them. The delta-net recurrence runs per head over the chunk, with
// the 64KB head state kept in cache.

#define Q35_PF_T 1024
#define MM_RB 16      // rows dequantized together

// Y[t * ldy + r] = W_r . X[t * ldx], rows [0, rows), tokens [0, T); W from `base`
static void mm_f32(int type, const uint8_t *base, uint64_t rb, int rows, int cols,
                   const float *X, int ldx, int T, float *Y, int ldy) {
    const int nblk = (rows + MM_RB - 1) / MM_RB;
    static float *wbuf[64];
    static int wcap[64];
    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        if (wcap[tid] < MM_RB * cols) { free(wbuf[tid]); wbuf[tid] = aligned_alloc(64, (size_t)MM_RB * cols * 4); wcap[tid] = MM_RB * cols; }
        float *Wf = wbuf[tid];
        #pragma omp for schedule(dynamic, 1)
        for (int bk = 0; bk < nblk; bk++) {
            const int r0 = bk * MM_RB, nr = rows - r0 < MM_RB ? rows - r0 : MM_RB;
            for (int r = 0; r < nr; r++) dequant_row(type, base + (uint64_t)(r0 + r) * rb, Wf + (size_t)r * cols, cols);
            // 4 rows x 4 tokens per inner kernel
            for (int t0 = 0; t0 < T; t0 += 4) {
                const int nt = T - t0 < 4 ? T - t0 : 4;
                for (int rr = 0; rr < nr; rr += 4) {
                    const int nrr = nr - rr < 4 ? nr - rr : 4;
                    __m512 acc[4][4];
                    for (int a = 0; a < 4; a++) for (int b = 0; b < 4; b++) acc[a][b] = _mm512_setzero_ps();
                    const float *w0 = Wf + (size_t)rr * cols;
                    const float *x0 = X + (size_t)t0 * ldx;
                    if (nrr == 4 && nt == 4 && cols % 16 == 0) {
                        for (int k = 0; k < cols; k += 16) {
                            const __m512 wa = _mm512_load_ps(w0 + k), wb = _mm512_load_ps(w0 + cols + k);
                            const __m512 wc = _mm512_load_ps(w0 + 2 * cols + k), wd = _mm512_load_ps(w0 + 3 * cols + k);
                            for (int b = 0; b < 4; b++) {
                                const __m512 xv = _mm512_loadu_ps(x0 + (size_t)b * ldx + k);
                                acc[0][b] = _mm512_fmadd_ps(wa, xv, acc[0][b]);
                                acc[1][b] = _mm512_fmadd_ps(wb, xv, acc[1][b]);
                                acc[2][b] = _mm512_fmadd_ps(wc, xv, acc[2][b]);
                                acc[3][b] = _mm512_fmadd_ps(wd, xv, acc[3][b]);
                            }
                        }
                        for (int a = 0; a < 4; a++)
                            for (int b = 0; b < 4; b++) Y[(size_t)(t0 + b) * ldy + r0 + rr + a] = _mm512_reduce_add_ps(acc[a][b]);
                    } else {
                        for (int a = 0; a < nrr; a++)
                            for (int b = 0; b < nt; b++)
                                Y[(size_t)(t0 + b) * ldy + r0 + rr + a] = dotf(w0 + (size_t)a * cols, x0 + (size_t)b * ldx, cols);
                    }
                }
            }
        }
    }
}

// any K-quant row -> int8 per 256 values (dequant, then symmetric int8). For Q4_K/Q5_K/Q6_K
// the extra rounding error (~amax/254 per value) is far below their own quantization error.
static void requant_row_s8(int type, const uint8_t *row, int cols, q3k_s8 *out, float *tmp) {
    dequant_row(type, row, tmp, cols);
    const __m512i ones = _mm512_set1_epi8(1);
    for (int b = 0; b < cols / QK_K; b++) {
        const float *v = tmp + b * QK_K;
        __m512 mx = _mm512_setzero_ps();
        for (int i = 0; i < QK_K; i += 16) mx = _mm512_max_ps(mx, _mm512_abs_ps(_mm512_loadu_ps(v + i)));
        const float amax = _mm512_reduce_max_ps(mx), id = amax > 0 ? 127.f / amax : 0.f;
        const __m512 vid = _mm512_set1_ps(id);
        for (int i = 0; i < QK_K; i += 16) {
            __m512i q = _mm512_cvtps_epi32(_mm512_roundscale_ps(_mm512_mul_ps(_mm512_loadu_ps(v + i), vid), _MM_FROUND_TO_NEAREST_INT));
            _mm_storeu_si128((__m128i *)(out[b].w + i), _mm512_cvtsepi32_epi8(q));
        }
        __m512i ws = _mm512_setzero_si512();
        for (int k = 0; k < QK_K; k += 64) ws = _mm512_dpbusd_epi32(ws, ones, _mm512_loadu_si512((const void *)(out[b].w + k)));
        out[b].d = amax / 127.f;
        out[b].corr = -128 * _mm512_reduce_add_epi32(ws);
    }
}

// Q3_K rows: repack once to int8 and run all tokens with VNNI dpbusd (activations as u8)
static int getenv_k2x4_off = -1;
static inline void dot2x4_s8(int n, const q3k_s8 *w0, const q3k_s8 *w1, const uint8_t *xu, const float *dx, float *out, int ldy);
static void mm_q3k_s8(int type, const uint8_t *base, uint64_t rb, int rows, int cols,
                      const float *X, int ldx, int T, float *Y, int ldy) {
    static uint8_t *xu; static float *dx; static size_t xcap;
    if (getenv_k2x4_off < 0) getenv_k2x4_off = getenv("QUARRY_K1") != NULL;
    const int nbk = cols / QK_K;
    if ((size_t)T * cols > xcap) {
        free(xu); free(dx);
        xcap = (size_t)T * cols;
        xu = aligned_alloc(64, (xcap + 63) / 64 * 64); dx = aligned_alloc(64, ((xcap / QK_K) * 4 + 63) / 64 * 64);
    }
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < T; t++) quantize_row_u8(X + (size_t)t * ldx, xu + (size_t)t * cols, dx + (size_t)t * nbk, cols);
    const int nblk = (rows + MM_RB - 1) / MM_RB;
    static q3k_s8 *pbuf[64];
    static float *tbuf[64];
    static int pcap[64];
    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        if (pcap[tid] < MM_RB * nbk) {
            free(pbuf[tid]); free(tbuf[tid]);
            pbuf[tid] = aligned_alloc(64, ((size_t)MM_RB * nbk * sizeof(q3k_s8) + 63) / 64 * 64);
            tbuf[tid] = aligned_alloc(64, ((size_t)cols * 4 + 63) / 64 * 64);
            pcap[tid] = MM_RB * nbk;
        }
        q3k_s8 *pk = pbuf[tid];
        #pragma omp for schedule(dynamic, 1)
        for (int bk = 0; bk < nblk; bk++) {
            const int r0 = bk * MM_RB, nr = rows - r0 < MM_RB ? rows - r0 : MM_RB;
            const double tr = tid == 0 ? qnow() : 0;
            for (int r = 0; r < nr; r++) {
                const uint8_t *row = base + (uint64_t)(r0 + r) * rb;
                if (type == GT_Q3_K) q3k_repack_row((const block_q3_K *)row, cols, pk + r * nbk);
                else requant_row_s8(type, row, cols, pk + r * nbk, tbuf[tid]);
            }
            if (tid == 0) q35_mp[type == GT_Q3_K ? 5 : 6] += qnow() - tr;
            for (int t = 0; t < T; t += 4) {
                const int nt = T - t < 4 ? T - t : 4;
                int r = 0;
                if (nt == 4 && !getenv_k2x4_off)
                    for (; r + 2 <= nr; r += 2)
                        dot2x4_s8(cols, pk + r * nbk, pk + (r + 1) * nbk, xu + (size_t)t * cols, dx + (size_t)t * nbk,
                                  Y + (size_t)t * ldy + r0 + r, ldy);
                for (; r < nr; r++)
                    dot_q3k_s8_multi(cols, pk + r * nbk, nt, xu + (size_t)t * cols, dx + (size_t)t * nbk,
                                     Y + (size_t)t * ldy + r0 + r, ldy);
            }
        }
    }
}

// two int8 rows x four tokens: out[t * ldy + {0,1}]
static inline void dot2x4_s8(int n, const q3k_s8 *w0, const q3k_s8 *w1, const uint8_t *xu, const float *dx,
                             float *out, int ldy) {
    const int nb = n / QK_K;
    __m512 f[2][4];
    for (int a = 0; a < 2; a++) for (int t = 0; t < 4; t++) f[a][t] = _mm512_setzero_ps();
    const uint8_t *x0 = xu, *x1 = xu + n, *x2 = x1 + n, *x3 = x2 + n;
    for (int b = 0; b < nb; b++) {
        const __m512i c0 = _mm512_zextsi128_si512(_mm_cvtsi32_si128(w0[b].corr));
        const __m512i c1 = _mm512_zextsi128_si512(_mm_cvtsi32_si128(w1[b].corr));
        __m512i a00 = c0, a01 = c0, a02 = c0, a03 = c0, a10 = c1, a11 = c1, a12 = c1, a13 = c1;
        for (int k = 0; k < QK_K; k += 64) {
            const __m512i wa = _mm512_loadu_si512((const void *)(w0[b].w + k));
            const __m512i wb = _mm512_loadu_si512((const void *)(w1[b].w + k));
            const int o = b * QK_K + k;
            __m512i xv = _mm512_loadu_si512((const void *)(x0 + o));
            a00 = _mm512_dpbusd_epi32(a00, xv, wa); a10 = _mm512_dpbusd_epi32(a10, xv, wb);
            xv = _mm512_loadu_si512((const void *)(x1 + o));
            a01 = _mm512_dpbusd_epi32(a01, xv, wa); a11 = _mm512_dpbusd_epi32(a11, xv, wb);
            xv = _mm512_loadu_si512((const void *)(x2 + o));
            a02 = _mm512_dpbusd_epi32(a02, xv, wa); a12 = _mm512_dpbusd_epi32(a12, xv, wb);
            xv = _mm512_loadu_si512((const void *)(x3 + o));
            a03 = _mm512_dpbusd_epi32(a03, xv, wa); a13 = _mm512_dpbusd_epi32(a13, xv, wb);
        }
        const float d0 = w0[b].d, d1 = w1[b].d;
        const float e0 = dx[b], e1 = dx[nb + b], e2 = dx[2 * nb + b], e3 = dx[3 * nb + b];
        f[0][0] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(a00), _mm512_set1_ps(d0 * e0), f[0][0]);
        f[0][1] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(a01), _mm512_set1_ps(d0 * e1), f[0][1]);
        f[0][2] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(a02), _mm512_set1_ps(d0 * e2), f[0][2]);
        f[0][3] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(a03), _mm512_set1_ps(d0 * e3), f[0][3]);
        f[1][0] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(a10), _mm512_set1_ps(d1 * e0), f[1][0]);
        f[1][1] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(a11), _mm512_set1_ps(d1 * e1), f[1][1]);
        f[1][2] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(a12), _mm512_set1_ps(d1 * e2), f[1][2]);
        f[1][3] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(a13), _mm512_set1_ps(d1 * e3), f[1][3]);
    }
    for (int t = 0; t < 4; t++) {
        out[(size_t)t * ldy] = _mm512_reduce_add_ps(f[0][t]);
        out[(size_t)t * ldy + 1] = _mm512_reduce_add_ps(f[1][t]);
    }
}

static void mm_any(int type, const uint8_t *base, uint64_t rb, int rows, int cols,
                   const float *X, int ldx, int T, float *Y, int ldy) {
    const int k = type == GT_Q3_K || type == GT_Q4_K || type == GT_Q5_K || type == GT_Q6_K ||
                  type == GT_Q4_0 || type == GT_Q4_1 || type == GT_Q8_0;
    if (k && cols % QK_K == 0 && !getenv("QUARRY_PF_F32")) mm_q3k_s8(type, base, rb, rows, cols, X, ldx, T, Y, ldy);
    else mm_f32(type, base, rb, rows, cols, X, ldx, T, Y, ldy);
}
static void mm(const qmat *W, const float *X, int ldx, int T, float *Y, int ldy) {
    mm_any(W->type, W->data, W->row_bytes, W->rows, W->cols, X, ldx, T, Y, ldy);
}

typedef struct {
    int cap;
    float *X, *XN, *A, *B, *C, *Z, *AB, *O, *R;
    float *G, *H, *EO;                 // per expert: gathered tokens' gate/up/out
    float *GX;                         // gathered inputs [T][D]
    int *sel; float *wsel;             // [T][K]
    int *cnt, *off, *tok; float *tokw; // tokens grouped by expert
} q35_pfbuf;

static q35_pfbuf *pf_get(q35_model *m) {
    static q35_pfbuf b;
    if (b.cap) return &b;
    const q35_cfg *c = &m->c;
    const int T = Q35_PF_T, D = c->n_embd;
    int mx = c->conv_dim;
    if (c->n_head * c->hd * 2 > mx) mx = c->n_head * c->hd * 2;
    b.cap = T;
    b.X = xalloc((size_t)T * D * 4); b.XN = xalloc((size_t)T * D * 4); b.R = xalloc((size_t)T * D * 4);
    b.A = xalloc((size_t)T * mx * 4); b.B = xalloc((size_t)T * mx * 4); b.C = xalloc((size_t)T * mx * 4);
    b.Z = xalloc((size_t)T * c->value_dim * 4); b.AB = xalloc((size_t)T * 2 * c->n_vh * 4);
    b.O = xalloc((size_t)T * mx * 4);
    const int F = c->n_ff > c->n_ff_sh ? c->n_ff : c->n_ff_sh;
    b.G = xalloc((size_t)T * F * 4); b.H = xalloc((size_t)T * F * 4); b.EO = xalloc((size_t)T * D * 4);
    b.GX = xalloc((size_t)T * D * 4);
    b.sel = xalloc((size_t)T * 16 * 4); b.wsel = xalloc((size_t)T * 16 * 4);
    b.cnt = xalloc((c->n_expert + 1) * 4); b.off = xalloc((c->n_expert + 1) * 4);
    b.tok = xalloc((size_t)T * 16 * 4); b.tokw = xalloc((size_t)T * 16 * 4);
    return &b;
}

static void pf_delta_net(q35_model *m, const q35_layer *L, int ri, q35_pfbuf *b, int T) {
    const q35_cfg *c = &m->c;
    const int D = c->n_embd, CD = c->conv_dim, VD = c->value_dim, NV = c->n_vh, DK = c->dk;
    float *QKV = b->A, *CV = b->B, *O = b->O;
    double tp = qnow();
    mm(&L->wqkv, b->XN, D, T, QKV, CD);
    mm(&L->wz, b->XN, D, T, b->Z, VD);
    mm(&L->walpha, b->XN, D, T, b->AB, 2 * NV);
    mm(&L->wbeta, b->XN, D, T, b->AB + NV, 2 * NV);
    q35_prof[6] += qnow() - tp; tp = qnow();
    // causal conv per channel across the chunk (state = previous nc-1 inputs), then SiLU
    float *cs = m->conv + (size_t)ri * (c->n_conv - 1) * CD;
    const int nc = c->n_conv;
    #pragma omp parallel for schedule(static)
    for (int ch = 0; ch < CD; ch += 16) {
        __m512 w[8], hist[8];
        for (int k = 0; k < nc; k++) w[k] = _mm512_loadu_ps(L->conv_wt + (size_t)k * CD + ch);
        for (int k = 0; k < nc - 1; k++) hist[k] = _mm512_loadu_ps(cs + (size_t)k * CD + ch);
        for (int t = 0; t < T; t++) {
            const __m512 cur = _mm512_loadu_ps(QKV + (size_t)t * CD + ch);
            __m512 acc = _mm512_mul_ps(w[nc - 1], cur);
            for (int k = 0; k < nc - 1; k++) acc = _mm512_fmadd_ps(w[k], hist[k], acc);
            for (int k = 0; k < nc - 2; k++) hist[k] = hist[k + 1];
            hist[nc - 2] = cur;
            _mm512_storeu_ps(CV + (size_t)t * CD + ch, silu_ps(acc));
        }
        for (int k = 0; k < nc - 1; k++) _mm512_storeu_ps(cs + (size_t)k * CD + ch, hist[k]);
    }
    // L2 norm of q and k heads
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < T; t++)
        for (int h = 0; h < 2 * c->n_kh; h++) {
            float *p = CV + (size_t)t * CD + (size_t)h * DK;
            float ss = 0;
            for (int i = 0; i < DK; i++) ss += p[i] * p[i];
            const float r = 1.0f / sqrtf(ss + c->eps);
            for (int i = 0; i < DK; i++) p[i] *= r;
        }
    q35_prof[7] += qnow() - tp; tp = qnow();
    // recurrence: each head runs through the whole chunk with its state in cache
    float *S = m->sp[ri];
    const float scale = 1.0f / sqrtf((float)c->dv);
    #pragma omp parallel for schedule(dynamic, 1)
    for (int h = 0; h < NV; h++) {
        float *M = S + (size_t)h * c->dv * DK;
        for (int t = 0; t < T; t++) {
            const float *ab = b->AB + (size_t)t * 2 * NV;
            const float g = expf(softplusf(ab[h] + L->dt_bias[h]) * L->ssm_a[h]);
            const float beta = sigmoidf(ab[NV + h]);
            const float *row_t = CV + (size_t)t * CD;
            const float *kh = row_t + c->key_dim + (size_t)(h % c->n_kh) * DK, *qh = row_t + (size_t)(h % c->n_kh) * DK;
            const float *vh = row_t + 2 * c->key_dim + (size_t)h * c->dv;
            float *oh = O + (size_t)t * VD + (size_t)h * c->dv;
            const __m512 gv = _mm512_set1_ps(g);
            for (int j = 0; j < c->dv; j++) {
                float *row = M + (size_t)j * DK;
                __m512 acc = _mm512_setzero_ps();
                for (int i = 0; i < DK; i += 16) {
                    __m512 r = _mm512_mul_ps(_mm512_loadu_ps(row + i), gv);
                    _mm512_storeu_ps(row + i, r);
                    acc = _mm512_fmadd_ps(r, _mm512_loadu_ps(kh + i), acc);
                }
                const __m512 dv = _mm512_set1_ps((vh[j] - _mm512_reduce_add_ps(acc)) * beta);
                __m512 oa = _mm512_setzero_ps();
                for (int i = 0; i < DK; i += 16) {
                    __m512 r = _mm512_fmadd_ps(dv, _mm512_loadu_ps(kh + i), _mm512_loadu_ps(row + i));
                    _mm512_storeu_ps(row + i, r);
                    oa = _mm512_fmadd_ps(r, _mm512_loadu_ps(qh + i), oa);
                }
                oh[j] = _mm512_reduce_add_ps(oa) * scale;
            }
            const float *zh = b->Z + (size_t)t * VD + (size_t)h * c->dv;
            const float r = 1.0f / sqrtf(dotf(oh, oh, c->dv) / c->dv + c->eps);
            const __m512 rv = _mm512_set1_ps(r);
            for (int j = 0; j < c->dv; j += 16)
                _mm512_storeu_ps(oh + j, _mm512_mul_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_loadu_ps(oh + j), rv),
                                 _mm512_loadu_ps(L->ssm_norm + j)), silu_ps(_mm512_loadu_ps(zh + j))));
        }
    }
    q35_prof[4] += qnow() - tp; tp = qnow();
    mm(&L->wout, O, VD, T, b->R, D);
    q35_prof[6] += qnow() - tp;
}

static void pf_attention(q35_model *m, const q35_layer *L, int ai, q35_pfbuf *b, int T, int pos0) {
    const q35_cfg *c = &m->c;
    const int D = c->n_embd, HD = c->hd, NH = c->n_head, NKV = c->n_head_kv, kvd = NKV * HD, gqa = NH / NKV;
    const int QS = c->gated ? 2 * HD : HD, QD = NH * QS;
    float *QG = b->A, *K = b->B, *V = b->C, *AO = b->O;
    mm(&L->wq, b->XN, D, T, QG, QD);
    mm(&L->wk, b->XN, D, T, K, kvd);
    mm(&L->wv, b->XN, D, T, V, kvd);
    uint16_t *kc = m->kc + (size_t)ai * c->n_ctx * kvd, *vc = m->vc + (size_t)ai * c->n_ctx * kvd;
    const int half = c->n_rot / 2;
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < T; t++) {
        const int pos = pos0 + t;
        float cs[256], sn[256];
        if (m->pf_rp) rope_cs(c, L, m->pf_rp[t], m->pf_rp[T + t], m->pf_rp[2 * T + t], cs, sn);
        else { const int rp = pos + m->rope_delta; rope_cs(c, L, rp, rp, rp, cs, sn); }
        float *qg = QG + (size_t)t * QD, *k = K + (size_t)t * kvd;
        for (int h = 0; h < NH + NKV; h++) {
            float *p = h < NH ? qg + (size_t)h * QS : k + (size_t)(h - NH) * HD;   // q in place, gate after it
            rmsnorm(p, p, h < NH ? L->q_norm : L->k_norm, HD, c->eps);
            for (int i = 0; i < half; i++) {
                const float a = p[i], bb = p[i + half];
                p[i] = a * cs[i] - bb * sn[i];
                p[i + half] = a * sn[i] + bb * cs[i];
            }
        }
        f32_to_f16(kc + (size_t)pos * kvd, k, kvd);
        f32_to_f16(vc + (size_t)pos * kvd, V + (size_t)t * kvd, kvd);
    }
    const float scale = 1.0f / sqrtf((float)HD);
    // work item: QB query tokens x the gqa heads sharing one KV head. K/V rows are converted
    // to f32 a block at a time and reused by all of the item's queries.
    enum { QB = 8, KB = 64 };
    const int NQ = QB * gqa, nqb = (T + QB - 1) / QB, S_all = pos0 + T;
    #pragma omp parallel
    {
        float *att = aligned_alloc(64, ((size_t)NQ * S_all * 4 + 63) / 64 * 64);
        float *Kf = aligned_alloc(64, (size_t)KB * HD * 4), *Vf = aligned_alloc(64, (size_t)KB * HD * 4);
        float *acc = aligned_alloc(64, (size_t)NQ * HD * 4);
        #pragma omp for schedule(dynamic, 1)
        for (int item = 0; item < nqb * NKV; item++) {
            const int qb = nqb - 1 - item / NKV, kvh = item % NKV;      // longest first
            const int tq0 = qb * QB, tq1 = tq0 + QB < T ? tq0 + QB : T, nt = tq1 - tq0;
            const int S = pos0 + tq1;                                   // keys needed by the last query
            const float *qp[64];
            int qpos[64], qlo[64];
            for (int a = 0; a < nt; a++)
                for (int g = 0; g < gqa; g++) {
                    qp[a * gqa + g] = QG + (size_t)(tq0 + a) * QD + (size_t)(kvh * gqa + g) * QS;
                    qpos[a * gqa + g] = pos0 + tq0 + a;
                    qlo[a * gqa + g] = attn_lo(c, L, pos0 + tq0 + a + 1);   // sliding window start
                }
            const int nq = nt * gqa, kstart = qlo[0] / KB * KB;          // first query has the lowest start
            for (int kb = kstart; kb < S; kb += KB) {
                const int ke = kb + KB < S ? kb + KB : S;
                for (int k = kb; k < ke; k++)
                    for (int i = 0; i < HD; i += 16)
                        _mm512_storeu_ps(Kf + (size_t)(k - kb) * HD + i, f16x16(kc + (size_t)k * kvd + kvh * HD + i));
                for (int q = 0; q < nq; q++) {
                    const int lim = qpos[q] + 1 < ke ? qpos[q] + 1 : ke;
                    float *aq = att + (size_t)q * S_all;
                    for (int k = kb > qlo[q] ? kb : qlo[q]; k < lim; k++) {
                        const float *kr = Kf + (size_t)(k - kb) * HD;
                        __m512 a0 = _mm512_setzero_ps(), a1 = a0;
                        for (int i = 0; i < HD; i += 32) {
                            a0 = _mm512_fmadd_ps(_mm512_loadu_ps(qp[q] + i), _mm512_load_ps(kr + i), a0);
                            a1 = _mm512_fmadd_ps(_mm512_loadu_ps(qp[q] + i + 16), _mm512_load_ps(kr + i + 16), a1);
                        }
                        aq[k] = _mm512_reduce_add_ps(_mm512_add_ps(a0, a1)) * scale;
                    }
                }
            }
            for (int q = 0; q < nq; q++) softmax_inplace(att + (size_t)q * S_all + qlo[q], qpos[q] + 1 - qlo[q]);
            memset(acc, 0, (size_t)nq * HD * 4);
            for (int kb = kstart; kb < S; kb += KB) {
                const int ke = kb + KB < S ? kb + KB : S;
                for (int k = kb; k < ke; k++)
                    for (int i = 0; i < HD; i += 16)
                        _mm512_storeu_ps(Vf + (size_t)(k - kb) * HD + i, f16x16(vc + (size_t)k * kvd + kvh * HD + i));
                for (int q = 0; q < nq; q++) {
                    const int lim = qpos[q] + 1 < ke ? qpos[q] + 1 : ke;
                    const float *aq = att + (size_t)q * S_all;
                    float *o = acc + (size_t)q * HD;
                    for (int i0 = 0; i0 < HD; i0 += 128) {
                        __m512 r[8];
                        for (int j = 0; j < 8; j++) r[j] = _mm512_loadu_ps(o + i0 + 16 * j);
                        for (int k = kb > qlo[q] ? kb : qlo[q]; k < lim; k++) {
                            const __m512 a = _mm512_set1_ps(aq[k]);
                            const float *vr = Vf + (size_t)(k - kb) * HD + i0;
                            for (int j = 0; j < 8; j++) r[j] = _mm512_fmadd_ps(a, _mm512_load_ps(vr + 16 * j), r[j]);
                        }
                        for (int j = 0; j < 8; j++) _mm512_storeu_ps(o + i0 + 16 * j, r[j]);
                    }
                }
            }
            for (int a = 0; a < nt; a++)
                for (int g = 0; g < gqa; g++) {
                    const int q = a * gqa + g, h = kvh * gqa + g;
                    float *o = AO + (size_t)(tq0 + a) * NH * HD + (size_t)h * HD;
                    const float *gate = qp[q] + HD;
                    for (int i = 0; i < HD; i++) o[i] = c->gated ? acc[(size_t)q * HD + i] * sigmoidf(gate[i]) : acc[(size_t)q * HD + i];
                }
        }
        free(att); free(Kf); free(Vf); free(acc);
    }
    mm(&L->wo, AO, NH * HD, T, b->R, D);
}

static void pf_moe(q35_model *m, int l, q35_pfbuf *b, int T) {
    const q35_cfg *c = &m->c;
    const q35_layer *L = &m->layers[l];
    const int D = c->n_embd, K = c->n_used, F = c->n_ff, FS = c->n_ff_sh, NE = c->n_expert;
    double tm = qnow();
    // shared expert for all tokens
    if (!L->sh_gate.data) memset(b->R, 0, (size_t)T * D * 4);
    else {
    mm(&L->sh_gate, b->XN, D, T, b->G, FS);
    mm(&L->sh_up, b->XN, D, T, b->H, FS);
    silu_mul(b->G, b->H, (size_t)T * FS);
    mm(&L->sh_down, b->G, FS, T, b->R, D);
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < T; t++) {
        const float sg = sigmoidf(dotf(L->shexp_gate, b->XN + (size_t)t * D, D));
        float *r = b->R + (size_t)t * D;
        for (int i = 0; i < D; i++) r[i] *= sg;
    }
    }
    q35_mp[0] += qnow() - tm; tm = qnow();
    // routing per token (router dots are parallel inside route(); tokens one after another)
    for (int t = 0; t < T; t++) route(L, b->XN + (size_t)t * D, c, K, b->sel + t * K, b->wsel + t * K);
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
    // experts: cached ones first while the misses stream in
    const int gt = L->gate_exps.type, ut = L->up_exps.type, dt = L->down_exps.type;
    const uint64_t grb = L->gate_exps.row_bytes, urb = L->up_exps.row_bytes, drb = L->down_exps.row_bytes;
    int miss[512], nm = 0, hq[512], issued = 0;
    const uint8_t *gw, *uw, *dw;
    for (int e = 0; e < NE; e++) {
        if (!b->cnt[e]) continue;
        if (experts_try_pin(m->experts, l, e, &gw, &uw, &dw)) experts_release(m->experts, l, e);
        else miss[nm++] = e;
    }
    q35_mp[1] += qnow() - tm;
#define TOPUP() while (issued < nm) { int h_ = experts_stream(m->experts, l, miss[issued]); if (h_ < 0) break; hq[issued++] = h_; }
    TOPUP();
    for (int pass = 0; pass < 2; pass++) {
        for (int e = 0, mi = 0; e < NE; e++) {
            if (!b->cnt[e]) continue;
            const int is_miss = mi < nm && miss[mi] == e;
            if (is_miss) mi++;
            if (pass == 0 && is_miss) continue;
            if (pass == 1 && !is_miss) continue;
            int h = -1;
            double te = qnow();
            if (pass == 0) { if (!experts_try_pin(m->experts, l, e, &gw, &uw, &dw)) continue; }
            else {
                TOPUP();
                int k = 0;
                while (miss[k] != e) k++;
                h = hq[k];
                experts_stream_get(m->experts, h, &gw, &uw, &dw);
                q35_mp[4] += qnow() - te; te = qnow();
            }
            const int n = b->cnt[e], o0 = b->off[e];
            for (int j = 0; j < n; j++) memcpy(b->GX + (size_t)j * D, b->XN + (size_t)b->tok[o0 + j] * D, D * 4);
            mm_any(gt, gw, grb, F, D, b->GX, D, n, b->G, F);
            mm_any(ut, uw, urb, F, D, b->GX, D, n, b->H, F);
            silu_mul(b->G, b->H, (size_t)n * F);
            mm_any(dt, dw, drb, D, F, b->G, F, n, b->EO, D);
            if (pass == 0) experts_release(m->experts, l, e);
            else experts_stream_done(m->experts, h);
            q35_mp[2 + pass] += qnow() - te;
            for (int j = 0; j < n; j++) {
                float *r = b->R + (size_t)b->tok[o0 + j] * D;
                const float w = b->tokw[o0 + j], *eo = b->EO + (size_t)j * D;
                for (int i = 0; i < D; i++) r[i] += w * eo[i];
            }
        }
    }
#undef TOPUP
    m->expert_loads += nm;
}

static void pf_chunk(q35_model *m, const int *tokens, const float *emb, int T, int pos0) {
    const q35_cfg *c = &m->c;
    q35_pfbuf *b = pf_get(m);
    const int D = c->n_embd;
    if (emb) memcpy(b->X, emb, (size_t)T * D * 4);
    else for (int t = 0; t < T; t++) embed(m, tokens[t], b->X + (size_t)t * D);
    for (int l = 0; l < c->n_layer; l++) {
        const q35_layer *L = &m->layers[l];
        #pragma omp parallel for schedule(static)
        for (int t = 0; t < T; t++) rmsnorm(b->XN + (size_t)t * D, b->X + (size_t)t * D, L->attn_norm, D, c->eps);
        double t0 = qnow();
        if (L->is_attn) { pf_attention(m, L, m->attn_idx[l], b, T, pos0); q35_prof[1] += qnow() - t0; }
        else { pf_delta_net(m, L, m->rec_idx[l], b, T); q35_prof[0] += qnow() - t0; }
        for (size_t i = 0; i < (size_t)T * D; i++) b->X[i] += b->R[i];
        #pragma omp parallel for schedule(static)
        for (int t = 0; t < T; t++) rmsnorm(b->XN + (size_t)t * D, b->X + (size_t)t * D, L->post_norm, D, c->eps);
        t0 = qnow();
        pf_moe(m, l, b, T);
        q35_prof[3] += qnow() - t0;
        for (size_t i = 0; i < (size_t)T * D; i++) b->X[i] += b->R[i];
    }
}

float *q35_prefill(q35_model *m, const int *tokens, int n, int pos0) {
    if (n < 8 || getenv("QUARRY_PER_TOKEN")) {
        float *lg = NULL;
        for (int i = 0; i < n; i++) lg = q35_forward(m, tokens[i], pos0 + i);
        return lg;
    }
    // all but the last token in chunks; the last one through the normal path for its logits
    int done = 0;
    while (done < n - 1) {
        int T = n - 1 - done;
        if (T > Q35_PF_T) T = Q35_PF_T;
        pf_chunk(m, tokens + done, NULL, T, pos0 + done);
        done += T;
    }
    return q35_forward(m, tokens[n - 1], pos0 + n - 1);
}

void q35_prefill_embd(q35_model *m, const float *emb, int n, int row0, const int *rp) {
    const int D = m->c.n_embd;
    int *crp = malloc(sizeof(int) * 3 * Q35_PF_T);
    for (int done = 0; done < n;) {
        const int T = n - done < Q35_PF_T ? n - done : Q35_PF_T;
        for (int k = 0; k < 3; k++) memcpy(crp + k * T, rp + k * n + done, sizeof(int) * T);
        m->pf_rp = crp;
        pf_chunk(m, NULL, emb + (size_t)done * D, T, row0 + done);
        m->pf_rp = NULL;
        done += T;
    }
    free(crp);
}
