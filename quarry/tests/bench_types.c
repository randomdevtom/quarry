// Matvec throughput per weight type on real Qwen3.5/3.6 tensors held in RAM:
// one input (decode) and two inputs (speculative verify), GB/s of weights read.
#define _GNU_SOURCE
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <math.h>
#include "../src/qgguf.h"
#include "../src/quants.h"

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

static block_q8_K *g_xp;   // permuted activations for GT_Q4_KR
static float dot1(int ty, const void *w, int n, const block_q8_K *x) {
    switch (ty) {
    case GT_Q4_KR: return vec_dot_q4_KR_q8_KR(n, w, g_xp ? g_xp : x);
    case GT_Q3_K: return vec_dot_q3_K_q8_K(n, w, x);
    case GT_Q4_K: return vec_dot_q4_K_q8_K(n, w, x);
    case GT_Q5_K: return vec_dot_q5_K_q8_K(n, w, x);
    default: return vec_dot_q6_K_q8_K(n, w, x);
    }
}
static void dot2(int ty, const void *w, int n, const block_q8_K *x0, const block_q8_K *x1, float *a, float *b) {
    switch (ty) {
    case GT_Q3_K: vec_dot2_q3_K_q8_K(n, w, x0, x1, a, b); return;
    case GT_Q4_K: vec_dot2_q4_K_q8_K(n, w, x0, x1, a, b); return;
    case GT_Q5_K: vec_dot2_q5_K_q8_K(n, w, x0, x1, a, b); return;
    default: *a = dot1(ty, w, n, x0); *b = dot1(ty, w, n, x1);
    }
}

static void bench(qg_file *g, const char *name, int64_t max_rows) {
    const qg_tensor *t = qg_find_tensor(g, name);
    if (!t) { printf("%s: missing\n", name); return; }
    const int n = (int)t->ne[0];
    int64_t rows = t->ne[1] * (t->ne[2] ? t->ne[2] : 1);
    if (rows > max_rows) rows = max_rows;
    const size_t rb = qg_row_bytes(t->type, n);
    uint8_t *w = aligned_alloc(64, (rb * rows + 63) / 64 * 64);
    if (pread(g->fd, w, rb * rows, t->offset) < 0) return;
    float *x = malloc(sizeof(float) * n), *y = malloc(sizeof(float) * rows * 2);
    for (int i = 0; i < n; i++) x[i] = (i % 17) * 0.1f - 0.8f;
    block_q8_K *x0 = malloc(sizeof(block_q8_K) * (n / QK_K)), *x1 = malloc(sizeof(block_q8_K) * (n / QK_K));
    quantize_row_q8_K(x, x0, n);
    int ty = t->type;
    if (getenv("KR") && ty == GT_Q4_K) {
        // reference dot before the repack, then repack and compare
        float ref = vec_dot_q4_K_q8_K(n, (const block_q4_K *)w, x0);
        q4k_repack_kr(w, rb * rows / sizeof(block_q4_K));
        g_xp = malloc(sizeof(block_q8_K) * (n / QK_K));
        q8k_permute_kr(x0, g_xp, n);
        float kr = vec_dot_q4_KR_q8_KR(n, (const block_q4_K *)w, g_xp);
        printf("%-26s Q4_KR row 0: %.6f vs Q4_K %.6f\n", name, kr, ref);
        ty = GT_Q4_KR;
    }
    for (int i = 0; i < n; i++) x[i] = -x[i] * 0.5f;
    quantize_row_q8_K(x, x1, n);
    {   // correctness: fast dot vs float reference over dequantized weights and inputs
        float *wf = malloc(sizeof(float) * n), *xf0 = malloc(sizeof(float) * n), *xf1 = malloc(sizeof(float) * n);
        for (int bk = 0; bk < n / QK_K; bk++)
            for (int i = 0; i < QK_K; i++) { xf0[bk * QK_K + i] = x0[bk].qs[i] * x0[bk].d; xf1[bk * QK_K + i] = x1[bk].qs[i] * x1[bk].d; }
        double worst = 0;
        for (int64_t r = 0; r < rows; r += rows / 64 + 1) {
            dequant_row(t->type, w + r * rb, wf, n);
            double ref0 = 0, ref1 = 0, mag = 0;
            for (int i = 0; i < n; i++) { ref0 += (double)wf[i] * xf0[i]; ref1 += (double)wf[i] * xf1[i]; mag += fabs((double)wf[i] * xf0[i]); }
            float a, b;
            dot2(t->type, w + r * rb, n, x0, x1, &a, &b);
            const float c = dot1(t->type, w + r * rb, n, x0);
            double e = fabs(c - ref0) / (mag + 1e-9);
            if (fabs(a - ref0) / (mag + 1e-9) > e) e = fabs(a - ref0) / (mag + 1e-9);
            if (fabs(b - ref1) / (mag + 1e-9) > e) e = fabs(b - ref1) / (mag + 1e-9);
            if (e > worst) worst = e;
        }
        printf("%-26s check: worst relative error %.2e %s\n", name, worst, worst < 1e-5 ? "ok" : "MISMATCH");
        free(wf); free(xf0); free(xf1);
    }
    for (int two = 0; two < 2; two++) {
        double best = 1e9;
        for (int it = 0; it < 8; it++) {
            double t0 = now();
            #pragma omp parallel for schedule(static)
            for (int64_t r = 0; r < rows; r++) {
                if (two) dot2(ty, w + r * rb, n, x0, x1, &y[2 * r], &y[2 * r + 1]);
                else y[r] = dot1(ty, w + r * rb, n, x0);
            }
            double dt = now() - t0;
            if (it && dt < best) best = dt;
        }
        if (ty == GT_Q4_KR && two) break;
        printf("%-26s %-5s %s: %6.2f ms, %5.1f GB/s (threads %d)\n", name, qg_type_name(t->type), two ? "2 inputs" : "1 input ",
               best * 1e3, rb * rows / best / 1e9, omp_get_max_threads());
        if (ty == GT_Q4_KR) printf("  (Q4_KR)\n");
    }
    free(w); free(x); free(y); free(x0); free(x1);
}

int main(int argc, char **argv) {
    qg_file g;
    if (argc < 2 || qg_open(&g, argv[1])) return 1;
    if (argc > 2) {   // tensor names to bench, optionally name:max_rows
        for (int i = 2; i < argc; i++) {
            char nm[128]; snprintf(nm, sizeof nm, "%s", argv[i]);
            char *c = strchr(nm, ':');
            int64_t mr = 1 << 30;
            if (c) { *c = 0; mr = atoll(c + 1); }
            bench(&g, nm, mr);
        }
        return 0;
    }
    bench(&g, "blk.0.attn_qkv.weight", 1 << 30);
    bench(&g, "blk.0.ssm_out.weight", 1 << 30);
    bench(&g, "blk.1.ffn_gate_exps.weight", 1 << 30);
    bench(&g, "blk.1.ffn_down_exps.weight", 1 << 30);
    bench(&g, "blk.0.ffn_down_exps.weight", 1 << 30);
    bench(&g, "blk.3.attn_output.weight", 1 << 30);
    for (int l = 2; l < 40; l++) { char nm[64]; snprintf(nm, sizeof nm, "blk.%d.ffn_down_exps.weight", l); const qg_tensor *t = qg_find_tensor(&g, nm); if (t && t->type == GT_Q4_K) { bench(&g, nm, 1 << 30); break; } }
    bench(&g, "output.weight", 1 << 30);
    return 0;
}
