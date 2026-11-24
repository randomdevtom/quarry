// K-quant kernels on random blocks, no model file needed: every fast dot (Q4_K, Q5_K, Q6_K,
// Q4_KR, the two-input variants and the batched kq_matmul) against float math on the
// scalar dequantization, plus the Q4_KR repack round trip.
#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/qgguf.h"
#include "../src/quants.h"

static uint16_t f16(float f) { return (uint16_t)_cvtss_sh(f, 0); }
static float frnd(void) { return rand() / (float)RAND_MAX; }

// random blocks with sane fp16 scales
static void fill(int type, uint8_t *p, int nb) {
    const size_t bs = qg_type_block_bytes(type);
    for (size_t i = 0; i < bs * nb; i++) p[i] = (uint8_t)rand();
    for (int b = 0; b < nb; b++) {
        uint8_t *blk = p + b * bs;
        if (type == GT_Q4_K || type == GT_Q5_K) { block_q4_K *x = (block_q4_K *)blk; x->d = f16(0.01f + 0.02f * frnd()); x->dmin = f16(0.01f * frnd()); }
        if (type == GT_Q6_K) ((block_q6_K *)blk)->d = f16(0.001f + 0.004f * frnd());
    }
}

static int fails;
static void check(const char *what, double got, double ref, double mag) {
    const double e = fabs(got - ref) / (mag + 1e-9);
    if (e > 1e-5) { printf("FAIL %s: %.6f vs %.6f (rel %.2e)\n", what, got, ref, e); fails++; }
}

int main(void) {
    srand(7);
    const int n = 2048, nb = n / QK_K, R = 6, T = 7;
    const int types[3] = {GT_Q4_K, GT_Q5_K, GT_Q6_K};
    float *x = malloc(4 * n * T), *xd = malloc(4 * n * T), *wd = malloc(4 * n);
    block_q8_K *xq = malloc(sizeof(block_q8_K) * nb * T), *xp = malloc(sizeof(block_q8_K) * nb * T);
    float *xs = malloc(4 * nb * 16 * T);
    for (int t = 0; t < T; t++) {
        for (int i = 0; i < n; i++) x[t * n + i] = (frnd() - 0.5f) * 4 * (1 + (i % 5 == 0) * 6);
        quantize_row_q8_K(x + t * n, xq + t * nb, n);
        q8k_permute_kr(xq + t * nb, xp + t * nb, n);
        kq_xs(xq + t * nb, n, xs + t * nb * 16);
        for (int i = 0; i < n; i++) xd[t * n + i] = xq[t * nb + i / QK_K].d * xq[t * nb + i / QK_K].qs[i % QK_K];
    }
    for (int ti = 0; ti < 3; ti++) {
        const int type = types[ti];
        const size_t rb = qg_row_bytes(type, n);
        uint8_t *w = malloc(rb * R), *wkr = malloc(rb * R);
        fill(type, w, nb * R);
        double ref[6][7], mag[6][7];
        for (int r = 0; r < R; r++) {
            dequant_row(type, w + r * rb, wd, n);
            for (int t = 0; t < T; t++) {
                ref[r][t] = mag[r][t] = 0;
                for (int i = 0; i < n; i++) { ref[r][t] += (double)wd[i] * xd[t * n + i]; mag[r][t] += fabs(wd[i] * xd[t * n + i]); }
            }
            const void *row = w + r * rb;
            const float got = type == GT_Q4_K ? vec_dot_q4_K_q8_K(n, row, xq) : type == GT_Q5_K ? vec_dot_q5_K_q8_K(n, row, xq) : vec_dot_q6_K_q8_K(n, row, xq);
            check(qg_type_name(type), got, ref[r][0], mag[r][0]);
            if (type != GT_Q6_K) {
                float a, b;
                if (type == GT_Q4_K) vec_dot2_q4_K_q8_K(n, row, xq, xq + nb, &a, &b); else vec_dot2_q5_K_q8_K(n, row, xq, xq + nb, &a, &b);
                check("dot2 a", a, ref[r][0], mag[r][0]);
                check("dot2 b", b, ref[r][1], mag[r][1]);
            }
        }
        int kt = type;
        const uint8_t *kw = w;
        if (type == GT_Q4_K) {
            memcpy(wkr, w, rb * R);
            q4k_repack_kr(wkr, (uint64_t)nb * R);
            for (int r = 0; r < R; r++) {
                float a[2048], b[2048];
                dequant_row(GT_Q4_K, w + r * rb, a, n);
                dequant_row(GT_Q4_KR, wkr + r * rb, b, n);
                if (memcmp(a, b, sizeof a)) { printf("FAIL Q4_KR dequant round trip, row %d\n", r); fails++; }
                check("Q4_KR", vec_dot_q4_KR_q8_KR(n, (const block_q4_K *)(wkr + r * rb), xp), ref[r][0], mag[r][0]);
            }
            kt = GT_Q4_KR;
            kw = wkr;
        }
        if (kq_supported(kt)) {   // batched: R rows x T tokens (exercises the edge tiles too)
            const block_q8_K *xpp[7]; const float *xsp[7];
            for (int t = 0; t < T; t++) { xpp[t] = (kt == GT_Q4_KR ? xp : xq) + t * nb; xsp[t] = xs + t * nb * 16; }
            float Y[7 * 6];
            kq_matmul(kt, kw, rb, R, n, T, xpp, xsp, Y, R);
            for (int r = 0; r < R; r++) for (int t = 0; t < T; t++) check("kq_matmul", Y[t * R + r], ref[r][t], mag[r][t]);
        }
        printf("%-5s ok so far (%d failures)\n", qg_type_name(type), fails);
        free(w); free(wkr);
    }
    printf("%s\n", fails ? "FAILED" : "all K-quant checks passed");
    return fails != 0;
}
