// kq_matmul throughput and correctness on a real expert tensor: GMAC/s, single call
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
int main(int argc, char **argv) {
    qg_file g;
    if (argc < 4 || qg_open(&g, argv[1])) return 1;
    const qg_tensor *t = qg_find_tensor(&g, argv[2]);
    const int T = atoi(argv[3]), n = (int)t->ne[0], rows = (int)t->ne[1];
    const size_t rb = qg_row_bytes(t->type, n);
    uint8_t *w = aligned_alloc(64, (rb * rows + 63) / 64 * 64);
    if (pread(g.fd, w, rb * rows, t->offset) < 0) return 1;
    int type = t->type;
    if (type == GT_Q4_K) { q4k_repack_kr(w, rb * rows / sizeof(block_q4_K)); type = GT_Q4_KR; }
    const int nb = n / QK_K;
    block_q8_K *qn = malloc(sizeof(block_q8_K) * nb * T), *qp = malloc(sizeof(block_q8_K) * nb * T);
    float *xs = malloc(sizeof(float) * nb * 16 * T), *x = malloc(sizeof(float) * n), *Y = malloc(sizeof(float) * rows * T);
    const block_q8_K *xp[256]; const float *xsp[256];
    for (int k = 0; k < T; k++) {
        for (int i = 0; i < n; i++) x[i] = sinf(i * 0.37f + k) * (1 + (i % 7));
        quantize_row_q8_K(x, qn + k * nb, n);
        q8k_permute_kr(qn + k * nb, qp + k * nb, n);
        kq_xs(qn + k * nb, n, xs + k * nb * 16);
        xp[k] = (type == GT_Q4_KR ? qp : qn) + k * nb; xsp[k] = xs + k * nb * 16;
    }
    double best = 1e9;
    for (int it = 0; it < 10; it++) {
        double t0 = now();
        kq_matmul(type, w, rb, rows, n, T, xp, xsp, Y, rows);
        double dt = now() - t0;
        if (dt < best) best = dt;
    }
    double err = 0;
    for (int r = 0; r < rows; r += 97) for (int k = 0; k < T; k += 5) {
        float ref = type == GT_Q4_KR ? vec_dot_q4_KR_q8_KR(n, (const block_q4_K *)(w + r * rb), qp + k * nb) : vec_dot_q6_K_q8_K(n, (const block_q6_K *)(w + r * rb), qn + k * nb);
        double e = fabs(ref - Y[(size_t)k * rows + r]) / (fabs(ref) + 1e-3); if (e > err) err = e;
    }
    printf("%s %s T=%d threads %d: %.2f ms, %.1f GMAC/s, max rel err %.2e\n", argv[2], type == GT_Q4_KR ? "Q4_KR" : "Q6_K", T, omp_get_max_threads(), best * 1e3, (double)rows * n * T / best / 1e9, err);
    return 0;
}
