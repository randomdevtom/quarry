// Check the unpacked Q3_K path against the fast kernel, and measure per-token cost.
#define _GNU_SOURCE
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include "../src/qgguf.h"
#include "../src/quants.h"
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
int main(int argc, char **argv) {
    qg_file g; qg_open(&g, argv[1]);
    const qg_tensor *t = qg_find_tensor(&g, "blk.0.ffn_gate_exps.weight");
    const int n = t->ne[0], rows = 768 * 16, B = 64, nb = n / QK_K;
    size_t rb = qg_row_bytes(t->type, n);
    uint8_t *w = aligned_alloc(64, rb * rows); pread(g.fd, w, rb * rows, t->offset);
    block_q8_K *xq = malloc(sizeof(block_q8_K) * nb * B);
    float *x = malloc(4 * n);
    srand(3);
    for (int k = 0; k < B; k++) { for (int i = 0; i < n; i++) x[i] = rand() / (float)RAND_MAX - 0.5f; quantize_row_q8_K(x, xq + k * nb, n); }
    float *y1 = malloc(4 * rows * B), *y2 = malloc(4 * rows * B);
    double t0 = now();
    #pragma omp parallel for
    for (int r = 0; r < rows; r++) for (int k = 0; k < B; k++) y1[r * B + k] = vec_dot_q3_K_q8_K(n, (block_q3_K *)(w + r * rb), xq + k * nb);
    double t_old = now() - t0;
    t0 = now();
    #pragma omp parallel
    {
        q3k_unpacked *u = aligned_alloc(64, sizeof(q3k_unpacked) * nb);
        #pragma omp for
        for (int r = 0; r < rows; r++) {
            q3k_unpack_row((block_q3_K *)(w + r * rb), n, u);
            for (int k = 0; k < B; k++) y2[r * B + k] = vec_dot_q3k_unpacked(n, u, xq + k * nb);
        }
    }
    double t_new = now() - t0;
    double worst = 0;
    for (int i = 0; i < rows * B; i++) { double e = fabs(y1[i] - y2[i]) / (fabs(y1[i]) + 1e-2); if (e > worst) worst = e; }
    printf("rows %d x tokens %d: per-token kernel %.1f ms, unpack-once %.1f ms -> %.2fx  | max rel diff %.1e %s\n",
           rows, B, t_old * 1e3, t_new * 1e3, t_old / t_new, worst, worst < 1e-4 ? "OK" : "FAIL");
}
