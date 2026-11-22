// Batched Q3_K path (repack + dpbusd over many tokens) vs per-token kernel:
// accuracy against float math, and speed per token at several batch sizes.
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
    const int n = t->ne[0], rows = 768 * 16, nb = n / QK_K, BMAX = 64;
    size_t rb = qg_row_bytes(t->type, n);
    uint8_t *w = aligned_alloc(64, rb * rows); pread(g.fd, w, rb * rows, t->offset);
    float *x = malloc(4 * (size_t)n * BMAX);
    srand(3);
    for (int i = 0; i < n * BMAX; i++) x[i] = (rand() / (float)RAND_MAX - 0.5f) * 3;
    block_q8_K *xq = malloc(sizeof(block_q8_K) * nb * BMAX);
    uint8_t *xu = aligned_alloc(64, (size_t)n * BMAX); float *dx = malloc(4 * nb * BMAX);
    for (int k = 0; k < BMAX; k++) { quantize_row_q8_K(x + k * n, xq + k * nb, n); quantize_row_u8(x + k * n, xu + k * n, dx + k * nb, n); }
    // accuracy on 64 rows: both against exact float dot of dequantized weights
    float *deq = malloc(4 * n); double e_old = 0, e_new = 0;
    q3k_s8 *pk = aligned_alloc(64, sizeof(q3k_s8) * nb);
    for (int r = 0; r < 64; r++) {
        dequant_row_q3_K((block_q3_K *)(w + r * rb), deq, n);
        q3k_repack_row((block_q3_K *)(w + r * rb), n, pk);
        float o[BMAX]; dot_q3k_s8_multi(n, pk, BMAX, xu, dx, o, 1);
        for (int k = 0; k < BMAX; k++) {
            double ref = 0, mag = 0;
            for (int i = 0; i < n; i++) { ref += (double)deq[i] * x[k * n + i]; mag += fabs(deq[i] * x[k * n + i]); }
            double old = vec_dot_q3_K_q8_K(n, (block_q3_K *)(w + r * rb), xq + k * nb);
            e_old = fmax(e_old, fabs(old - ref) / mag); e_new = fmax(e_new, fabs(o[k] - ref) / mag);
        }
    }
    printf("error vs float (relative to sum|w*x|): per-token kernel %.2e, batched %.2e\n", e_old, e_new);
    float *y = malloc(4 * (size_t)rows * BMAX);
    double t0 = now();
    #pragma omp parallel for
    for (int r = 0; r < rows; r++) y[r] = vec_dot_q3_K_q8_K(n, (block_q3_K *)(w + r * rb), xq);
    double t1 = now() - t0;
    printf("per-token kernel: %.2f ms per token\n", t1 * 1e3);
    for (int B = 1; B <= BMAX; B *= 4) {
        t0 = now();
        #pragma omp parallel
        {
            q3k_s8 *p = aligned_alloc(64, sizeof(q3k_s8) * nb);
            #pragma omp for
            for (int r = 0; r < rows; r++) { q3k_repack_row((block_q3_K *)(w + r * rb), n, p); dot_q3k_s8_multi(n, p, B, xu, dx, y + r, rows); }
            free(p);
        }
        double tb = now() - t0;
        printf("batched B=%-3d: %.2f ms per token  (%.1fx)\n", B, tb * 1e3 / B, t1 / (tb / B));
    }
}
