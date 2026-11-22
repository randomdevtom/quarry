// matvec throughput (GB/s of weights) for the Inkling formats, weights in RAM
#define _GNU_SOURCE
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include "../src/qgguf.h"
#include "../src/quants.h"
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static void bench(qg_file *g, const char *name, int rows) {
    const qg_tensor *t = qg_find_tensor(g, name);
    const int n = t->ne[0]; const size_t rb = qg_row_bytes(t->type, n);
    uint8_t *w = aligned_alloc(64, (rb * rows + 63) / 64 * 64); pread(g->fd, w, rb * rows, t->offset);
    float *x = malloc(4 * n), *y = malloc(4 * rows);
    for (int i = 0; i < n; i++) x[i] = (i % 13) * 0.1f - 0.6f;
    block_q8_K *xq = malloc(sizeof(block_q8_K) * (n / 256 + 1));
    int8_t *qs = malloc(n); float *d = malloc(n / 8); int32_t *s = malloc(n / 8);
    act32 a = {qs, d, s};
    if (t->type >= GT_Q2_K) quantize_row_q8_K(x, xq, n); else quantize_row_a32(x, a, n);
    double best = 1e9;
    for (int it = 0; it < 5; it++) {
        double t0 = now();
        #pragma omp parallel for schedule(static)
        for (int r = 0; r < rows; r++) {
            const void *row = w + (size_t)r * rb;
            switch (t->type) {
            case GT_Q8_0: y[r] = vec_dot_q8_0_a32(n, row, a); break;
            case GT_Q5_0: y[r] = vec_dot_q5_0_a32(n, row, a); break;
            case GT_Q4_K: y[r] = vec_dot_q4_K_q8_K(n, row, xq); break;
            case GT_Q5_K: y[r] = vec_dot_q5_K_q8_K(n, row, xq); break;
            case GT_Q6_K: y[r] = vec_dot_q6_K_q8_K(n, row, xq); break;
            }
        }
        double dt = now() - t0; if (it && dt < best) best = dt;
    }
    printf("%-32s %s %6d x %5d: %6.2f ms  %5.1f GB/s\n", name, qg_type_name(t->type), rows, n, best * 1e3, rb * rows / best / 1e9);
}
int main(int argc, char **argv) {
    qg_file g; qg_open(&g, argv[1]);
    bench(&g, "blk.0.attn_q.weight", 3072);
    bench(&g, "blk.6.ffn_gate_up_exps.weight", 2304 * 8);   // Q4_K, 8 experts
    bench(&g, "blk.7.ffn_gate_up_exps.weight", 2304 * 8);
    bench(&g, "blk.1.ffn_gate_up_exps.weight", 2304 * 8);   // Q5_K
    bench(&g, "blk.6.ffn_down_exps.weight", 2304 * 8);      // Q5_0
    bench(&g, "blk.0.ffn_down_exps.weight", 2304 * 8);      // Q8_0
    bench(&g, "blk.0.attn_q.weight", 3072);                 // Q8_0 dense
    return 0;
}
