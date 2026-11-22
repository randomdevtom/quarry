// Throughput of Q3_K / Q6_K matvec on real weights held in RAM.
#define _GNU_SOURCE
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include "../src/qgguf.h"
#include "../src/quants.h"

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

static void bench(qg_file *g, const char *name, int64_t max_rows) {
    const qg_tensor *t = qg_find_tensor(g, name);
    int n = (int)t->ne[0];
    int64_t rows = t->ne[1] * t->ne[2];
    if (rows > max_rows) rows = max_rows;
    size_t rb = qg_row_bytes(t->type, n);
    uint8_t *w = aligned_alloc(64, (rb * rows + 63) / 64 * 64);
    pread(g->fd, w, rb * rows, t->offset);
    float *x = malloc(sizeof(float) * n), *y = malloc(sizeof(float) * rows);
    for (int i = 0; i < n; i++) x[i] = (i % 17) * 0.1f - 0.8f;
    block_q8_K *xq = malloc(sizeof(block_q8_K) * (n / QK_K));
    quantize_row_q8_K(x, xq, n);
    double best = 1e9;
    for (int it = 0; it < 6; it++) {
        double t0 = now();
        #pragma omp parallel for schedule(static)
        for (int64_t r = 0; r < rows; r++)
            y[r] = t->type == GT_Q3_K ? vec_dot_q3_K_q8_K(n, (block_q3_K *)(w + r * rb), xq)
                                      : vec_dot_q6_K_q8_K(n, (block_q6_K *)(w + r * rb), xq);
        double dt = now() - t0;
        if (it && dt < best) best = dt;
    }
    printf("%-28s %s %6lld rows x %5d: %6.2f ms, %5.1f GB/s (threads=%d)\n", name, qg_type_name(t->type),
           (long long)rows, n, best * 1e3, rb * rows / best / 1e9, omp_get_max_threads());
    free(w);
}

int main(int argc, char **argv) {
    qg_file g;
    if (qg_open(&g, argv[1])) return 1;
    bench(&g, "blk.0.ffn_gate_exps.weight", 1 << 30);  // all 128 experts, 86 MB
    bench(&g, "blk.0.ffn_down_exps.weight", 1 << 30);
    bench(&g, "output.weight", 1 << 30);                // 151936 x 2048 Q6_K, 255 MB
    return 0;
}
