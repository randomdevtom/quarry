// Dequantize the first ROWS rows of some real tensors with Quarry's code and
// write them as float32 files; tests/check_quants.py compares against gguf-py.
// Also checks the integer dot products against float dot of dequantized data.
#define _GNU_SOURCE
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "../src/qgguf.h"
#include "../src/quants.h"

#define ROWS 8

static void *read_rows(qg_file *g, const qg_tensor *t, int rows, size_t *row_bytes) {
    *row_bytes = qg_row_bytes(t->type, t->ne[0]);
    void *buf = malloc(*row_bytes * rows);
    if (pread(g->fd, buf, *row_bytes * rows, t->offset) != (ssize_t)(*row_bytes * rows)) return NULL;
    return buf;
}

int main(int argc, char **argv) {
    qg_file g;
    if (qg_open(&g, argv[1])) return 1;
    const char *names[] = {"blk.0.attn_q.weight", "blk.5.ffn_down_exps.weight", "output.weight"};
    int fails = 0;
    srand(7);
    for (int k = 0; k < 3; k++) {
        const qg_tensor *t = qg_find_tensor(&g, names[k]);
        size_t rb;
        uint8_t *raw = read_rows(&g, t, ROWS, &rb);
        int n = (int)t->ne[0];
        float *deq = malloc(sizeof(float) * n * ROWS);
        for (int r = 0; r < ROWS; r++) {
            if (t->type == GT_Q3_K) dequant_row_q3_K((block_q3_K *)(raw + r * rb), deq + r * n, n);
            else dequant_row_q6_K((block_q6_K *)(raw + r * rb), deq + r * n, n);
        }
        char path[256];
        snprintf(path, sizeof path, "tests/out_%s.f32", names[k]);
        FILE *f = fopen(path, "wb");
        fwrite(deq, sizeof(float), (size_t)n * ROWS, f);
        fclose(f);

        // dot check: random activations, quantize to q8_K, compare to float math
        float *x = malloc(sizeof(float) * n);
        for (int i = 0; i < n; i++) x[i] = (rand() / (float)RAND_MAX - 0.5f) * 4;
        block_q8_K *xq = malloc(sizeof(block_q8_K) * (n / QK_K));
        quantize_row_q8_K(x, xq, n);
        float *xd = malloc(sizeof(float) * n);  // dequantized activations
        for (int b = 0; b < n / QK_K; b++)
            for (int i = 0; i < QK_K; i++) xd[b * QK_K + i] = xq[b].d * xq[b].qs[i];
        double worst = 0;
        for (int r = 0; r < ROWS; r++) {
            double ref = 0;
            for (int i = 0; i < n; i++) ref += (double)deq[r * n + i] * xd[i];
            float got_ref, got_fast;
            if (t->type == GT_Q3_K) {
                got_ref = vec_dot_q3_K_q8_K_ref(n, (block_q3_K *)(raw + r * rb), xq);
                got_fast = vec_dot_q3_K_q8_K(n, (block_q3_K *)(raw + r * rb), xq);
            } else {
                got_ref = vec_dot_q6_K_q8_K_ref(n, (block_q6_K *)(raw + r * rb), xq);
                got_fast = vec_dot_q6_K_q8_K(n, (block_q6_K *)(raw + r * rb), xq);
            }
            double e1 = fabs(got_ref - ref) / (fabs(ref) + 1e-3), e2 = fabs(got_fast - got_ref) / (fabs(got_ref) + 1e-3);
            if (e1 > worst) worst = e1;
            if (e2 > worst) worst = e2;
        }
        printf("%-28s %s n=%d  dot rel err (ref vs float, fast vs ref): %.2e  %s\n", names[k],
               qg_type_name(t->type), n, worst, worst < 1e-4 ? "OK" : "FAIL");
        fails += worst >= 1e-4;
    }
    return fails;
}
