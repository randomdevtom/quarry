// Inkling formats: dump dequantized rows (for check_quants2.py vs gguf-py) and check
// the fast dots against float math on the dequantized weights.
#define _GNU_SOURCE
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "../src/qgguf.h"
#include "../src/quants.h"
int main(int argc, char **argv) {
    qg_file g; if (qg_open(&g, argv[1])) return 1;
    const char *names[] = {"blk.0.attn_q.weight", "blk.0.ffn_down_exps.weight", "blk.6.ffn_down_exps.weight",
                           "blk.6.ffn_gate_up_exps.weight", "blk.0.ffn_gate_up_exps.weight", NULL, NULL};
    // find one Q5_K gate_up
    static char q5k[64];
    for (int l = 0; l < 24; l++) { snprintf(q5k, sizeof q5k, "blk.%d.ffn_gate_up_exps.weight", l);
        if (qg_find_tensor(&g, q5k)->type == GT_Q5_K) { names[5] = q5k; break; } }
    int fails = 0; srand(5);
    for (int k = 0; names[k]; k++) {
        const qg_tensor *t = qg_find_tensor(&g, names[k]);
        const int n = t->ne[0], R = 8; size_t rb = qg_row_bytes(t->type, n);
        uint8_t *raw = malloc(rb * R); pread(g.fd, raw, rb * R, t->offset);
        float *deq = malloc(4 * n * R);
        for (int r = 0; r < R; r++) dequant_row(t->type, raw + r * rb, deq + r * n, n);
        char path[128]; snprintf(path, sizeof path, "build/deq_%d.f32", k);
        FILE *f = fopen(path, "wb"); fwrite(deq, 4, n * R, f); fclose(f);
        printf("%s %s %d\n", path, names[k], t->type);
        float *x = malloc(4 * n);
        for (int i = 0; i < n; i++) x[i] = (rand() / (float)RAND_MAX - 0.5f) * 4;
        block_q8_K *xq = malloc(sizeof(block_q8_K) * (n / QK_K + 1));
        int8_t *qs = malloc(n); float *ad = malloc(4 * n / 32); int32_t *as = malloc(4 * n / 32);
        act32 a = {qs, ad, as};
        float *xd = malloc(4 * n);
        int k_quant = t->type >= GT_Q2_K;
        if (k_quant) { quantize_row_q8_K(x, xq, n); for (int i = 0; i < n; i++) xd[i] = xq[i / QK_K].d * xq[i / QK_K].qs[i % QK_K]; }
        else { quantize_row_a32(x, a, n); for (int i = 0; i < n; i++) xd[i] = ad[i / 32] * qs[i]; }
        double worst = 0;
        for (int r = 0; r < R; r++) {
            double ref = 0, mag = 0;
            for (int i = 0; i < n; i++) { ref += (double)deq[r * n + i] * xd[i]; mag += fabs(deq[r * n + i] * xd[i]); }
            const void *row = raw + r * rb; float got = 0;
            switch (t->type) {
            case GT_Q8_0: got = vec_dot_q8_0_a32(n, row, a); break;
            case GT_Q5_0: got = vec_dot_q5_0_a32(n, row, a); break;
            case GT_Q4_K: got = vec_dot_q4_K_q8_K(n, row, xq); break;
            case GT_Q5_K: got = vec_dot_q5_K_q8_K(n, row, xq); break;
            case GT_Q6_K: got = vec_dot_q6_K_q8_K(n, row, xq); break;
            }
            double e = fabs(got - ref) / mag; if (e > worst) worst = e;
        }
        printf("  %-32s %s n=%d dot rel err %.2e %s\n", names[k], qg_type_name(t->type), n, worst, worst < 1e-5 ? "OK" : "FAIL");
        fails += worst >= 1e-5;
    }
    return fails;
}
