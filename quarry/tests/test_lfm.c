// LFM2 MoE check: run the token ids of a llama.cpp reference (build/ref_logits) token by token
// and batched, and compare the last logits with the reference's.
//   test_lfm model.gguf ref_prefix     (reads ref_prefix.ids, ref_prefix.logits)
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/lfm2.h"

static void compare(const char *what, const float *a, const float *ref, int nv) {
    double ab = 0, aa = 0, bb = 0, md = 0;
    int ta[5], tr[5];
    for (int i = 0; i < nv; i++) {
        ab += (double)a[i] * ref[i]; aa += (double)a[i] * a[i]; bb += (double)ref[i] * ref[i];
        if (fabs(a[i] - ref[i]) > md) md = fabs(a[i] - ref[i]);
    }
    for (int k = 0; k < 5; k++) {
        ta[k] = tr[k] = -1;
        for (int i = 0; i < nv; i++) {
            int skip_a = 0, skip_r = 0;
            for (int j = 0; j < k; j++) { skip_a |= ta[j] == i; skip_r |= tr[j] == i; }
            if (!skip_a && (ta[k] < 0 || a[i] > a[ta[k]])) ta[k] = i;
            if (!skip_r && (tr[k] < 0 || ref[i] > ref[tr[k]])) tr[k] = i;
        }
    }
    printf("%s: cosine %.6f, max |diff| %.3f | top5 quarry", what, ab / sqrt(aa * bb), md);
    for (int k = 0; k < 5; k++) printf(" %d", ta[k]);
    printf(" | ref");
    for (int k = 0; k < 5; k++) printf(" %d", tr[k]);
    printf("\n");
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s model.gguf ref_prefix\n", argv[0]); return 1; }
    char path[512];
    snprintf(path, sizeof path, "%s.ids", argv[2]);
    FILE *f = fopen(path, "r");
    if (!f) { perror(path); return 1; }
    int ids[4096], n = 0;
    while (n < 4096 && fscanf(f, "%d", &ids[n]) == 1) n++;
    fclose(f);
    static lfm_model m;
    if (lfm_load(&m, argv[1], 2048)) return 1;
    const int nv = m.c.n_vocab;
    float *ref = malloc(sizeof(float) * nv), *a = malloc(sizeof(float) * nv);
    snprintf(path, sizeof path, "%s.logits", argv[2]);
    f = fopen(path, "rb");
    if (!f || fread(ref, sizeof(float), nv, f) != (size_t)nv) { fprintf(stderr, "cannot read %s\n", path); return 1; }
    fclose(f);
    printf("%d tokens; %d layers (%d attention), %d experts / %d used\n", n, m.c.n_layer, m.n_attn, m.c.n_expert, m.c.n_used);
    const float *lg = NULL;
    for (int i = 0; i < n; i++) lg = lfm_forward(&m, ids[i], i);
    memcpy(a, lg, sizeof(float) * nv);
    compare("token by token", a, ref, nv);
    lfm_state_reset(&m);
    lg = lfm_prefill(&m, ids, n, 0);
    compare("batched", lg, ref, nv);
    compare("batched vs token by token", lg, a, nv);
    return 0;
}
