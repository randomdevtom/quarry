// Run Quarry's Inkling forward pass on reference token ids and compare the final logits
// and every slot's layer output (l_out) with the llama.cpp dumps from ref_dump.
#define _GNU_SOURCE
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../src/inkling.h"

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

static double cosine(const float *a, const float *b, int n) {
    double d = 0, na = 0, nb = 0;
    for (int i = 0; i < n; i++) { d += (double)a[i] * b[i]; na += (double)a[i] * a[i]; nb += (double)b[i] * b[i]; }
    return d / sqrt(na * nb + 1e-30);
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s model ref_prefix\n", argv[0]); return 1; }
    ink_model m;
    if (ink_load(&m, argv[1], 512)) return 1;
    const int D = m.c.n_embd;
    m.dbg_lout = calloc((size_t)m.c.n_slots * D, 4);
    char path[512];
    snprintf(path, sizeof path, "%s.ids", argv[2]);
    FILE *f = fopen(path, "r");
    int ids[512], n = 0;
    while (n < 512 && fscanf(f, "%d", &ids[n]) == 1) n++;
    fclose(f);
    float *lg = NULL;
    double t0 = now();
    for (int i = 0; i < n; i++) lg = ink_forward(&m, ids[i], i);
    printf("%d tokens in %.1fs (%.2f tok/s), experts read %.2f GB\n", n, now() - t0, n / (now() - t0), m.expert_bytes_read / 1e9);
    float *ref = malloc(D * 4);
    for (int s = 0; s < m.c.n_slots; s++) {
        snprintf(path, sizeof path, "%s.l_out-%d.f32", argv[2], s);
        f = fopen(path, "rb");
        if (!f) continue;
        fread(ref, 4, D, f); fclose(f);
        printf("slot %2d (layer %2d)  l_out cosine %.6f\n", s, m.slot_phys[s], cosine(m.dbg_lout + (size_t)s * D, ref, D));
    }
    const int nv = m.c.n_vocab;
    float *rl = malloc(nv * 4);
    snprintf(path, sizeof path, "%s.logits", argv[2]);
    f = fopen(path, "rb"); fread(rl, 4, nv, f); fclose(f);
    int a = 0, b = 0;
    for (int i = 1; i < nv; i++) { if (lg[i] > lg[a]) a = i; if (rl[i] > rl[b]) b = i; }
    printf("logits cosine %.6f  top quarry %d  top llama %d  %s\n", cosine(lg, rl, nv), a, b, a == b ? "MATCH" : "MISMATCH");
    return 0;
}
