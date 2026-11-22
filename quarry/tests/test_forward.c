// Run Quarry's forward pass on the reference token ids and compare final logits.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../src/model.h"

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

static void top5(const float *l, int n, int out[5]) {
    for (int k = 0; k < 5; k++) {
        int b = -1;
        for (int i = 0; i < n; i++) {
            int skip = 0;
            for (int j = 0; j < k; j++) skip |= out[j] == i;
            if (!skip && (b < 0 || l[i] > l[b])) b = i;
        }
        out[k] = b;
    }
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s model ref_prefix\n", argv[0]); return 1; }
    qmodel m;
    double t0 = now();
    if (model_load(&m, argv[1], 512)) return 1;
    printf("loaded in %.1fs\n", now() - t0);
    char path[512];
    snprintf(path, sizeof path, "%s.ids", argv[2]);
    FILE *f = fopen(path, "r");
    int ids[512], n = 0;
    while (n < 512 && fscanf(f, "%d", &ids[n]) == 1) n++;
    fclose(f);
    float *lg = NULL;
    t0 = now();
    if (getenv("QUARRY_PER_TOKEN")) for (int i = 0; i < n; i++) lg = model_forward(&m, ids[i], i);
    else lg = model_prefill(&m, ids, n, 0);
    double dt = now() - t0;
    int nv = m.c.n_vocab;
    float *ref = malloc(nv * 4);
    snprintf(path, sizeof path, "%s.logits", argv[2]);
    f = fopen(path, "rb");
    fread(ref, 4, nv, f);
    fclose(f);
    double dot = 0, na = 0, nb = 0, maxd = 0;
    for (int i = 0; i < nv; i++) {
        dot += (double)lg[i] * ref[i]; na += (double)lg[i] * lg[i]; nb += (double)ref[i] * ref[i];
        double d = fabs(lg[i] - ref[i]); if (d > maxd) maxd = d;
    }
    int a[5], b[5];
    top5(lg, nv, a);
    top5(ref, nv, b);
    printf("%d tokens in %.1fs (%.2f tok/s, experts read %.2f GB)\n", n, dt, n / dt, m.expert_bytes_read / 1e9);
    printf("cosine %.6f   max |diff| %.3f\n", dot / sqrt(na * nb), maxd);
    printf("top5 quarry: %d %d %d %d %d\ntop5 llama : %d %d %d %d %d\n", a[0], a[1], a[2], a[3], a[4], b[0], b[1], b[2], b[3], b[4]);
    printf("%s\n", a[0] == b[0] && dot / sqrt(na * nb) > 0.99 ? "MATCH" : "MISMATCH");
    return 0;
}
