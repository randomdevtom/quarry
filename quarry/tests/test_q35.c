// Qwen3.5/3.6 MoE: compare Quarry's last-token logits with llama.cpp's
// (from build/ref_logits model "prompt" prefix -> prefix.ids, prefix.logits)
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "../src/qwen35.h"
int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s model prefix\n", argv[0]); return 1; }
    q35_model m;
    if (q35_load(&m, argv[1], 4096)) return 1;
    char path[512];
    snprintf(path, sizeof path, "%s.ids", argv[2]);
    FILE *f = fopen(path, "r");
    int ids[4096], n = 0;
    while (n < 4096 && fscanf(f, "%d", &ids[n]) == 1) n++;
    fclose(f);
    float *lg = NULL;
    lg = getenv("BATCH") ? q35_prefill(&m, ids, n, 0) : NULL;
    if (!lg) for (int i = 0; i < n; i++) lg = q35_forward(&m, ids[i], i);
    snprintf(path, sizeof path, "%s.logits", argv[2]);
    f = fopen(path, "rb");
    float *ref = malloc(sizeof(float) * m.c.n_vocab);
    if (fread(ref, sizeof(float), m.c.n_vocab, f) != (size_t)m.c.n_vocab) { fprintf(stderr, "short ref\n"); return 1; }
    double dot = 0, na = 0, nb = 0, md = 0;
    int ta = 0, tb = 0;
    for (int i = 0; i < m.c.n_vocab; i++) {
        dot += (double)lg[i] * ref[i]; na += (double)lg[i] * lg[i]; nb += (double)ref[i] * ref[i];
        if (fabs(lg[i] - ref[i]) > md) md = fabs(lg[i] - ref[i]);
        if (lg[i] > lg[ta]) ta = i;
        if (ref[i] > ref[tb]) tb = i;
    }
    printf("%d tokens: cosine %.6f  max |diff| %.3f  top quarry %d ref %d  %s\n", n, dot / sqrt(na * nb), md, ta, tb, ta == tb ? "MATCH" : "DIFFERENT");
    return 0;
}
