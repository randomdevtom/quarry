// quarry-bench: run a tokenized prompt, then greedy-generate N tokens.
// Reports generation speed, expert cache hit rate, prefetch accuracy, SSD bytes.
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../src/model.h"

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

extern double qt_prof[8];
int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s model prompt.ids n_gen [out_ids]\n", argv[0]); return 1; }
    qmodel m;
    if (model_load(&m, argv[1], 1024)) return 1;
    FILE *f = fopen(argv[2], "r");
    int ids[1024], n = 0;
    while (n < 512 && fscanf(f, "%d", &ids[n]) == 1) n++;
    fclose(f);
    int n_gen = atoi(argv[3]);
    double t0 = now();
    float *lg = NULL;
    for (int i = 0; i < n; i++) lg = model_forward(&m, ids[i], i);
    double t_prompt = now() - t0;
    uint64_t b0 = m.expert_bytes_read;
    experts_reset_stats(m.experts);
    for (int i = 0; i < 8; i++) qt_prof[i] = 0;
    m.pred_total = m.pred_correct = 0;
    t0 = now();
    int pos = n, gen = 0;
    for (; gen < n_gen; gen++) {
        int best = 0;
        for (int v = 1; v < m.c.n_vocab; v++) if (lg[v] > lg[best]) best = v;
        if (best == 151645 || best == 151643) break;  // <|im_end|> / <|endoftext|>
        ids[pos] = best;
        lg = model_forward(&m, best, pos++);
        if (gen % 16 == 15) { fprintf(stderr, "  %d tokens, %.2f tok/s\n", gen + 1, (gen + 1) / (now() - t0)); }
    }
    double t_gen = now() - t0;
    uint64_t h, mi, pi, pu;
    experts_stats(m.experts, &h, &mi, &pi, &pu);
    printf("prompt: %d tokens in %.1fs\n", n, t_prompt);
    printf("generate: %d tokens in %.1fs = %.2f tok/s\n", gen, t_gen, gen / t_gen);
    printf("expert hit rate (at layer start): %.1f%%  | SSD read %.2f GB (%.0f MB/token)\n",
           100.0 * h / (h + mi), (m.expert_bytes_read - b0) / 1e9, (m.expert_bytes_read - b0) / 1e6 / gen);
    printf("prefetch: %llu issued, %llu used (%.0f%%) | next-layer prediction accuracy %.1f%%\n",
           (unsigned long long)pi, (unsigned long long)pu, pi ? 100.0 * pu / pi : 0,
           m.pred_total ? 100.0 * m.pred_correct / m.pred_total : 0);
    const char *nm[] = {"attn matvecs", "attn core", "router+predict", "experts (incl wait)", "output"};
    for (int i = 0; i < 5; i++) printf("  %-22s %6.1f ms/token\n", nm[i], qt_prof[i] * 1e3 / gen);
    printf("time blocked on SSD: %.1fs of %.1fs (%.0f%%)\n", experts_wait_seconds(m.experts), t_gen,
           100 * experts_wait_seconds(m.experts) / t_gen);
    if (argc > 4) {
        f = fopen(argv[4], "w");
        for (int i = n; i < pos; i++) fprintf(f, "%d\n", ids[i]);
        fclose(f);
    }
    return 0;
}
