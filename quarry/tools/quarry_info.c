// quarry-info: print model metadata and the RAM/SSD split Quarry would use.
#include <stdio.h>
#include <string.h>
#include "../src/qgguf.h"

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s model.gguf\n", argv[0]); return 1; }
    qg_file g;
    if (qg_open(&g, argv[1])) return 1;
    const char *arch = qg_get_str(&g, "general.architecture");
    printf("arch: %s  name: %s\n", arch, qg_get_str(&g, "general.name"));
    char k[128];
    const char *keys[] = {"block_count", "embedding_length", "attention.head_count",
                          "attention.head_count_kv", "attention.key_length", "expert_count",
                          "expert_used_count", "expert_feed_forward_length", "context_length"};
    for (unsigned i = 0; i < sizeof keys / sizeof *keys; i++) {
        snprintf(k, sizeof k, "%s.%s", arch, keys[i]);
        printf("  %-28s %lld\n", keys[i], (long long)qg_get_int(&g, k, -1));
    }
    snprintf(k, sizeof k, "%s.rope.freq_base", arch);
    printf("  %-28s %g\n", "rope.freq_base", qg_get_float(&g, k, -1));
    snprintf(k, sizeof k, "%s.attention.layer_norm_rms_epsilon", arch);
    printf("  %-28s %g\n", "rms_eps", qg_get_float(&g, k, -1));

    uint64_t dense = 0, experts = 0, total = 0;
    for (int j = 0; j < g.n_tensors; j++) {
        qg_tensor *t = &g.tensors[j];
        total += t->nbytes;
        if (strstr(t->name, "_exps.")) experts += t->nbytes; else dense += t->nbytes;
    }
    printf("tensors: %d  data offset: %llu\n", g.n_tensors, (unsigned long long)g.data_offset);
    printf("dense (kept in RAM): %.2f GB   experts (streamed): %.2f GB   total %.2f GB\n",
           dense / 1e9, experts / 1e9, total / 1e9);
    // sanity: last tensor must end inside the file
    qg_tensor *last = &g.tensors[0];
    for (int j = 0; j < g.n_tensors; j++) if (g.tensors[j].offset > last->offset) last = &g.tensors[j];
    printf("last tensor %s ends at %llu, file size %llu -> %s\n", last->name,
           (unsigned long long)(last->offset + last->nbytes), (unsigned long long)g.file_size,
           last->offset + last->nbytes <= g.file_size ? "OK" : "MISMATCH");
    qg_close(&g);
    return 0;
}
