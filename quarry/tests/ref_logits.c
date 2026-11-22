// Reference: tokenize a prompt with llama.cpp, run it, save token ids + last logits.
// Build: gcc -O2 ref_logits.c -o ref_logits -lllama -lggml -lggml-base
#include <llama.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s model prompt out_prefix\n", argv[0]); return 1; }
    llama_backend_init();
    struct llama_model *model = llama_model_load_from_file(argv[1], llama_model_default_params());
    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512; cp.n_threads = 4; cp.n_threads_batch = 4;
    struct llama_context *ctx = llama_init_from_model(model, cp);
    const char *p = argv[2];
    int n = -llama_tokenize(vocab, p, strlen(p), NULL, 0, true, true);
    llama_token *toks = malloc(n * sizeof *toks);
    llama_tokenize(vocab, p, strlen(p), toks, n, true, true);
    if (llama_decode(ctx, llama_batch_get_one(toks, n))) return 1;
    const float *lg = llama_get_logits_ith(ctx, -1);
    int nv = llama_vocab_n_tokens(vocab);
    char path[512];
    snprintf(path, sizeof path, "%s.ids", argv[3]);
    FILE *f = fopen(path, "w");
    for (int i = 0; i < n; i++) fprintf(f, "%d\n", toks[i]);
    fclose(f);
    snprintf(path, sizeof path, "%s.logits", argv[3]);
    f = fopen(path, "wb");
    fwrite(lg, sizeof(float), nv, f);
    fclose(f);
    printf("%d tokens, vocab %d\n", n, nv);
    return 0;
}
