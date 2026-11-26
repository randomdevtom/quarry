// Reference from llama.cpp: tokenize a prompt, run it, save token ids, last logits and
// selected intermediate tensors of the last token (out_prefix.<name>.f32), for layer-by-layer
// comparison. Build against the llama.cpp in vendor/ (see Makefile target ref).
#include <ggml-backend.h>
#include <llama.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *g_prefix;
static const char *keep[] = {"inp_embd", "norm-", "attn_norm-", "kq_soft_max-", "kqv_out-", "l_out-", "attn_out-", "ffn_inp-", "ffn_out-", "result_norm", "Qcur-", "Kcur-", "Vcur-", "Rcur-", "kq_b-", "ffn_norm-", "attn_gated-", "attn_o_proj-", "ffn_moe_out-", "ffn_shexp-", "attn_gate_softplus-", "Qcur_rope-", "Kcur_rope-", "Qcur_normed-", NULL};

static int wanted(const char *name) {
    for (int i = 0; keep[i]; i++) if (!strncmp(name, keep[i], strlen(keep[i]))) return 1;
    return 0;
}

static bool cb(struct ggml_tensor *t, bool ask, void *ud) {
    (void)ud;
    if (ask) return wanted(t->name);
    if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) return true;
    int64_t rows = t->ne[3] > 1 ? t->ne[3] : t->ne[2] > 1 ? t->ne[2] : t->ne[1];
    size_t row = ggml_nbytes(t) / rows;
    char *buf = malloc(row);
    ggml_backend_tensor_get(t, buf, (rows - 1) * row, row);
    char path[512];
    snprintf(path, sizeof path, "%s.%s.f32", g_prefix, t->name);
    FILE *f = fopen(path, "wb");
    fwrite(buf, 1, row, f);
    fclose(f);
    free(buf);
    return true;
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s model prompt out_prefix\n", argv[0]); return 1; }
    g_prefix = argv[3];
    llama_backend_init();
    struct llama_model *model = llama_model_load_from_file(argv[1], llama_model_default_params());
    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512; cp.n_threads = 4; cp.n_threads_batch = 4;
    cp.cb_eval = cb;
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
    int best = 0;
    for (int i = 1; i < nv; i++) if (lg[i] > lg[best]) best = i;
    char piece[64] = {0};
    llama_token_to_piece(vocab, best, piece, sizeof piece - 1, 0, true);
    printf("%d tokens, vocab %d, top token %d '%s'\n", n, nv, best, piece);
    return 0;
}
