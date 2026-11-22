// Record MoE routing (selected experts per layer per token) using libllama's
// eval callback. Generates greedily so the trace reflects real decoding.
// Output lines: <token_pos> <layer> <e0> ... <e{k-1}>
//
// Build: gcc -O2 trace.c -o trace -lllama -lggml -lggml-base
// Run:   ./trace model.gguf "prompt" n_gen out.txt
#include <ggml-backend.h>
#include <ggml.h>
#include <llama.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    FILE *out;
    int pos0;   // position of first token in the current batch
} trace_ctx;

static bool cb(struct ggml_tensor *t, bool ask, void *ud) {
    if (strncmp(t->name, "ffn_moe_topk-", 13) != 0) return ask ? false : true;
    if (ask) return true;  // yes, we want this tensor's data
    trace_ctx *tc = ud;
    int layer = atoi(t->name + 13);
    int k = (int)t->ne[0], n_tok = (int)t->ne[1];
    // copy out (tensor may live in a backend buffer; view with strides)
    size_t nbytes = ggml_nbytes(t);
    char *buf = malloc(nbytes);
    ggml_backend_tensor_get(t, buf, 0, nbytes);
    for (int j = 0; j < n_tok; j++) {
        fprintf(tc->out, "%d %d", tc->pos0 + j, layer);
        for (int i = 0; i < k; i++) {
            int32_t e = *(int32_t *)(buf + j * t->nb[1] + i * t->nb[0] - 0);
            fprintf(tc->out, " %d", e);
        }
        fputc('\n', tc->out);
    }
    free(buf);
    return true;
}

int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s model prompt n_gen out\n", argv[0]); return 1; }
    const char *prompt = argv[2];
    int n_gen = atoi(argv[3]);
    trace_ctx tc = {fopen(argv[4], "w"), 0};

    llama_backend_init();
    struct llama_model_params mp = llama_model_default_params();
    struct llama_model *model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 1;
    const struct llama_vocab *vocab = llama_model_get_vocab(model);

    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 4096;
    cp.n_batch = 512;
    cp.n_threads = 4;
    cp.n_threads_batch = 4;
    cp.cb_eval = cb;
    cp.cb_eval_user_data = &tc;
    struct llama_context *ctx = llama_init_from_model(model, cp);

    int n_prompt = -llama_tokenize(vocab, prompt, strlen(prompt), NULL, 0, true, true);
    llama_token *toks = malloc(sizeof(llama_token) * (n_prompt + n_gen + 1));
    llama_tokenize(vocab, prompt, strlen(prompt), toks, n_prompt, true, true);

    struct llama_sampler *smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    struct llama_batch batch = llama_batch_get_one(toks, n_prompt);
    tc.pos0 = 0;
    if (llama_decode(ctx, batch)) { fprintf(stderr, "decode failed\n"); return 1; }
    int pos = n_prompt;
    for (int i = 0; i < n_gen; i++) {
        llama_token id = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, id)) break;
        char piece[256];
        int n = llama_token_to_piece(vocab, id, piece, sizeof piece, 0, true);
        fwrite(piece, 1, n > 0 ? n : 0, stderr);
        toks[pos] = id;
        batch = llama_batch_get_one(&toks[pos], 1);
        tc.pos0 = pos;
        if (llama_decode(ctx, batch)) break;
        pos++;
    }
    fprintf(stderr, "\n[prompt %d tokens, generated %d]\n", n_prompt, pos - n_prompt);
    fclose(tc.out);
    return 0;
}
