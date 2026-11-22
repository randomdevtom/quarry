// quarry: CPU MoE inference with an SSD-backed expert cache.
//
//   quarry run   -m model.gguf [-p "prompt"] [-n 512] [--temp 0.7]
//   quarry serve -m model.gguf [--port 8090] [--host 127.0.0.1]
//
// Env: QUARRY_CACHE_GB (expert cache size, default 5.5), QUARRY_IO_THREADS (2)
#define _GNU_SOURCE
#include <arpa/inet.h>
#include <llama.h>
#include <math.h>
#include <netinet/in.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#include "inkling.h"
#include "model.h"
#include "qwen35.h"
#include "chat_html.h"   // generated from src/chat.html by the Makefile

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

// ---------------- engine state (single sequence, prefix-reusing) ----------------

typedef struct {
    int ink;                           // 1: Inkling 2, 0: Qwen3-MoE
    int q35;                           // 1: Qwen3.5/3.6 MoE (gated delta net + attention)
    qmodel m;
    ink_model im;
    q35_model qm;
    void *snap; int snap_pos;          // q35: recurrent state saved at the end of the last prompt
    int t_think_end, think, think_default;   // q35: "</think>" token; thinking on for this request / by default
    // Inkling special tokens and output state (0 header, 1 thinking, 2 answer text)
    int t_think, t_text, t_end_msg, t_msg_model, mode, show_thinking;
    float effort;
    struct llama_model *tok_model;     // vocab-only libllama model: tokenizer
    const struct llama_vocab *vocab;
    int *ctx_tokens, n_ctx_tokens;     // tokens whose KV is in the cache
    int n_ctx;
    unsigned long long rng;
} engine;

static void quiet_log(enum ggml_log_level level, const char *text, void *ud) { (void)level; (void)text; (void)ud; }

static int eng_load(engine *e, const char *path, int n_ctx) {
    memset(e, 0, sizeof *e);
    e->n_ctx = n_ctx;
    llama_backend_init();
    llama_log_set(quiet_log, NULL);
    struct llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    e->tok_model = llama_model_load_from_file(path, mp);
    if (!e->tok_model) return -1;
    e->vocab = llama_model_get_vocab(e->tok_model);
    e->q35 = q35_detect(path);
    e->ink = !e->q35 && ink_detect(path);
    // Qwen MoE default: 6 of the 8 routed experts per token. Same HumanEval score in our tests
    // (Qwen3-Coder 19/21, Qwen3.8 distill 20/21), 10-25% faster. `--experts 8` restores the exact model.
    if (!e->ink && !getenv("QUARRY_TOPK")) setenv("QUARRY_TOPK", "6", 1);
    if (e->q35) {
        if (q35_load(&e->qm, path, n_ctx)) return -1;
        e->snap = malloc(q35_state_bytes(&e->qm));
        int t[8];
        e->t_think_end = llama_tokenize(e->vocab, "</think>", 8, t, 8, false, true) == 1 ? t[0] : -1;
    } else if (e->ink ? ink_load(&e->im, path, n_ctx) : model_load(&e->m, path, n_ctx)) return -1;
    if (e->ink) {
        int t[8];
        const char *names[4] = {"<|content_thinking|>", "<|content_text|>", "<|end_message|>", "<|message_model|>"};
        int *dst[4] = {&e->t_think, &e->t_text, &e->t_end_msg, &e->t_msg_model};
        for (int i = 0; i < 4; i++)
            *dst[i] = llama_tokenize(e->vocab, names[i], strlen(names[i]), t, 8, false, true) == 1 ? t[0] : -1;
    }
    e->effort = 0.2f;
    e->ctx_tokens = malloc(sizeof(int) * n_ctx);
    e->rng = (unsigned long long)time(NULL) * 2654435761ULL;
    return 0;
}

static int tokenize(engine *e, const char *text, int *out, int cap) {
    int n = llama_tokenize(e->vocab, text, strlen(text), out, cap, false, true);
    return n < 0 ? -1 : n;
}

static int piece(engine *e, int tok, char *buf, int cap) {
    int n = llama_token_to_piece(e->vocab, tok, buf, cap - 1, 0, false);
    if (n < 0) n = 0;
    buf[n] = 0;
    return n;
}

static int is_eog(engine *e, int tok) { return llama_vocab_is_eog(e->vocab, tok); }

static int n_vocab(engine *e) { return e->q35 ? e->qm.c.n_vocab : e->ink ? e->im.c.n_vocab : e->m.c.n_vocab; }
static uint64_t bytes_read(engine *e) { return e->q35 ? e->qm.expert_bytes_read : e->ink ? e->im.expert_bytes_read : e->m.expert_bytes_read; }
static struct expert_store *eng_experts(engine *e) { return e->q35 ? e->qm.experts : e->ink ? e->im.experts : e->m.experts; }
static float *fwd(engine *e, int tok, int pos) {
    return e->q35 ? q35_forward(&e->qm, tok, pos) : e->ink ? ink_forward(&e->im, tok, pos) : model_forward(&e->m, tok, pos);
}
static float *prefill(engine *e, const int *toks, int n, int pos0) {
    if (e->q35) return q35_prefill(&e->qm, toks, n, pos0);
    if (!e->ink) return model_prefill(&e->m, toks, n, pos0);
    float *lg = NULL;
    for (int i = 0; i < n; i++) lg = ink_forward(&e->im, toks[i], pos0 + i);
    return lg;
}

static float frand(engine *e) {
    e->rng ^= e->rng << 13; e->rng ^= e->rng >> 7; e->rng ^= e->rng << 17;
    return (e->rng >> 11) * (1.0 / 9007199254740992.0);
}

static int sample(engine *e, float *lg, float temp, float top_p) {
    int nv = n_vocab(e);
    int best = 0;
    for (int i = 1; i < nv; i++) if (lg[i] > lg[best]) best = i;
    if (temp <= 0) return best;
    // top-p over the top 64 candidates (plenty for code/chat)
    enum { K = 64 };
    int idx[K] = {0};
    float p[K];
    int n = 0;
    for (int i = 0; i < nv; i++) {
        if (n < K || lg[i] > lg[idx[n - 1]]) {
            int j = n < K ? n++ : K - 1;
            while (j > 0 && lg[idx[j - 1]] < lg[i]) { idx[j] = idx[j - 1]; j--; }
            idx[j] = i;
        }
    }
    float mx = lg[idx[0]], tot = 0;
    for (int j = 0; j < n; j++) { p[j] = expf((lg[idx[j]] - mx) / temp); tot += p[j]; }
    float cum = 0;
    int cut = n;
    for (int j = 0; j < n; j++) { cum += p[j] / tot; if (cum >= top_p) { cut = j + 1; break; } }
    float r = frand(e) * cum, acc = 0;
    for (int j = 0; j < cut; j++) { acc += p[j] / tot; if (acc >= r) return idx[j]; }
    return idx[0];
}

// feed prompt tokens, reusing the KV of the longest common prefix
static float *eng_prefill(engine *e, const int *toks, int n) {
    int common = 0;
    while (common < n && common < e->n_ctx_tokens && e->ctx_tokens[common] == toks[common]) common++;
    if (common == n) common = n - 1;   // need logits for the last token
    if (e->q35) {
        // the delta-net state can't be rewound: continue from the current end, or from the
        // snapshot taken at the end of the last prompt, or start over
        if (!(common == e->n_ctx_tokens && common > 0)) {
            if (e->snap_pos > 0 && common >= e->snap_pos) { q35_state_load(&e->qm, e->snap); common = e->snap_pos; }
            else {
                q35_state_reset(&e->qm);
                common = 0;
            }
        }
        float *lg = prefill(e, toks + common, n - common, common);
        for (int i = common; i < n; i++) e->ctx_tokens[i] = toks[i];
        e->n_ctx_tokens = n;
        q35_state_save(&e->qm, e->snap);
        e->snap_pos = n;
        return lg;
    }
    float *lg = prefill(e, toks + common, n - common, common);
    for (int i = common; i < n; i++) e->ctx_tokens[i] = toks[i];
    e->n_ctx_tokens = n;
    return lg;
}

// generation callback: return nonzero to stop
typedef int (*emit_fn)(const char *text, void *ud);

// print one generated token (thinking/answer handling); nonzero to stop
static int emit_tok(engine *e, int tok, int g, char *buf, int cap, emit_fn emit, void *ud) {
    int stop = 0;
    if (e->q35 && tok == e->t_think_end && e->mode == 1) {
        // end of the thinking block: what follows is the answer
        if (e->show_thinking) stop = emit("\033[0m\n", ud);
        e->mode = 2;
    } else if (e->q35 && e->mode == 1) {
        piece(e, tok, buf, cap);
        if (e->show_thinking) stop = emit(g == 0 ? "\033[2m" : "", ud) || emit(buf, ud);
    } else if (e->ink && (tok == e->t_think || tok == e->t_text || tok == e->t_end_msg || tok == e->t_msg_model)) {
        // Inkling structure tokens: switch between thinking and answer, print nothing
        const int was = e->mode;
        e->mode = tok == e->t_think ? 1 : tok == e->t_text ? 2 : 0;
        if (e->show_thinking && was == 1 && e->mode != 1) stop = emit("\033[0m\n", ud);
        if (e->show_thinking && e->mode == 1) stop = emit("\033[2m", ud);
    } else {
        piece(e, tok, buf, cap);
        if (e->mode == 2) stop = emit(buf, ud);
        else if (e->mode == 1 && e->show_thinking) stop = emit(buf, ud);
    }
    return stop;
}

// probability of `tok` under sample()'s distribution (temperature, top-p over the top 64)
static float sample_prob(engine *e, const float *lg, float temp, float top_p, int tok) {
    enum { K = 64 };
    int idx[K], n = 0;
    const int nv = n_vocab(e);
    for (int i = 0; i < nv; i++) {
        if (n < K || lg[i] > lg[idx[n - 1]]) {
            int j = n < K ? n++ : K - 1;
            while (j > 0 && lg[idx[j - 1]] < lg[i]) { idx[j] = idx[j - 1]; j--; }
            idx[j] = i;
        }
    }
    float p[K], tot = 0, mx = lg[idx[0]];
    for (int j = 0; j < n; j++) { p[j] = expf((lg[idx[j]] - mx) / temp); tot += p[j]; }
    float cum = 0;
    int cut = n;
    for (int j = 0; j < n; j++) { cum += p[j] / tot; if (cum >= top_p) { cut = j + 1; break; } }
    for (int j = 0; j < cut; j++) if (idx[j] == tok) return p[j] / tot / cum;
    return 0;
}

typedef struct { int prompt_tokens, gen_tokens; double t_prompt, t_gen; } gen_stats;

static void eng_generate(engine *e, const char *prompt, int max_tokens, float temp, float top_p,
                         emit_fn emit, void *ud, gen_stats *st) {
    int *toks = malloc(sizeof(int) * e->n_ctx);
    int n = tokenize(e, prompt, toks, e->n_ctx);
    if (n < 0 || n >= e->n_ctx - 8) { emit("[prompt too long]", ud); free(toks); return; }
    if (max_tokens > e->n_ctx - n) max_tokens = e->n_ctx - n;
    double t0 = now();
    float *lg = eng_prefill(e, toks, n);
    st->prompt_tokens = n;
    st->t_prompt = now() - t0;
    t0 = now();
    char buf[256];
    int pos = n, g = 0;
    e->mode = e->ink ? 0 : e->q35 && e->think ? 1 : 2;
    // Qwen3.5/3.6 with an MTP layer: draft one token, verify two at a time (output unchanged:
    // greedy keeps the draft only if it is the argmax; sampling uses the speculative rule)
    const int spec = e->q35 && e->qm.has_mtp && !getenv("QUARRY_NO_SPEC");
    while (g < max_tokens) {
        int tok = sample(e, lg, temp, top_p);
        if (is_eog(e, tok)) break;
        int stop = emit_tok(e, tok, g, buf, sizeof buf, emit, ud);
        g++;
        if (stop) break;
        e->ctx_tokens[pos] = tok;
        if (spec && g < max_tokens && pos + 2 < e->n_ctx) {
            const int d = q35_draft(&e->qm, tok, pos);
            q35_forward2(&e->qm, tok, d, pos);
            float *l0 = e->qm.logits2[0], *l1 = e->qm.logits2[1];
            int acc;
            if (temp <= 0) {
                int best = 0;
                for (int i = 1; i < n_vocab(e); i++) if (l0[i] > l0[best]) best = i;
                acc = best == d;
            } else {
                acc = frand(e) < sample_prob(e, l0, temp, top_p, d);
                if (!acc) l0[d] = -INFINITY;   // resample from the rest
            }
            q35_commit(&e->qm, acc);
            if (acc) {
                e->ctx_tokens[pos + 1] = d;
                pos += 2;
                e->n_ctx_tokens = pos;
                lg = l1;
                if (is_eog(e, d)) break;
                stop = emit_tok(e, d, g, buf, sizeof buf, emit, ud);
                g++;
                if (stop) break;
            } else {
                pos += 1;
                e->n_ctx_tokens = pos;
                lg = l0;
            }
        } else {
            lg = fwd(e, tok, pos++);
            e->n_ctx_tokens = pos;
        }
    }
    st->gen_tokens = g;
    st->t_gen = now() - t0;
    free(toks);
}

static char *chat_q35(const char **roles, const char **contents, const char **tcalls, int n, const char *tools, int think);

// chat prompt from role/content pairs: ChatML for Qwen, Inkling's own format otherwise
static engine *g_eng;
static char *chatml(const char **roles, const char **contents, int n) {
    size_t len = 256;
    for (int i = 0; i < n; i++) len += strlen(roles[i]) + strlen(contents[i]) + 96;
    char *p = malloc(len), *w = p;
    if (g_eng && g_eng->ink) {
        // the thinking-effort line goes before the first non-system message
        int effort_done = 0;
        for (int i = 0; i < n; i++) {
            const int sys = !strcmp(roles[i], "system"), asst = !strcmp(roles[i], "assistant");
            if (!effort_done && !sys) {
                w += sprintf(w, "<|message_system|><|content_text|>Thinking effort level: %g<|end_message|>", g_eng->effort);
                effort_done = 1;
            }
            const char *tag = sys ? "<|message_system|>" : asst ? "<|message_model|>" : !strcmp(roles[i], "tool") ? "<|message_tool|>" : "<|message_user|>";
            w += sprintf(w, "%s<|content_text|>%s<|end_message|>%s", tag, contents[i], asst ? "<|content_model_end_sampling|>" : "");
        }
        if (!effort_done) sprintf(w, "<|message_system|><|content_text|>Thinking effort level: %g<|end_message|>", g_eng->effort);
        return p;
    }
    for (int i = 0; i < n; i++) w += sprintf(w, "<|im_start|>%s\n%s<|im_end|>\n", roles[i], contents[i]);
    sprintf(w, "<|im_start|>assistant\n");
    return p;
}

// ---------------- run (CLI) ----------------

typedef struct { char *buf; size_t len; } acc_t;

// print and keep a copy (for the conversation history)
static int emit_cli(const char *t, void *ud) {
    acc_t *a = ud;
    size_t l = strlen(t);
    // keep only the answer for the conversation history, not the thinking
    if ((!g_eng || g_eng->mode == 2) && a->len + l < (1 << 20) - 1) { memcpy(a->buf + a->len, t, l); a->len += l; }
    fputs(t, stdout);
    fflush(stdout);
    return 0;
}

static int cmd_run(engine *e, const char *prompt, int n, float temp) {
    char line[16384];
    const char *roles[64], *contents[64];
    int nm = 0;
    int interactive = prompt == NULL;
    for (;;) {
        if (interactive) {
            printf("\n> ");
            fflush(stdout);
            if (!fgets(line, sizeof line, stdin)) break;
            line[strcspn(line, "\n")] = 0;
            if (!line[0]) continue;
            prompt = strdup(line);
        }
        if (nm >= 62) nm = 0;
        roles[nm] = "user"; contents[nm++] = prompt;
        const char *tcnull[64] = {0};
        e->think = e->think_default;
        char *p = e->q35 ? chat_q35(roles, contents, tcnull, nm, NULL, e->think) : chatml(roles, contents, nm);
        char *reply = calloc(1, 1 << 20);
        gen_stats st = {0};
        acc_t acc = {reply, 0};
        e->show_thinking = 1;
        memset(q35_prof, 0, sizeof q35_prof);
        eng_generate(e, p, n, temp, 0.9f, emit_cli, &acc, &st);
        e->show_thinking = 0;
        fprintf(stderr, "\n[%d prompt tokens %.1fs | %d tokens %.2f tok/s]\n", st.prompt_tokens, st.t_prompt,
                st.gen_tokens, st.gen_tokens / (st.t_gen > 0 ? st.t_gen : 1));
        if (getenv("QUARRY_STATS")) {
            struct expert_store *es = eng_experts(e);
            uint64_t h, mi, pi, pu;
            experts_stats(es, &h, &mi, &pi, &pu);
            fprintf(stderr, "[experts: hit %.1f%%, prefetch %llu issued / %llu used, io wait %.1fs, SSD %.2f GB",
                    100.0 * h / (h + mi + 1e-9), (unsigned long long)pi, (unsigned long long)pu, experts_wait_seconds(es), bytes_read(e) / 1e9);
            if (e->ink) fprintf(stderr, " | prediction %.0f%% (layer-start guess %.0f%%) | attn %.1fs moe %.1fs output %.1fs mix %.1fs",
                                100.0 * e->im.pred_correct / (e->im.pred_total + 1e-9),
                                100.0 * e->im.pred2_correct / (e->im.pred_total + 1e-9), ink_prof[0], ink_prof[1], ink_prof[2], ink_prof[3]);
            if (e->ink && e->im.skipped) fprintf(stderr, " | skipped %llu experts", (unsigned long long)e->im.skipped);
            if (e->q35 && e->qm.spec_steps) fprintf(stderr, " | MTP: %llu steps, %.0f%% drafts accepted, draft %.1fs",
                                (unsigned long long)e->qm.spec_steps, 100.0 * e->qm.spec_accepted / e->qm.spec_steps, e->qm.draft_seconds);
            if (e->q35) fprintf(stderr, " | prediction %.0f%% | delta-net %.1fs attn %.1fs route %.1fs experts %.1fs shared %.1fs output %.1fs (delta-net proj %.1fs, conv+recur %.1fs)",
                                100.0 * e->qm.pred_correct / (e->qm.pred_total + 1e-9), q35_prof[0], q35_prof[1], q35_prof[2], q35_prof[3], q35_prof[4], q35_prof[5], q35_prof[6], q35_prof[7]);
            if (e->ink) fprintf(stderr, " | ctrl %.1fs proj %.1fs small %.1fs core %.1fs", ink_prof[4], ink_prof[5], ink_prof[6], ink_prof[7]);
            fprintf(stderr, "]\n");
        }
        roles[nm] = "assistant"; contents[nm++] = reply;
        free(p);
        if (!interactive) break;
    }
    return 0;
}

// ---------------- serve (OpenAI-compatible) ----------------

// minimal JSON helpers: enough for chat-completion requests
static const char *skip_ws(const char *s) { while (*s == ' ' || *s == '\n' || *s == '\r' || *s == '\t') s++; return s; }

// parse a JSON string at s (pointing at the opening quote); returns malloc'd value, sets *end
static char *json_str(const char *s, const char **end) {
    if (*s != '"') return NULL;
    s++;
    size_t cap = 256, n = 0;
    char *o = malloc(cap);
    while (*s && *s != '"') {
        if (n + 8 >= cap) o = realloc(o, cap *= 2);
        if (*s == '\\') {
            s++;
            switch (*s) {
            case 'n': o[n++] = '\n'; break; case 't': o[n++] = '\t'; break;
            case 'r': o[n++] = '\r'; break; case 'b': o[n++] = '\b'; break;
            case 'f': o[n++] = '\f'; break;
            case 'u': {
                unsigned cp = strtoul((char[]){s[1], s[2], s[3], s[4], 0}, NULL, 16);
                s += 4;
                if (cp >= 0xD800 && cp <= 0xDBFF && s[1] == '\\' && s[2] == 'u') {
                    unsigned lo = strtoul((char[]){s[3], s[4], s[5], s[6], 0}, NULL, 16);
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    s += 6;
                }
                if (cp < 0x80) o[n++] = cp;
                else if (cp < 0x800) { o[n++] = 0xC0 | cp >> 6; o[n++] = 0x80 | (cp & 63); }
                else if (cp < 0x10000) { o[n++] = 0xE0 | cp >> 12; o[n++] = 0x80 | ((cp >> 6) & 63); o[n++] = 0x80 | (cp & 63); }
                else { o[n++] = 0xF0 | cp >> 18; o[n++] = 0x80 | ((cp >> 12) & 63); o[n++] = 0x80 | ((cp >> 6) & 63); o[n++] = 0x80 | (cp & 63); }
                break;
            }
            default: o[n++] = *s;
            }
            s++;
        } else o[n++] = *s++;
    }
    o[n] = 0;
    if (end) *end = *s == '"' ? s + 1 : s;
    return o;
}

// skip any JSON value
static const char *json_skip(const char *s) {
    s = skip_ws(s);
    if (*s == '"') { const char *e; free(json_str(s, &e)); return e; }
    if (*s == '{' || *s == '[') {
        int depth = 0;
        while (*s) {
            if (*s == '"') { const char *e; free(json_str(s, &e)); s = e; continue; }
            if (*s == '{' || *s == '[') depth++;
            if (*s == '}' || *s == ']') { depth--; if (!depth) return s + 1; }
            s++;
        }
        return s;
    }
    while (*s && *s != ',' && *s != '}' && *s != ']') s++;
    return s;
}

// find key at top level of object starting at s ('{'); returns pointer to value or NULL
static const char *json_get(const char *s, const char *key) {
    s = skip_ws(s);
    if (*s != '{') return NULL;
    s++;
    for (;;) {
        s = skip_ws(s);
        if (*s != '"') return NULL;
        const char *e;
        char *k = json_str(s, &e);
        s = skip_ws(e);
        if (*s == ':') s = skip_ws(s + 1);
        int match = !strcmp(k, key);
        free(k);
        if (match) return s;
        s = skip_ws(json_skip(s));
        if (*s == ',') s++; else return NULL;
    }
}

// message content: string, or array of parts with "text"
static char *content_text(const char *v) {
    v = skip_ws(v);
    if (*v == '"') return json_str(v, NULL);
    char *out = calloc(1, 1);
    if (*v != '[') return out;
    v++;
    for (;;) {
        v = skip_ws(v);
        if (*v != '{') break;
        const char *t = json_get(v, "text");
        if (t && *skip_ws(t) == '"') {
            char *s = json_str(skip_ws(t), NULL);
            out = realloc(out, strlen(out) + strlen(s) + 1);
            strcat(out, s);
            free(s);
        }
        v = skip_ws(json_skip(v));
        if (*v == ',') v++; else break;
    }
    return out;
}

static void json_escape(const char *s, char *o, size_t cap) {
    size_t n = 0;
    for (; *s && n + 8 < cap; s++) {
        unsigned char c = *s;
        if (c == '"' || c == '\\') { o[n++] = '\\'; o[n++] = c; }
        else if (c == '\n') { o[n++] = '\\'; o[n++] = 'n'; }
        else if (c == '\r') { o[n++] = '\\'; o[n++] = 'r'; }
        else if (c == '\t') { o[n++] = '\\'; o[n++] = 't'; }
        else if (c < 0x20) n += snprintf(o + n, cap - n, "\\u%04x", c);
        else o[n++] = c;
    }
    o[n] = 0;
}

// ---------------- tool calling (Qwen3-Coder format) ----------------

typedef struct { char *p; size_t n, cap; } sbuf;
static void sb_putn(sbuf *b, const char *s, size_t n) {
    if (b->n + n + 1 > b->cap) { b->cap = (b->n + n + 1) * 2; b->p = realloc(b->p, b->cap); }
    memcpy(b->p + b->n, s, n);
    b->n += n;
    b->p[b->n] = 0;
}
static void sb_put(sbuf *b, const char *s) { sb_putn(b, s, strlen(s)); }

// raw JSON text of the value at v (malloc'd)
static char *json_raw(const char *v) {
    v = skip_ws(v);
    const char *e = json_skip(v);
    while (e > v && (e[-1] == ' ' || e[-1] == '\n' || e[-1] == '\r' || e[-1] == '\t')) e--;
    char *o = malloc(e - v + 1);
    memcpy(o, v, e - v);
    o[e - v] = 0;
    return o;
}
// iterate an object: *s at '{' first time, then whatever this returns; NULL at the end
static const char *json_obj_next(const char *s, char **key, const char **val) {
    s = skip_ws(s);
    if (*s == '{' || *s == ',') s = skip_ws(s + 1);
    if (*s != '"') return NULL;
    const char *e;
    *key = json_str(s, &e);
    s = skip_ws(e);
    if (*s == ':') s = skip_ws(s + 1);
    *val = s;
    return skip_ws(json_skip(s));
}
// iterate an array: *s at '[' first time; returns pointer to the element or NULL; *next set
static const char *json_arr_next(const char *s, const char **next) {
    s = skip_ws(s);
    if (*s == '[' || *s == ',') s = skip_ws(s + 1);
    if (*s == ']' || !*s) return NULL;
    *next = skip_ws(json_skip(s));
    return s;
}
static char *json_str_or_raw(const char *v) {   // Jinja's `x | string`, close enough
    v = skip_ws(v);
    return *v == '"' ? json_str(v, NULL) : json_raw(v);
}
static void put_trimmed(sbuf *b, const char *s) {
    while (*s == ' ' || *s == '\n' || *s == '\t' || *s == '\r') s++;
    size_t n = strlen(s);
    while (n && (s[n - 1] == ' ' || s[n - 1] == '\n' || s[n - 1] == '\t' || s[n - 1] == '\r')) n--;
    sb_putn(b, s, n);
}
// render_item_list() of the template: "\n<tag>[`a`, `b`]</tag>"
static void put_item_list(sbuf *b, const char *arr, const char *tag) {
    if (!arr || *skip_ws(arr) != '[') return;
    const char *nx, *it = json_arr_next(arr, &nx);
    if (!it) return;
    sb_put(b, "\n<"); sb_put(b, tag); sb_put(b, ">[");
    for (int i = 0; it; i++, it = json_arr_next(nx, &nx)) {
        if (i) sb_put(b, ", ");
        if (*it == '"') { char *x = json_str(it, NULL); sb_put(b, "`"); sb_put(b, x); sb_put(b, "`"); free(x); }
        else { char *x = json_raw(it); sb_put(b, x); free(x); }
    }
    sb_put(b, "]</"); sb_put(b, tag); sb_put(b, ">");
}
static void put_tools(sbuf *b, const char *tools) {
    sb_put(b, "\n\nYou have access to the following functions:\n\n<tools>");
    const char *nx, *t = json_arr_next(tools, &nx);
    for (; t; t = json_arr_next(nx, &nx)) {
        const char *f = json_get(t, "function");
        if (!f) f = t;
        const char *v;
        char *name = (v = json_get(f, "name")) ? json_str(skip_ws(v), NULL) : strdup("");
        char *desc = (v = json_get(f, "description")) ? json_str_or_raw(v) : strdup("");
        sb_put(b, "\n<function>\n<name>"); sb_put(b, name); sb_put(b, "</name>");
        sb_put(b, "\n<description>"); put_trimmed(b, desc); sb_put(b, "</description>");
        sb_put(b, "\n<parameters>");
        const char *params = json_get(f, "parameters"), *props = params ? json_get(params, "properties") : NULL;
        if (props && *skip_ws(props) == '{') {
            char *k; const char *pv, *it = props;
            while ((it = json_obj_next(it, &k, &pv))) {
                sb_put(b, "\n<parameter>\n<name>"); sb_put(b, k); sb_put(b, "</name>");
                const char *x;
                if ((x = json_get(pv, "type"))) { char *ty = json_str_or_raw(x); sb_put(b, "\n<type>"); sb_put(b, ty); sb_put(b, "</type>"); free(ty); }
                if ((x = json_get(pv, "description"))) { char *d = json_str_or_raw(x); sb_put(b, "\n<description>"); put_trimmed(b, d); sb_put(b, "</description>"); free(d); }
                put_item_list(b, json_get(pv, "enum"), "enum");
                if (*skip_ws(pv) == '{') {   // other keys, as <key>value</key>
                    char *k2; const char *v2, *it2 = skip_ws(pv);
                    while ((it2 = json_obj_next(it2, &k2, &v2))) {
                        if (strcmp(k2, "type") && strcmp(k2, "description") && strcmp(k2, "enum") && strcmp(k2, "required")) {
                            char *val = *skip_ws(v2) == '{' ? json_raw(v2) : json_str_or_raw(v2);
                            sb_put(b, "\n<"); sb_put(b, k2); sb_put(b, ">"); sb_put(b, val); sb_put(b, "</"); sb_put(b, k2); sb_put(b, ">");
                            free(val);
                        }
                        free(k2);
                    }
                }
                put_item_list(b, json_get(pv, "required"), "required");
                sb_put(b, "\n</parameter>");
                free(k);
            }
        }
        if (params) put_item_list(b, json_get(params, "required"), "required");
        sb_put(b, "\n</parameters>\n</function>");
        free(name); free(desc);
    }
    sb_put(b, "\n</tools>");
    sb_put(b, "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>");
}

// the Qwen3-Coder chat template (tools, assistant tool calls, tool results), as in the GGUF
static char *chat_qwen(const char **roles, const char **contents, const char **tcalls, int n, const char *tools) {
    sbuf b = {0};
    int has_tools = tools && *skip_ws(tools) == '[' && *skip_ws(skip_ws(tools) + 1) != ']';
    int i0 = 0;
    if (n > 0 && !strcmp(roles[0], "system")) {
        sb_put(&b, "<|im_start|>system\n"); sb_put(&b, contents[0]);
        i0 = 1;
    } else if (has_tools) sb_put(&b, "<|im_start|>system\nYou are Qwen, a helpful AI assistant that can interact with a computer to solve tasks.");
    if (has_tools) put_tools(&b, tools);
    if (i0 || has_tools) sb_put(&b, "<|im_end|>\n");
    for (int i = i0; i < n; i++) {
        const char *r = roles[i];
        if (!strcmp(r, "assistant") && tcalls[i] && *skip_ws(tcalls[i]) == '[') {
            sb_put(&b, "<|im_start|>assistant");
            const char *c = contents[i];
            while (*c == ' ' || *c == '\n') c++;
            if (*c) { sb_put(&b, "\n"); put_trimmed(&b, contents[i]); sb_put(&b, "\n"); }
            const char *nx, *tc = json_arr_next(tcalls[i], &nx);
            for (; tc; tc = json_arr_next(nx, &nx)) {
                const char *f = json_get(tc, "function");
                if (!f) f = tc;
                const char *v;
                char *name = (v = json_get(f, "name")) ? json_str(skip_ws(v), NULL) : strdup("");
                sb_put(&b, "\n<tool_call>\n<function="); sb_put(&b, name); sb_put(&b, ">\n");
                if ((v = json_get(f, "arguments"))) {
                    // OpenAI sends arguments as a JSON-encoded string; accept an object too
                    char *args = *skip_ws(v) == '"' ? json_str(skip_ws(v), NULL) : json_raw(v);
                    if (*skip_ws(args) == '{') {
                        char *k; const char *av, *it = skip_ws(args);
                        while ((it = json_obj_next(it, &k, &av))) {
                            char *val = json_str_or_raw(av);
                            sb_put(&b, "<parameter="); sb_put(&b, k); sb_put(&b, ">\n"); sb_put(&b, val); sb_put(&b, "\n</parameter>\n");
                            free(val); free(k);
                        }
                    }
                    free(args);
                }
                sb_put(&b, "</function>\n</tool_call>");
                free(name);
            }
            sb_put(&b, "<|im_end|>\n");
        } else if (!strcmp(r, "tool")) {
            if (i == i0 || strcmp(roles[i - 1], "tool")) sb_put(&b, "<|im_start|>user\n");
            sb_put(&b, "<tool_response>\n"); sb_put(&b, contents[i]); sb_put(&b, "\n</tool_response>\n");
            if (i == n - 1 || strcmp(roles[i + 1], "tool")) sb_put(&b, "<|im_end|>\n");
        } else {
            sb_put(&b, "<|im_start|>"); sb_put(&b, r); sb_put(&b, "\n"); sb_put(&b, contents[i]); sb_put(&b, "<|im_end|>\n");
        }
    }
    sb_put(&b, "<|im_start|>assistant\n");
    return b.p;
}

// Qwen3.5/3.6 chat template (as in the GGUF): tools as JSON lines, system text after the
// tool block, thinking off unless asked (the generation prompt closes an empty <think>)
static char *chat_q35(const char **roles, const char **contents, const char **tcalls, int n, const char *tools, int think) {
    sbuf b = {0};
    int has_tools = tools && *skip_ws(tools) == '[' && *skip_ws(skip_ws(tools) + 1) != ']';
    const int sys0 = n > 0 && !strcmp(roles[0], "system");
    if (has_tools) {
        sb_put(&b, "<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>");
        const char *nx, *t = json_arr_next(tools, &nx);
        for (; t; t = json_arr_next(nx, &nx)) { char *raw = json_raw(t); sb_put(&b, "\n"); sb_put(&b, raw); free(raw); }
        sb_put(&b, "\n</tools>");
        sb_put(&b, "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>");
        if (sys0) {
            sbuf t2 = {0}; put_trimmed(&t2, contents[0]);
            if (t2.n) { sb_put(&b, "\n\n"); sb_put(&b, t2.p); }
            free(t2.p);
        }
        sb_put(&b, "<|im_end|>\n");
    } else if (sys0) {
        sb_put(&b, "<|im_start|>system\n"); put_trimmed(&b, contents[0]); sb_put(&b, "<|im_end|>\n");
    }
    // index of the last real user query (not a wrapped tool response)
    int last_q = n - 1;
    for (int i = n - 1; i >= 0; i--) {
        if (strcmp(roles[i], "user")) continue;
        sbuf t2 = {0}; put_trimmed(&t2, contents[i]);
        int wrapped = t2.n >= 31 && !strncmp(t2.p, "<tool_response>", 15) && !strcmp(t2.p + t2.n - 16, "</tool_response>");
        free(t2.p);
        if (!wrapped) { last_q = i; break; }
    }
    for (int i = sys0; i < n; i++) {
        const char *r = roles[i];
        if (!strcmp(r, "system")) continue;   // only allowed first
        if (!strcmp(r, "user")) {
            sb_put(&b, "<|im_start|>user\n"); put_trimmed(&b, contents[i]); sb_put(&b, "<|im_end|>\n");
        } else if (!strcmp(r, "assistant")) {
            // drop any reasoning from the content; after the last query keep an (empty) think block
            const char *c = contents[i], *te = strstr(c, "</think>");
            if (te) { c = te + 8; while (*c == '\n') c++; }
            sbuf ct = {0}; put_trimmed(&ct, c);
            sb_put(&b, "<|im_start|>assistant\n");
            if (i > last_q) sb_put(&b, "<think>\n\n</think>\n\n");
            if (ct.n) sb_put(&b, ct.p);
            if (tcalls[i] && *skip_ws(tcalls[i]) == '[') {
                const char *nx, *tc = json_arr_next(tcalls[i], &nx);
                for (int k = 0; tc; tc = json_arr_next(nx, &nx), k++) {
                    const char *f = json_get(tc, "function"), *v;
                    if (!f) f = tc;
                    char *name = (v = json_get(f, "name")) ? json_str(skip_ws(v), NULL) : strdup("");
                    sb_put(&b, k ? "\n<tool_call>\n<function=" : ct.n ? "\n\n<tool_call>\n<function=" : "<tool_call>\n<function=");
                    sb_put(&b, name); sb_put(&b, ">\n");
                    if ((v = json_get(f, "arguments"))) {
                        char *args = *skip_ws(v) == '"' ? json_str(skip_ws(v), NULL) : json_raw(v);
                        if (*skip_ws(args) == '{') {
                            char *key; const char *av, *it = skip_ws(args);
                            while ((it = json_obj_next(it, &key, &av))) {
                                char *val = *skip_ws(av) == '"' ? json_str(skip_ws(av), NULL) : json_raw(av);
                                sb_put(&b, "<parameter="); sb_put(&b, key); sb_put(&b, ">\n"); sb_put(&b, val); sb_put(&b, "\n</parameter>\n");
                                free(val); free(key);
                            }
                        }
                        free(args);
                    }
                    sb_put(&b, "</function>\n</tool_call>");
                    free(name);
                }
            }
            sb_put(&b, "<|im_end|>\n");
            free(ct.p);
        } else if (!strcmp(r, "tool")) {
            if (i == sys0 || strcmp(roles[i - 1], "tool")) sb_put(&b, "<|im_start|>user");
            sb_put(&b, "\n<tool_response>\n"); put_trimmed(&b, contents[i]); sb_put(&b, "\n</tool_response>");
            if (i == n - 1 || strcmp(roles[i + 1], "tool")) sb_put(&b, "<|im_end|>\n");
        }
    }
    sb_put(&b, think ? "<|im_start|>assistant\n<think>\n" : "<|im_start|>assistant\n<think>\n\n</think>\n\n");
    return b.p;
}

// schema type of parameter `param` of function `fn` in the tools array ("" if unknown)
static char *tool_param_type(const char *tools, const char *fn, const char *param) {
    if (!tools || *skip_ws(tools) != '[') return strdup("");
    const char *nx, *t = json_arr_next(tools, &nx);
    for (; t; t = json_arr_next(nx, &nx)) {
        const char *f = json_get(t, "function"), *v;
        if (!f) f = t;
        char *name = (v = json_get(f, "name")) ? json_str(skip_ws(v), NULL) : strdup("");
        int match = !strcmp(name, fn);
        free(name);
        if (!match) continue;
        const char *params = json_get(f, "parameters"), *props = params ? json_get(params, "properties") : NULL;
        const char *pv = props ? json_get(props, param) : NULL, *ty = pv ? json_get(pv, "type") : NULL;
        return ty && *skip_ws(ty) == '"' ? json_str(skip_ws(ty), NULL) : strdup("");
    }
    return strdup("");
}

// one parameter value as JSON: typed values stay raw when they parse, everything else is a string
static void put_arg_value(sbuf *b, const char *val, const char *type) {
    const char *s = val;
    while (*s == ' ' || *s == '\n') s++;
    size_t n = strlen(s);
    while (n && (s[n - 1] == ' ' || s[n - 1] == '\n')) n--;
    char *t = strndup(s, n);
    int raw = 0;
    if (!strcmp(type, "integer") || !strcmp(type, "number")) { char *e; strtod(t, &e); raw = n && !*e; }
    else if (!strcmp(type, "boolean")) raw = !strcmp(t, "true") || !strcmp(t, "false");
    else if (!strcmp(type, "object") || !strcmp(type, "array")) raw = n && (t[0] == '{' || t[0] == '[') && !*skip_ws(json_skip(t));
    else if (!strcmp(type, "null")) raw = !strcmp(t, "null");
    if (raw) sb_put(b, t);
    else {
        size_t cap = strlen(val) * 6 + 16;
        char *esc = malloc(cap);
        json_escape(val, esc, cap);
        sb_put(b, "\""); sb_put(b, esc); sb_put(b, "\"");
        free(esc);
    }
    free(t);
}

// split generated text into content (before the first <tool_call>) and an OpenAI tool_calls
// JSON array (NULL when there are no calls)
static char *parse_tool_calls(const char *text, const char *tools, long id, size_t *content_len) {
    const char *first = strstr(text, "<tool_call>");
    *content_len = first ? (size_t)(first - text) : strlen(text);
    if (!first) return NULL;
    sbuf out = {0};
    sb_put(&out, "[");
    int k = 0;
    for (const char *p = first; (p = strstr(p, "<tool_call>")); k++) {
        const char *end = strstr(p, "</tool_call>");
        const char *fn = strstr(p, "<function=");
        if (!fn || (end && fn > end)) break;
        fn += 10;
        const char *fe = strchr(fn, '>');
        if (!fe) break;
        char *name = strndup(fn, fe - fn);
        const char *body_end = end ? end : p + strlen(p);
        sbuf args = {0};
        sb_put(&args, "{");
        int na = 0;
        for (const char *q = fe + 1; (q = strstr(q, "<parameter=")) && q < body_end;) {
            q += 11;
            const char *qe = strchr(q, '>');
            if (!qe) break;
            char *pname = strndup(q, qe - q);
            const char *vs = qe + 1, *ve = strstr(vs, "</parameter>");
            if (!ve || ve > body_end) ve = body_end;
            if (*vs == '\n') vs++;
            size_t vl = ve - vs;
            if (vl && vs[vl - 1] == '\n') vl--;
            char *val = strndup(vs, vl);
            char *ty = tool_param_type(tools, name, pname);
            size_t cap = strlen(pname) * 6 + 16;
            char *pesc = malloc(cap);
            json_escape(pname, pesc, cap);
            if (na++) sb_put(&args, ",");
            sb_put(&args, "\""); sb_put(&args, pesc); sb_put(&args, "\":");
            put_arg_value(&args, val, ty);
            free(pesc); free(ty); free(val); free(pname);
            q = ve;
        }
        sb_put(&args, "}");
        size_t cap = args.n * 2 + 16;
        char *aesc = malloc(cap), nesc[512], head[1024];
        json_escape(args.p, aesc, cap);
        json_escape(name, nesc, sizeof nesc);
        snprintf(head, sizeof head, "%s{\"index\":%d,\"id\":\"call_%ld_%d\",\"type\":\"function\",\"function\":{\"name\":\"%s\",\"arguments\":\"",
                 k ? "," : "", k, id, k, nesc);
        sb_put(&out, head); sb_put(&out, aesc); sb_put(&out, "\"}}");
        free(aesc); free(args.p); free(name);
        if (!end) break;
        p = end + 12;
    }
    sb_put(&out, "]");
    if (k == 0) { free(out.p); return NULL; }
    return out.p;
}

static void send_all(int fd, const char *b, size_t n) {
    while (n) { ssize_t w = send(fd, b, n, MSG_NOSIGNAL); if (w <= 0) return; b += w; n -= w; }
}

typedef struct { int fd; int stream; char *acc; size_t acc_len, acc_cap; int broken; long id; int tools; size_t sent; int in_call; } req_ctx;

static int send_delta(req_ctx *r, const char *t, size_t n) {
    char *tmp = strndup(t, n);
    size_t cap = n * 6 + 16;
    char *esc = malloc(cap), *msg = malloc(cap + 256);
    json_escape(tmp, esc, cap);
    int len = sprintf(msg,
                      "data: {\"id\":\"chatcmpl-%ld\",\"object\":\"chat.completion.chunk\",\"model\":\"quarry\","
                      "\"choices\":[{\"index\":0,\"delta\":{\"content\":\"%s\"},\"finish_reason\":null}]}\n\n",
                      r->id, esc);
    int bad = send(r->fd, msg, len, MSG_NOSIGNAL) <= 0;
    free(tmp); free(esc); free(msg);
    if (bad) r->broken = 1;
    return bad;
}

static int emit_http(const char *t, void *ud) {
    req_ctx *r = ud;
    size_t l = strlen(t);
    if (r->acc_len + l + 1 > r->acc_cap) r->acc = realloc(r->acc, r->acc_cap = (r->acc_cap + l) * 2);
    memcpy(r->acc + r->acc_len, t, l);
    r->acc_len += l;
    r->acc[r->acc_len] = 0;
    if (!r->stream || r->in_call) return 0;
    if (!r->tools) { size_t n = r->acc_len - r->sent; r->sent = r->acc_len; return send_delta(r, r->acc + r->acc_len - n, n); }
    // tools on: never stream the <tool_call> markup; hold back a possible start of it
    const char *tc = strstr(r->acc + r->sent, "<tool_call>");
    size_t upto = tc ? (size_t)(tc - r->acc) : r->acc_len;
    if (tc) r->in_call = 1;
    else {
        static const char tag[] = "<tool_call>";
        for (size_t k = sizeof tag - 2; k > 0; k--)
            if (r->acc_len - r->sent >= k && !strncmp(r->acc + r->acc_len - k, tag, k)) { upto = r->acc_len - k; break; }
    }
    if (upto <= r->sent) return 0;
    int bad = send_delta(r, r->acc + r->sent, upto - r->sent);
    r->sent = upto;
    return bad;
}

static void handle(engine *e, int fd) {
    // read headers + body
    size_t cap = 1 << 16, len = 0;
    char *buf = malloc(cap);
    char *body = NULL;
    long clen = 0;
    for (;;) {
        if (len + 1 >= cap) buf = realloc(buf, cap *= 2);
        ssize_t r = recv(fd, buf + len, cap - len - 1, 0);
        if (r <= 0) { free(buf); return; }
        len += r;
        buf[len] = 0;
        char *he = strstr(buf, "\r\n\r\n");
        if (he) {
            char *cl = strcasestr(buf, "content-length:");
            clen = cl ? atol(cl + 15) : 0;
            size_t need = (he + 4 - buf) + clen;
            while (len < need) {
                if (len + 1 >= cap) { size_t off = he - buf; buf = realloc(buf, cap *= 2); he = buf + off; }
                r = recv(fd, buf + len, cap - len - 1, 0);
                if (r <= 0) break;
                len += r;
                buf[len] = 0;
            }
            body = he + 4;
            break;
        }
    }
    char method[8] = {0}, path[256] = {0};
    sscanf(buf, "%7s %255s", method, path);
    const char *cors = "Access-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: *\r\n";
    char hdr[512];
    if (!strcmp(method, "OPTIONS")) {
        int n = snprintf(hdr, sizeof hdr, "HTTP/1.1 204 No Content\r\n%sContent-Length: 0\r\n\r\n", cors);
        send_all(fd, hdr, n);
    } else if (!strcmp(method, "GET") && (!strcmp(path, "/") || !strcmp(path, "/chat"))) {
        // built-in chat page (src/chat.html)
        int n = snprintf(hdr, sizeof hdr, "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nCache-Control: no-cache\r\nContent-Length: %u\r\n\r\n", src_chat_html_len);
        send_all(fd, hdr, n);
        send_all(fd, (const char *)src_chat_html, src_chat_html_len);
    } else if (!strcmp(path, "/health")) {
        const char *b = "{\"status\":\"ok\"}";
        int n = snprintf(hdr, sizeof hdr, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n%sContent-Length: %zu\r\n\r\n%s", cors, strlen(b), b);
        send_all(fd, hdr, n);
    } else if (!strcmp(path, "/v1/models")) {
        const char *b = "{\"object\":\"list\",\"data\":[{\"id\":\"quarry\",\"object\":\"model\",\"owned_by\":\"local\"}]}";
        int n = snprintf(hdr, sizeof hdr, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n%sContent-Length: %zu\r\n\r\n%s", cors, strlen(b), b);
        send_all(fd, hdr, n);
    } else if (!strcmp(path, "/v1/chat/completions") && body) {
        const char *roles[256], *contents[256], *tcalls[256];
        int nm = 0;
        const char *msgs = json_get(body, "messages");
        if (msgs && *(msgs = skip_ws(msgs)) == '[') {
            const char *v = msgs + 1;
            while (nm < 256) {
                v = skip_ws(v);
                if (*v != '{') break;
                const char *r = json_get(v, "role"), *c = json_get(v, "content");
                roles[nm] = r ? json_str(skip_ws(r), NULL) : strdup("user");
                contents[nm] = c ? content_text(c) : strdup("");
                tcalls[nm] = json_get(v, "tool_calls");
                nm++;
                v = skip_ws(json_skip(v));
                if (*v == ',') v++; else break;
            }
        }
        const char *v;
        int max_tokens = (v = json_get(body, "max_tokens")) ? atoi(v) : 1024;
        if ((v = json_get(body, "max_completion_tokens"))) max_tokens = atoi(v);
        float temp = (v = json_get(body, "temperature")) ? atof(v) : 0.7f;
        float top_p = (v = json_get(body, "top_p")) ? atof(v) : 0.9f;
        int stream = (v = json_get(body, "stream")) && !strncmp(skip_ws(v), "true", 4);
        const char *tools = json_get(body, "tools");
        if (tools && *skip_ws(tools) != '[') tools = NULL;
        // thinking (Qwen3.5/3.6): "chat_template_kwargs": {"enable_thinking": true}, or top-level
        // "enable_thinking"; default from --think
        int think = e->think_default;
        const char *ck = json_get(body, "chat_template_kwargs");
        if (ck && (v = json_get(ck, "enable_thinking"))) think = !strncmp(skip_ws(v), "true", 4);
        if ((v = json_get(body, "enable_thinking"))) think = !strncmp(skip_ws(v), "true", 4);
        e->think = think;
        char *prompt = e->ink ? chatml(roles, contents, nm)
                     : e->q35 ? chat_q35(roles, contents, tcalls, nm, tools, think)
                     : chat_qwen(roles, contents, tcalls, nm, tools);
        req_ctx rc = {fd, stream, calloc(1, 1), 0, 1, 0, (long)time(NULL), tools != NULL && !e->ink, 0, 0};
        if (stream) {
            int n = snprintf(hdr, sizeof hdr, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n%s\r\n", cors);
            send_all(fd, hdr, n);
        }
        gen_stats st = {0};
        eng_generate(e, prompt, max_tokens, temp, top_p, emit_http, &rc, &st);
        fprintf(stderr, "[req: %d msgs, %d prompt tok (%.1fs), %d gen tok, %.2f tok/s]\n", nm, st.prompt_tokens,
                st.t_prompt, st.gen_tokens, st.gen_tokens / (st.t_gen > 0 ? st.t_gen : 1));
        const char *finish = st.gen_tokens >= max_tokens ? "length" : "stop";
        size_t content_len = rc.acc_len;
        char *calls = rc.tools ? parse_tool_calls(rc.acc, tools, rc.id, &content_len) : NULL;
        if (calls) finish = "tool_calls";
        if (stream) {
            // text held back while it could have been the start of a tool call
            if (!calls && rc.sent < rc.acc_len) send_delta(&rc, rc.acc + rc.sent, rc.acc_len - rc.sent);
            else if (calls && rc.sent < content_len) send_delta(&rc, rc.acc + rc.sent, content_len - rc.sent);
            if (calls) {
                size_t cl = strlen(calls) + 512;
                char *m = malloc(cl);
                int n = snprintf(m, cl,
                                 "data: {\"id\":\"chatcmpl-%ld\",\"object\":\"chat.completion.chunk\",\"model\":\"quarry\","
                                 "\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":%s},\"finish_reason\":null}]}\n\n",
                                 rc.id, calls);
                send_all(fd, m, n);
                free(m);
            }
            char end[512];
            int n = snprintf(end, sizeof end,
                             "data: {\"id\":\"chatcmpl-%ld\",\"object\":\"chat.completion.chunk\",\"model\":\"quarry\","
                             "\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"%s\"}],"
                             "\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d}}\n\ndata: [DONE]\n\n",
                             rc.id, finish, st.prompt_tokens, st.gen_tokens, st.prompt_tokens + st.gen_tokens);
            send_all(fd, end, n);
        } else {
            char *ctext = strndup(rc.acc, content_len);
            size_t ecap = content_len * 6 + 64;
            char *esc = malloc(ecap), *resp = malloc(ecap + (calls ? strlen(calls) : 0) + 1024);
            json_escape(ctext, esc, ecap);
            int bl = sprintf(resp,
                             "{\"id\":\"chatcmpl-%ld\",\"object\":\"chat.completion\",\"model\":\"quarry\","
                             "\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"%s\"%s%s},"
                             "\"finish_reason\":\"%s\"}],\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,"
                             "\"total_tokens\":%d}}",
                             rc.id, esc, calls ? ",\"tool_calls\":" : "", calls ? calls : "", finish,
                             st.prompt_tokens, st.gen_tokens, st.prompt_tokens + st.gen_tokens);
            int n = snprintf(hdr, sizeof hdr, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n%sContent-Length: %d\r\n\r\n", cors, bl);
            send_all(fd, hdr, n);
            send_all(fd, resp, bl);
            free(esc); free(resp); free(ctext);
        }
        free(calls);
        free(prompt);
        free(rc.acc);
        for (int i = 0; i < nm; i++) { free((void *)roles[i]); free((void *)contents[i]); }
    } else {
        const char *b = "{\"error\":\"not found\"}";
        int n = snprintf(hdr, sizeof hdr, "HTTP/1.1 404 Not Found\r\nContent-Type: application/json\r\n%sContent-Length: %zu\r\n\r\n%s", cors, strlen(b), b);
        send_all(fd, hdr, n);
    }
    free(buf);
}

static int cmd_serve(engine *e, const char *host, int port) {
    signal(SIGPIPE, SIG_IGN);
    int s = socket(AF_INET, SOCK_STREAM, 0);
    int one = 1;
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    struct sockaddr_in a = {.sin_family = AF_INET, .sin_port = htons(port)};
    inet_pton(AF_INET, host, &a.sin_addr);
    if (bind(s, (struct sockaddr *)&a, sizeof a) || listen(s, 16)) { perror("bind"); return 1; }
    fprintf(stderr, "quarry: serving OpenAI-compatible API on http://%s:%d/v1\n", host, port);
    for (;;) {
        int c = accept(s, NULL, NULL);
        if (c < 0) continue;
        handle(e, c);
        close(c);
    }
}

static void usage(void) {
    fprintf(stderr,
            "quarry - CPU MoE inference with an SSD-backed expert cache\n\n"
            "  quarry run   -m model.gguf [-p prompt] [-n max_tokens] [--temp t] [-c ctx]\n"
            "  quarry serve -m model.gguf [--host 127.0.0.1] [--port 8090] [-c ctx]\n"
            "  models: Qwen3-MoE (qwen3moe), Inkling 2 (inkling2_loop; --effort 0..0.99 thinking, default 0.2)\n"
            "  --skip 0.1   Inkling, lossy: skip experts not in RAM whose routing weight is < 10%% (faster, output changes)\n"
            "  --experts N  Qwen3-MoE experts per token (default 6 of 8; 8 = exact model)\n\n"
            "env: QUARRY_CACHE_GB (expert cache, default 5.5)  QUARRY_IO_THREADS (default 2)\n");
}

// prompt processing speed: run the same prompt twice (cold, then warm cache)
static int cmd_bench_prefill(engine *e, const char *path) {
    FILE *f = path ? fopen(path, "rb") : NULL;
    if (!f) { fprintf(stderr, "bench-prefill: -p <text file>\n"); return 1; }
    static char text[1 << 20];
    size_t len = fread(text, 1, sizeof text - 1, f);
    text[len] = 0;
    fclose(f);
    int *toks = malloc(sizeof(int) * e->n_ctx);
    int n = tokenize(e, text, toks, e->n_ctx);
    if (n <= 0 || n >= e->n_ctx) { fprintf(stderr, "prompt has %d tokens, ctx %d\n", n, e->n_ctx); return 1; }
    for (int run = 0; run < 2; run++) {
        memset(qt_pf, 0, sizeof qt_pf);
        memset(q35_prof, 0, sizeof q35_prof); memset(q35_mp, 0, sizeof q35_mp);
        uint64_t b0 = bytes_read(e);
        e->n_ctx_tokens = 0;
        double t0 = now();
        eng_prefill(e, toks, n);
        double dt = now() - t0;
        printf("%s: %d tokens in %.1fs = %.1f tok/s | SSD %.2f GB | attn mm %.1fs, attn %.1fs, route %.1fs, experts %.1fs (io wait %.1fs, gate+up %.1fs, down %.1fs)\n",
               run ? "warm" : "cold", n, dt, n / dt, (bytes_read(e) - b0) / 1e9,
               qt_pf[0], qt_pf[1], qt_pf[2], qt_pf[3], qt_pf[4], qt_pf[5], qt_pf[6]);
        if (e->q35) printf("  q35: delta-net %.1fs (proj %.1fs, conv+norm %.1fs, recur %.1fs), attention %.1fs, moe %.1fs\n", q35_prof[0], q35_prof[6], q35_prof[7], q35_prof[4], q35_prof[1], q35_prof[3]);
        if (e->q35) printf("  moe: shared %.1fs, route %.1fs, cached experts %.1fs, streamed %.1fs, stream wait %.1fs | repack (thread 0) q3k %.1fs, other %.1fs\n", q35_mp[0], q35_mp[1], q35_mp[2], q35_mp[3], q35_mp[4], q35_mp[5], q35_mp[6]);
    }
    free(toks);
    return 0;
}

static char *read_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *b = malloc(n + 1);
    b[fread(b, 1, n, f)] = 0;
    fclose(f);
    return b;
}

// debug: `quarry render req.json` prints the Qwen prompt for a chat request;
// `quarry parse-tools req.json output.txt` prints the tool_calls parsed from model output
static int cmd_debug_tools(int argc, char **argv) {
    char *body = read_file(argv[2]);
    if (!body) { perror(argv[2]); return 1; }
    const char *tools = json_get(body, "tools");
    if (tools && *skip_ws(tools) != '[') tools = NULL;
    if (!strcmp(argv[1], "parse-tools")) {
        char *out = read_file(argv[3]);
        size_t cl;
        char *calls = parse_tool_calls(out, tools, 1, &cl);
        printf("content: [%.*s]\ntool_calls: %s\n", (int)cl, out, calls ? calls : "(none)");
        return 0;
    }
    const char *roles[256], *contents[256], *tcalls[256];
    int nm = 0;
    const char *msgs = json_get(body, "messages");
    if (msgs && *(msgs = skip_ws(msgs)) == '[') {
        const char *v = msgs + 1;
        while (nm < 256) {
            v = skip_ws(v);
            if (*v != '{') break;
            const char *r = json_get(v, "role"), *c = json_get(v, "content");
            roles[nm] = r ? json_str(skip_ws(r), NULL) : strdup("user");
            contents[nm] = c ? content_text(c) : strdup("");
            tcalls[nm] = json_get(v, "tool_calls");
            nm++;
            v = skip_ws(json_skip(v));
            if (*v == ',') v++; else break;
        }
    }
    fputs(chat_qwen(roles, contents, tcalls, nm, tools), stdout);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) { usage(); return 1; }
    if ((!strcmp(argv[1], "render") && argc > 2) || (!strcmp(argv[1], "parse-tools") && argc > 3)) return cmd_debug_tools(argc, argv);
    const char *cmd = argv[1], *model = NULL, *prompt = NULL, *host = "127.0.0.1";
    int n = 1024, port = 8090, ctx = 4096;
    float temp = 0.7f, effort = 0.2f;
    int think_flag = 0;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "-p") && i + 1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-c") && i + 1 < argc) ctx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--experts") && i + 1 < argc) setenv("QUARRY_TOPK", argv[++i], 1);
        else if (!strcmp(argv[i], "--think")) think_flag = 1;
        else if (!strcmp(argv[i], "--temp") && i + 1 < argc) temp = atof(argv[++i]);
        else if (!strcmp(argv[i], "--port") && i + 1 < argc) port = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--host") && i + 1 < argc) host = argv[++i];
        else if (!strcmp(argv[i], "--effort") && i + 1 < argc) effort = atof(argv[++i]);
        else if (!strcmp(argv[i], "--skip") && i + 1 < argc) setenv("QUARRY_SKIP", argv[++i], 1);
        else { usage(); return 1; }
    }
    if (!model) { usage(); return 1; }
    // idle compute threads must yield the CPU to the I/O threads (measured: +12%).
    // libgomp reads this before main(), so re-exec once with it set.
    if (!getenv("OMP_WAIT_POLICY")) {
        setenv("OMP_WAIT_POLICY", "PASSIVE", 1);
        char self[4096];
        ssize_t len = readlink("/proc/self/exe", self, sizeof self - 1);   // real path keeps the process name
        if (len > 0) { self[len] = 0; execv(self, argv); }
        execv("/proc/self/exe", argv);
    }
    engine e;
    if (eng_load(&e, model, ctx)) { fprintf(stderr, "failed to load %s\n", model); return 1; }
    e.effort = effort;
    e.think_default = think_flag;
    g_eng = &e;
    if (!strcmp(cmd, "run")) return cmd_run(&e, prompt, n, temp);
    if (!strcmp(cmd, "serve")) return cmd_serve(&e, host, port);
    if (!strcmp(cmd, "bench-prefill")) return cmd_bench_prefill(&e, prompt);
    usage();
    return 1;
}
