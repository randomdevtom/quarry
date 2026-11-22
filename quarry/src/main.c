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

#include "model.h"

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

// ---------------- engine state (single sequence, prefix-reusing) ----------------

typedef struct {
    qmodel m;
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
    if (model_load(&e->m, path, n_ctx)) return -1;
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

static float frand(engine *e) {
    e->rng ^= e->rng << 13; e->rng ^= e->rng >> 7; e->rng ^= e->rng << 17;
    return (e->rng >> 11) * (1.0 / 9007199254740992.0);
}

static int sample(engine *e, float *lg, float temp, float top_p) {
    int nv = e->m.c.n_vocab;
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
    float *lg = NULL;
    for (int i = common; i < n; i++) {
        lg = model_forward(&e->m, toks[i], i);
        e->ctx_tokens[i] = toks[i];
    }
    e->n_ctx_tokens = n;
    return lg;
}

// generation callback: return nonzero to stop
typedef int (*emit_fn)(const char *text, void *ud);

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
    for (; g < max_tokens; g++) {
        int tok = sample(e, lg, temp, top_p);
        if (is_eog(e, tok)) break;
        piece(e, tok, buf, sizeof buf);
        if (emit(buf, ud)) { g++; break; }
        e->ctx_tokens[pos] = tok;
        lg = model_forward(&e->m, tok, pos++);
        e->n_ctx_tokens = pos;
    }
    st->gen_tokens = g;
    st->t_gen = now() - t0;
    free(toks);
}

// ChatML prompt (Qwen); messages: role/content pairs
static char *chatml(const char **roles, const char **contents, int n) {
    size_t len = 64;
    for (int i = 0; i < n; i++) len += strlen(roles[i]) + strlen(contents[i]) + 32;
    char *p = malloc(len), *w = p;
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
    if (a->len + l < (1 << 20) - 1) { memcpy(a->buf + a->len, t, l); a->len += l; }
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
        char *p = chatml(roles, contents, nm);
        char *reply = calloc(1, 1 << 20);
        gen_stats st = {0};
        acc_t acc = {reply, 0};
        eng_generate(e, p, n, temp, 0.9f, emit_cli, &acc, &st);
        fprintf(stderr, "\n[%d prompt tokens %.1fs | %d tokens %.2f tok/s]\n", st.prompt_tokens, st.t_prompt,
                st.gen_tokens, st.gen_tokens / (st.t_gen > 0 ? st.t_gen : 1));
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

static void send_all(int fd, const char *b, size_t n) {
    while (n) { ssize_t w = send(fd, b, n, MSG_NOSIGNAL); if (w <= 0) return; b += w; n -= w; }
}

typedef struct { int fd; int stream; char *acc; size_t acc_len, acc_cap; int broken; long id; } req_ctx;

static int emit_http(const char *t, void *ud) {
    req_ctx *r = ud;
    if (r->stream) {
        char esc[2048], msg[2400];
        json_escape(t, esc, sizeof esc);
        int n = snprintf(msg, sizeof msg,
                         "data: {\"id\":\"chatcmpl-%ld\",\"object\":\"chat.completion.chunk\",\"model\":\"quarry\","
                         "\"choices\":[{\"index\":0,\"delta\":{\"content\":\"%s\"},\"finish_reason\":null}]}\n\n",
                         r->id, esc);
        if (send(r->fd, msg, n, MSG_NOSIGNAL) <= 0) { r->broken = 1; return 1; }
    } else {
        size_t l = strlen(t);
        if (r->acc_len + l + 1 > r->acc_cap) r->acc = realloc(r->acc, r->acc_cap = (r->acc_cap + l) * 2);
        memcpy(r->acc + r->acc_len, t, l);
        r->acc_len += l;
        r->acc[r->acc_len] = 0;
    }
    return 0;
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
    } else if (!strcmp(path, "/health")) {
        const char *b = "{\"status\":\"ok\"}";
        int n = snprintf(hdr, sizeof hdr, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n%sContent-Length: %zu\r\n\r\n%s", cors, strlen(b), b);
        send_all(fd, hdr, n);
    } else if (!strcmp(path, "/v1/models")) {
        const char *b = "{\"object\":\"list\",\"data\":[{\"id\":\"quarry\",\"object\":\"model\",\"owned_by\":\"local\"}]}";
        int n = snprintf(hdr, sizeof hdr, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n%sContent-Length: %zu\r\n\r\n%s", cors, strlen(b), b);
        send_all(fd, hdr, n);
    } else if (!strcmp(path, "/v1/chat/completions") && body) {
        const char *roles[256], *contents[256];
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
        char *prompt = chatml(roles, contents, nm);
        req_ctx rc = {fd, stream, calloc(1, 1), 0, 1, 0, (long)time(NULL)};
        if (stream) {
            int n = snprintf(hdr, sizeof hdr, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n%s\r\n", cors);
            send_all(fd, hdr, n);
        }
        gen_stats st = {0};
        eng_generate(e, prompt, max_tokens, temp, top_p, emit_http, &rc, &st);
        fprintf(stderr, "[req: %d msgs, %d prompt tok (%.1fs), %d gen tok, %.2f tok/s]\n", nm, st.prompt_tokens,
                st.t_prompt, st.gen_tokens, st.gen_tokens / (st.t_gen > 0 ? st.t_gen : 1));
        const char *finish = st.gen_tokens >= max_tokens ? "length" : "stop";
        if (stream) {
            char end[512];
            int n = snprintf(end, sizeof end,
                             "data: {\"id\":\"chatcmpl-%ld\",\"object\":\"chat.completion.chunk\",\"model\":\"quarry\","
                             "\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"%s\"}]}\n\ndata: [DONE]\n\n",
                             rc.id, finish);
            send_all(fd, end, n);
        } else {
            size_t ecap = rc.acc_len * 2 + 64;
            char *esc = malloc(ecap), *resp = malloc(ecap + 1024);
            json_escape(rc.acc, esc, ecap);
            int bl = sprintf(resp,
                             "{\"id\":\"chatcmpl-%ld\",\"object\":\"chat.completion\",\"model\":\"quarry\","
                             "\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"%s\"},"
                             "\"finish_reason\":\"%s\"}],\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,"
                             "\"total_tokens\":%d}}",
                             rc.id, esc, finish, st.prompt_tokens, st.gen_tokens, st.prompt_tokens + st.gen_tokens);
            int n = snprintf(hdr, sizeof hdr, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n%sContent-Length: %d\r\n\r\n", cors, bl);
            send_all(fd, hdr, n);
            send_all(fd, resp, bl);
            free(esc); free(resp);
        }
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
            "  quarry serve -m model.gguf [--host 127.0.0.1] [--port 8090] [-c ctx]\n\n"
            "env: QUARRY_CACHE_GB (expert cache, default 5.5)  QUARRY_IO_THREADS (default 2)\n");
}

int main(int argc, char **argv) {
    if (argc < 2) { usage(); return 1; }
    const char *cmd = argv[1], *model = NULL, *prompt = NULL, *host = "127.0.0.1";
    int n = 1024, port = 8090, ctx = 4096;
    float temp = 0.7f;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "-p") && i + 1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-c") && i + 1 < argc) ctx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--temp") && i + 1 < argc) temp = atof(argv[++i]);
        else if (!strcmp(argv[i], "--port") && i + 1 < argc) port = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--host") && i + 1 < argc) host = argv[++i];
        else { usage(); return 1; }
    }
    if (!model) { usage(); return 1; }
    // idle compute threads must yield the CPU to the I/O threads (measured: +12%).
    // libgomp reads this before main(), so re-exec once with it set.
    if (!getenv("OMP_WAIT_POLICY")) {
        setenv("OMP_WAIT_POLICY", "PASSIVE", 1);
        execv("/proc/self/exe", argv);
    }
    engine e;
    if (eng_load(&e, model, ctx)) { fprintf(stderr, "failed to load %s\n", model); return 1; }
    if (!strcmp(cmd, "run")) return cmd_run(&e, prompt, n, temp);
    if (!strcmp(cmd, "serve")) return cmd_serve(&e, host, port);
    usage();
    return 1;
}
