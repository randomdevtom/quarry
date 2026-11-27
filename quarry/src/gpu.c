// Quarry GPU expert tier: see gpu.h.
#include "gpu.h"
#include <stdio.h>

#ifndef QUARRY_GPU
// built without GPU support: everything stays on the CPU
qgpu *gpu_open(int n_layer, int n_expert, int n_used, int D, int F, const int *types, double gb) {
    (void)n_layer; (void)n_expert; (void)n_used; (void)D; (void)F; (void)types; (void)gb;
    return NULL;
}
void gpu_close(qgpu *g) { (void)g; }
int gpu_slot(const qgpu *g, int l, int e) { (void)g; (void)l; (void)e; return -1; }
void gpu_offer(qgpu *g, int l, int e, const uint8_t *a, const uint8_t *b, const uint8_t *c) { (void)g; (void)l; (void)e; (void)a; (void)b; (void)c; }
void gpu_moe_start(qgpu *g, int l, int T, const float *const *X, int n, const int *slot, const float *w) { (void)g; (void)l; (void)T; (void)X; (void)n; (void)slot; (void)w; }
void gpu_moe_finish(qgpu *g, float *const *out) { (void)g; (void)out; }
void gpu_stats(const qgpu *g, char *buf, int len) { (void)g; if (len) buf[0] = 0; }
#else

#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#define MAX_NODES 96

struct qgpu {
    ggml_backend_t be;
    struct ggml_context *wctx;       // the weight tensors: [layer * 3 + part]
    ggml_backend_buffer_t wbuf;
    struct ggml_tensor **W;
    int n_layer, n_expert, n_used, D, F, cap;
    int *slot_of;                    // [layer * n_expert + e] -> slot, or -1
    int *owner;                      // [layer * cap + slot] -> expert, or -1
    float *score;                    // [layer * n_expert + e] decayed use count
    uint32_t *seen;                  // [layer * n_expert + e] tick of the last score update
    uint32_t tick;                   // ~ tokens
    int last_l, uploads_tick, n_resident;
    ggml_gallocr_t ga;
    uint8_t *cmem;
    size_t cmem_size;
    struct ggml_context *cctx;       // graph of the running call
    struct ggml_tensor *out;
    int T, busy;
    float *tmp;
    // the GPU call runs on its own thread, so the CPU keeps computing even if the backend's
    // "async" compute blocks until the GPU is done
    pthread_t th;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    int job, done, quit;             // job: a call is waiting; done: its result is in tmp
    int jl, jn;
    float *jx;                       // [2][D]
    int jslot[64];
    float jw[64];
    uint64_t calls, experts_gpu, uploads, evictions;
    double wait_s;
    char name[128];
};

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

// score with exponential decay (half-life ~64 tokens), updated lazily
static float score_now(qgpu *g, int key) {
    uint32_t dt = g->tick - g->seen[key];
    if (dt) { g->score[key] *= powf(0.5f, dt / 64.0f); g->seen[key] = g->tick; }
    return g->score[key];
}

static ggml_backend_dev_t pick_device(void) {
    const char *want = getenv("QUARRY_GPU_DEVICE");   // a name like "Vulkan0" or "CUDA1"
    ggml_backend_dev_t best = NULL, igpu = NULL;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        enum ggml_backend_dev_type t = ggml_backend_dev_type(d);
        if (want) { if (!strcmp(ggml_backend_dev_name(d), want)) return d; continue; }
        if (t == GGML_BACKEND_DEVICE_TYPE_GPU && !best) best = d;
        if (t == GGML_BACKEND_DEVICE_TYPE_IGPU && !igpu) igpu = d;
    }
    // an integrated GPU shares RAM with the expert cache; only on request (QUARRY_GPU_GB)
    if (!best && igpu && getenv("QUARRY_GPU_GB")) best = igpu;
    return best;
}

// the graph for layer l: out[D, T] = sum_j w[j, t] * down_j(silu(gate_j x_t) * up_j x_t)
static struct ggml_cgraph *build(qgpu *g, int l, int T, int n, struct ggml_tensor **px, struct ggml_tensor **pids,
                                 struct ggml_tensor **pw) {
    struct ggml_init_params ip = {g->cmem_size, g->cmem, true};
    g->cctx = ggml_init(ip);
    struct ggml_context *c = g->cctx;
    struct ggml_tensor *x = ggml_new_tensor_3d(c, GGML_TYPE_F32, g->D, 1, T);
    struct ggml_tensor *ids = ggml_new_tensor_2d(c, GGML_TYPE_I32, n, T);
    struct ggml_tensor *w = ggml_new_tensor_3d(c, GGML_TYPE_F32, 1, n, T);
    ggml_set_input(x); ggml_set_input(ids); ggml_set_input(w);
    struct ggml_tensor *gt = ggml_mul_mat_id(c, g->W[l * 3 + 0], x, ids);   // [F, n, T]
    struct ggml_tensor *up = ggml_mul_mat_id(c, g->W[l * 3 + 1], x, ids);
    struct ggml_tensor *a = ggml_swiglu_split(c, gt, up);
    struct ggml_tensor *d = ggml_mul_mat_id(c, g->W[l * 3 + 2], a, ids);    // [D, n, T]
    d = ggml_mul(c, d, w);
    struct ggml_tensor *s = ggml_view_2d(c, d, g->D, T, d->nb[2], 0);
    for (int j = 1; j < n; j++) s = ggml_add(c, s, ggml_view_2d(c, d, g->D, T, d->nb[2], j * d->nb[1]));
    if (n == 1) s = ggml_cont(c, s);
    ggml_set_output(s);
    struct ggml_cgraph *gf = ggml_new_graph_custom(c, MAX_NODES, false);
    ggml_build_forward_expand(gf, s);
    g->out = s;
    *px = x; *pids = ids; *pw = w;
    return gf;
}

static void *worker(void *arg);

qgpu *gpu_open(int n_layer, int n_expert, int n_used, int D, int F, const int *types, double gb) {
    const char *off = getenv("QUARRY_GPU");
    if (off && !strcmp(off, "0")) return NULL;
    ggml_backend_load_all();   // no-op unless ggml was built with dynamic backends
    ggml_backend_dev_t dev = pick_device();
    if (!dev) return NULL;
    size_t vfree = 0, vtotal = 0;
    ggml_backend_dev_memory(dev, &vfree, &vtotal);
    if (gb <= 0) gb = ((double)vfree - 1.0e9) / 1e9;   // leave ~1 GB for the driver and the graph
    // bytes of one expert summed over all layers: the per-layer slot count is uniform
    double per = 0;
    for (int l = 0; l < n_layer; l++)
        per += ggml_row_size(types[l * 3 + 0], D) * F + ggml_row_size(types[l * 3 + 1], D) * F + ggml_row_size(types[l * 3 + 2], F) * D;
    int cap = (int)(gb * 1e9 / per);
    if (cap > n_expert) cap = n_expert;
    if (cap < 1) { fprintf(stderr, "quarry: GPU %s has too little free memory, not used\n", ggml_backend_dev_name(dev)); return NULL; }

    qgpu *g = calloc(1, sizeof *g);
    g->n_layer = n_layer; g->n_expert = n_expert; g->n_used = n_used; g->D = D; g->F = F;
    snprintf(g->name, sizeof g->name, "%s (%s)", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev));
    g->be = ggml_backend_dev_init(dev, NULL);
    if (!g->be) { free(g); return NULL; }
    struct ggml_init_params wp = {ggml_tensor_overhead() * (n_layer * 3 + 4), NULL, true};
    g->wctx = ggml_init(wp);
    g->W = calloc(n_layer * 3, sizeof *g->W);
    // allocate; on failure (fragmented or over-reported VRAM) retry smaller
    for (;;) {
        for (int l = 0; l < n_layer; l++) {
            g->W[l * 3 + 0] = ggml_new_tensor_3d(g->wctx, types[l * 3 + 0], D, F, cap);
            g->W[l * 3 + 1] = ggml_new_tensor_3d(g->wctx, types[l * 3 + 1], D, F, cap);
            g->W[l * 3 + 2] = ggml_new_tensor_3d(g->wctx, types[l * 3 + 2], F, D, cap);
        }
        g->wbuf = ggml_backend_alloc_ctx_tensors(g->wctx, g->be);
        if (g->wbuf || cap <= 1) break;
        ggml_free(g->wctx);
        g->wctx = ggml_init(wp);
        cap = cap * 3 / 4;
    }
    if (!g->wbuf) { fprintf(stderr, "quarry: could not allocate GPU memory, GPU not used\n"); gpu_close(g); return NULL; }
    g->cap = cap;
    g->slot_of = malloc(sizeof(int) * n_layer * n_expert);
    for (int i = 0; i < n_layer * n_expert; i++) g->slot_of[i] = -1;
    g->owner = malloc(sizeof(int) * n_layer * cap);
    for (int i = 0; i < n_layer * cap; i++) g->owner[i] = -1;
    g->score = calloc(n_layer * n_expert, sizeof(float));
    g->seen = calloc(n_layer * n_expert, sizeof(uint32_t));
    g->last_l = n_layer;
    g->cmem_size = ggml_tensor_overhead() * MAX_NODES + ggml_graph_overhead_custom(MAX_NODES, false);
    g->cmem = malloc(g->cmem_size);
    g->tmp = malloc(sizeof(float) * D * 2);
    g->jx = malloc(sizeof(float) * D * 2);
    // reserve the compute buffer for the largest call (2 tokens, all experts on the GPU)
    g->ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(g->be));
    struct ggml_tensor *x, *ids, *w;
    struct ggml_cgraph *gf = build(g, 0, 2, n_used, &x, &ids, &w);
    ggml_gallocr_reserve(g->ga, gf);
    ggml_free(g->cctx); g->cctx = NULL;
    pthread_mutex_init(&g->mu, NULL);
    pthread_cond_init(&g->cv, NULL);
    pthread_create(&g->th, NULL, worker, g);
    fprintf(stderr, "quarry: GPU %s: %d of %d experts per layer in VRAM (%.1f GB)\n",
            g->name, cap, n_expert, per * cap / 1e9);
    return g;
}

void gpu_close(qgpu *g) {
    if (!g) return;
    if (g->th) {
        pthread_mutex_lock(&g->mu); g->quit = 1; pthread_cond_broadcast(&g->cv); pthread_mutex_unlock(&g->mu);
        pthread_join(g->th, NULL);
    }
    if (g->ga) ggml_gallocr_free(g->ga);
    if (g->wbuf) ggml_backend_buffer_free(g->wbuf);
    if (g->wctx) ggml_free(g->wctx);
    if (g->be) ggml_backend_free(g->be);
    free(g->W); free(g->slot_of); free(g->owner); free(g->score); free(g->seen); free(g->cmem); free(g->tmp); free(g->jx);
    free(g);
}

int gpu_slot(const qgpu *g, int l, int e) { return g ? g->slot_of[l * g->n_expert + e] : -1; }

void gpu_offer(qgpu *g, int l, int e, const uint8_t *gate, const uint8_t *up, const uint8_t *down) {
    if (!g || g->busy) return;
    if (l <= g->last_l) { g->tick++; g->uploads_tick = 0; }   // layers restart: next token
    g->last_l = l;
    const int key = l * g->n_expert + e;
    if (g->slot_of[key] >= 0) return;
    const float sc = score_now(g, key) + 1;
    g->score[key] = sc;
    // a free slot, or the coldest resident expert if this one is clearly hotter
    int victim = -1;
    float vs = 1e30f;
    for (int s = 0; s < g->cap; s++) {
        const int o = g->owner[l * g->cap + s];
        if (o < 0) { victim = s; vs = -1; break; }
        const float os = score_now(g, l * g->n_expert + o);
        if (os < vs) { vs = os; victim = s; }
    }
    if (victim < 0) return;
    if (vs >= 0) {
        if (sc < 2 * vs + 2) return;              // hysteresis: no thrashing
        if (g->uploads_tick >= 8) return;         // bound the copying per token once VRAM is full
    } else if (g->uploads_tick >= 96) return;     // first fill: faster, still bounded
    const uint8_t *src[3] = {gate, up, down};
    for (int p = 0; p < 3; p++) {
        struct ggml_tensor *t = g->W[l * 3 + p];
        ggml_backend_tensor_set(t, src[p], (size_t)victim * t->nb[2], t->nb[2]);
    }
    const int old = g->owner[l * g->cap + victim];
    if (old >= 0) { g->slot_of[l * g->n_expert + old] = -1; g->evictions++; } else g->n_resident++;
    g->owner[l * g->cap + victim] = e;
    g->slot_of[key] = victim;
    g->uploads++;
    g->uploads_tick++;
}

static void run_job(qgpu *g) {
    struct ggml_tensor *x, *ids, *wt;
    struct ggml_cgraph *gf = build(g, g->jl, g->T, g->jn, &x, &ids, &wt);
    if (!ggml_gallocr_alloc_graph(g->ga, gf)) { fprintf(stderr, "quarry: GPU graph allocation failed\n"); abort(); }
    ggml_backend_tensor_set(x, g->jx, 0, sizeof(float) * g->D * g->T);
    ggml_backend_tensor_set(ids, g->jslot, 0, sizeof(int) * g->jn * g->T);
    ggml_backend_tensor_set(wt, g->jw, 0, sizeof(float) * g->jn * g->T);
    ggml_backend_graph_compute(g->be, gf);
    ggml_backend_tensor_get(g->out, g->tmp, 0, sizeof(float) * g->D * g->T);
    ggml_free(g->cctx);
    g->cctx = NULL;
}

static void *worker(void *arg) {
    qgpu *g = arg;
    pthread_mutex_lock(&g->mu);
    for (;;) {
        while (!g->job && !g->quit) pthread_cond_wait(&g->cv, &g->mu);
        if (g->quit) break;
        g->job = 0;
        pthread_mutex_unlock(&g->mu);
        run_job(g);
        pthread_mutex_lock(&g->mu);
        g->done = 1;
        pthread_cond_broadcast(&g->cv);
    }
    pthread_mutex_unlock(&g->mu);
    return NULL;
}

void gpu_moe_start(qgpu *g, int l, int T, const float *const *X, int n, const int *slot, const float *w) {
    for (int t = 0; t < T; t++) memcpy(g->jx + (size_t)t * g->D, X[t], sizeof(float) * g->D);
    memcpy(g->jslot, slot, sizeof(int) * n * T);
    memcpy(g->jw, w, sizeof(float) * n * T);
    // used on the GPU: keeps their scores up, so hot experts stay
    for (int i = 0; i < n * T; i++)
        if (w[i] > 0) {
            const int key = l * g->n_expert + g->owner[l * g->cap + slot[i]];
            g->score[key] = score_now(g, key) + 1;
            g->experts_gpu++;
        }
    pthread_mutex_lock(&g->mu);
    g->jl = l; g->jn = n; g->T = T;
    g->done = 0; g->job = 1; g->busy = 1;
    pthread_cond_broadcast(&g->cv);
    pthread_mutex_unlock(&g->mu);
    g->calls++;
}

void gpu_moe_finish(qgpu *g, float *const *out) {
    double t0 = now_s();
    pthread_mutex_lock(&g->mu);
    while (!g->done) pthread_cond_wait(&g->cv, &g->mu);
    pthread_mutex_unlock(&g->mu);
    g->wait_s += now_s() - t0;
    for (int t = 0; t < g->T; t++) {
        const float *s = g->tmp + (size_t)t * g->D;
        float *o = out[t];
        for (int i = 0; i < g->D; i++) o[i] += s[i];
    }
    g->busy = 0;
}

void gpu_stats(const qgpu *g, char *buf, int len) {
    if (!g) { if (len) buf[0] = 0; return; }
    snprintf(buf, len, "gpu %s: %d/%d experts resident, %llu run on GPU, %llu copied in (%llu replaced), %.1fs waited",
             g->name, g->n_resident, g->n_layer * g->cap, (unsigned long long)g->experts_gpu,
             (unsigned long long)g->uploads, (unsigned long long)g->evictions, g->wait_s);
}
#endif
