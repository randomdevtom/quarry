// Quarry expert cache.
//  - fixed slab of slots, one expert (gate+up+down) per slot, LRU eviction
//  - experts read with O_DIRECT (bypass the page cache: we decide what stays)
//  - background I/O threads; demand loads jump the queue, prefetches go last
//  - slots used by the current layer are pinned so prefetch can't evict them
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "model.h"

#define ALIGN 4096
enum { S_EMPTY = 0, S_LOADING, S_READY };

typedef struct {
    int key;                    // layer * n_expert + expert, -1 if empty
    int state, pins;
    int prev, next;             // LRU list (head = most recent)
    uint8_t *mem;
    const uint8_t *p[3];        // gate, up, down inside mem
    int prefetched, used;       // stats: loaded by prefetch, later used
} slot_t;

struct expert_store {
    qmodel *m;
    int fd_direct;
    int n_slots, n_keys;
    slot_t *slots;
    int *map;                   // key -> slot or -1
    int head, tail;             // LRU
    uint64_t part_bytes[3];
    uint64_t slot_bytes;
    // I/O queue (ring of slot ids)
    int *q, q_cap, q_head, q_len;
    pthread_mutex_t mu;
    pthread_cond_t cv_work, cv_done;
    pthread_t threads[4];
    int n_threads;
    // stats
    uint64_t hits, misses, prefetch_issued, prefetch_useful;
    double wait_s;              // time the compute thread spent blocked on I/O
};

#include <time.h>
static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

static void lru_unlink(struct expert_store *s, int i) {
    slot_t *x = &s->slots[i];
    if (x->prev >= 0) s->slots[x->prev].next = x->next; else s->head = x->next;
    if (x->next >= 0) s->slots[x->next].prev = x->prev; else s->tail = x->prev;
    x->prev = x->next = -1;
}
static void lru_push_front(struct expert_store *s, int i) {
    slot_t *x = &s->slots[i];
    x->prev = -1;
    x->next = s->head;
    if (s->head >= 0) s->slots[s->head].prev = i;
    s->head = i;
    if (s->tail < 0) s->tail = i;
}

static uint64_t part_off(struct expert_store *s, int key, int part) {
    const qlayer *L = &s->m->layers[key / s->m->c.n_expert];
    const qmat *M = part == 0 ? &L->gate_exps : part == 1 ? &L->up_exps : &L->down_exps;
    return M->file_off + (uint64_t)(key % s->m->c.n_expert) * s->part_bytes[part];
}

// read [off, off+len) with O_DIRECT into an aligned window; returns pointer to data
static const uint8_t *read_direct(struct expert_store *s, uint8_t *dst, uint64_t off, uint64_t len) {
    uint64_t a0 = off & ~(uint64_t)(ALIGN - 1);
    uint64_t a1 = (off + len + ALIGN - 1) & ~(uint64_t)(ALIGN - 1);
    uint64_t n = a1 - a0, done = 0;
    const uint64_t need_end = off + len - a0;   // bytes of the window we actually need
    while (done < need_end) {
        ssize_t r = pread(s->fd_direct, dst + done, n - done, a0 + done);
        if (r < 0 && errno == EINTR) continue;
        if (r == 0 && a0 + done >= s->m->g.file_size) break;  // aligned window ran past EOF
        if (r <= 0) {
            // file tail may not be a multiple of ALIGN: finish with buffered read
            ssize_t r2 = pread(s->m->g.fd, dst + done, n - done, a0 + done);
            if (r2 <= 0) { perror("expert read"); exit(1); }
            r = r2;
        }
        done += r;
    }
    return dst + (off - a0);
}

static void *io_worker(void *arg) {
    struct expert_store *s = arg;
    pthread_mutex_lock(&s->mu);
    for (;;) {
        while (!s->q_len) pthread_cond_wait(&s->cv_work, &s->mu);
        int i = s->q[s->q_head];
        s->q_head = (s->q_head + 1) % s->q_cap;
        s->q_len--;
        int key = s->slots[i].key;
        pthread_mutex_unlock(&s->mu);
        uint8_t *mem = s->slots[i].mem;
        const uint8_t *p[3];
        uint64_t stride = s->slot_bytes / 3;
        for (int k = 0; k < 3; k++)
            p[k] = read_direct(s, mem + k * stride, part_off(s, key, k), s->part_bytes[k]);
        pthread_mutex_lock(&s->mu);
        for (int k = 0; k < 3; k++) s->slots[i].p[k] = p[k];
        s->slots[i].state = S_READY;
        s->m->expert_bytes_read += s->part_bytes[0] + s->part_bytes[1] + s->part_bytes[2];
        s->m->expert_loads++;
        pthread_cond_broadcast(&s->cv_done);
    }
    return NULL;
}

// enqueue slot i: front (demand) or back (prefetch). caller holds mu.
static void enqueue(struct expert_store *s, int i, int front) {
    if (front) {
        s->q_head = (s->q_head - 1 + s->q_cap) % s->q_cap;
        s->q[s->q_head] = i;
    } else {
        s->q[(s->q_head + s->q_len) % s->q_cap] = i;
    }
    s->q_len++;
    pthread_cond_signal(&s->cv_work);
}

// find a victim slot (LRU tail that is not pinned or loading). caller holds mu.
static int evict(struct expert_store *s) {
    for (int i = s->tail; i >= 0; i = s->slots[i].prev) {
        slot_t *x = &s->slots[i];
        if (x->pins || x->state == S_LOADING) continue;
        if (x->key >= 0) s->map[x->key] = -1;
        x->key = -1;
        x->state = S_EMPTY;
        return i;
    }
    return -1;
}

// make sure key is resident or loading; returns slot or -1 (no free slot)
static int request(struct expert_store *s, int key, int demand) {
    int i = s->map[key];
    if (i >= 0) {
        if (demand && s->slots[i].state == S_LOADING) {
            // promote: it may sit at the back of the queue as a prefetch
            for (int k = 0; k < s->q_len; k++) {
                int qi = (s->q_head + k) % s->q_cap;
                if (s->q[qi] == i) {
                    for (int j = k; j > 0; j--) {
                        int a = (s->q_head + j) % s->q_cap, b = (s->q_head + j - 1) % s->q_cap;
                        s->q[a] = s->q[b];
                    }
                    s->q[s->q_head] = i;
                    break;
                }
            }
        }
        return i;
    }
    i = evict(s);
    if (i < 0) return -1;
    slot_t *x = &s->slots[i];
    x->key = key;
    x->state = S_LOADING;
    x->prefetched = !demand;
    x->used = 0;
    s->map[key] = i;
    lru_unlink(s, i);
    lru_push_front(s, i);
    enqueue(s, i, demand);
    if (!demand) s->prefetch_issued++;
    return i;
}

struct expert_store *experts_open(qmodel *m) {
    struct expert_store *s = calloc(1, sizeof *s);
    s->m = m;
    const qlayer *L = &m->layers[0];
    s->part_bytes[0] = L->gate_exps.row_bytes * m->c.n_ff_exp;
    s->part_bytes[1] = L->up_exps.row_bytes * m->c.n_ff_exp;
    s->part_bytes[2] = L->down_exps.row_bytes * m->c.n_embd;
    uint64_t maxp = 0;
    for (int k = 0; k < 3; k++) if (s->part_bytes[k] > maxp) maxp = s->part_bytes[k];
    uint64_t stride = (maxp + 2 * ALIGN + ALIGN - 1) / ALIGN * ALIGN;   // window + alignment slack
    s->slot_bytes = 3 * stride;
    double gb = getenv("QUARRY_CACHE_GB") ? atof(getenv("QUARRY_CACHE_GB")) : 5.5;
    s->n_keys = m->c.n_layer * m->c.n_expert;
    s->n_slots = (int)(gb * 1e9 / s->slot_bytes);
    if (s->n_slots > s->n_keys) s->n_slots = s->n_keys;
    if (s->n_slots < 2 * m->c.n_expert_used) s->n_slots = 2 * m->c.n_expert_used;
    s->slots = calloc(s->n_slots, sizeof(slot_t));
    s->map = malloc(s->n_keys * sizeof(int));
    for (int k = 0; k < s->n_keys; k++) s->map[k] = -1;
    uint8_t *slab = aligned_alloc(ALIGN, (uint64_t)s->n_slots * s->slot_bytes);
    if (!slab) { fprintf(stderr, "cannot allocate %.2f GB expert cache\n", s->n_slots * s->slot_bytes / 1e9); exit(1); }
    s->head = s->tail = -1;
    for (int i = 0; i < s->n_slots; i++) {
        s->slots[i] = (slot_t){.key = -1, .prev = -1, .next = -1, .mem = slab + (uint64_t)i * s->slot_bytes};
        lru_push_front(s, i);
    }
    s->q_cap = s->n_slots + 8;
    s->q = malloc(s->q_cap * sizeof(int));
    char p[64];
    snprintf(p, sizeof p, "/proc/self/fd/%d", m->g.fd);
    s->fd_direct = open(p, O_RDONLY | O_DIRECT);
    if (s->fd_direct < 0) { perror("open O_DIRECT"); s->fd_direct = m->g.fd; }
    pthread_mutex_init(&s->mu, NULL);
    pthread_cond_init(&s->cv_work, NULL);
    pthread_cond_init(&s->cv_done, NULL);
    s->n_threads = getenv("QUARRY_IO_THREADS") ? atoi(getenv("QUARRY_IO_THREADS")) : 2;
    for (int t = 0; t < s->n_threads; t++) pthread_create(&s->threads[t], NULL, io_worker, s);
    fprintf(stderr, "quarry: expert cache %d slots (%.2f GB), %d io threads\n", s->n_slots,
            s->n_slots * s->slot_bytes / 1e9, s->n_threads);
    return s;
}

// ---- API used by the forward pass ----

// start loading this layer's selected experts (demand) and pin them
void experts_begin_layer(struct expert_store *s, int l, const int *sel, int k) {
    pthread_mutex_lock(&s->mu);
    for (int j = 0; j < k; j++) {
        int key = l * s->m->c.n_expert + sel[j];
        int i = s->map[key];
        if (i >= 0 && s->slots[i].state == S_READY) s->hits++; else s->misses++;
        i = request(s, key, 1);
        if (i >= 0) s->slots[i].pins++;
    }
    pthread_mutex_unlock(&s->mu);
}

void experts_prefetch(struct expert_store *s, int l, const int *sel, int k) {
    pthread_mutex_lock(&s->mu);
    for (int j = 0; j < k; j++) request(s, l * s->m->c.n_expert + sel[j], 0);
    pthread_mutex_unlock(&s->mu);
}

int experts_ready(struct expert_store *s, int l, int e) {
    pthread_mutex_lock(&s->mu);
    int i = s->map[l * s->m->c.n_expert + e];
    int r = i >= 0 && s->slots[i].state == S_READY;
    pthread_mutex_unlock(&s->mu);
    return r;
}

// blocking get; also unpins when done=1 is called via experts_release
void experts_get(struct expert_store *s, int l, int e, const uint8_t **gate, const uint8_t **up,
                 const uint8_t **down) {
    int key = l * s->m->c.n_expert + e;
    pthread_mutex_lock(&s->mu);
    int i = s->map[key];
    if (i < 0) i = request(s, key, 1);   // not requested via begin_layer
    if (s->slots[i].state != S_READY) {
        double t0 = now_s();
        while (s->slots[i].state != S_READY) pthread_cond_wait(&s->cv_done, &s->mu);
        s->wait_s += now_s() - t0;
    }
    slot_t *x = &s->slots[i];
    if (x->prefetched && !x->used) s->prefetch_useful++;
    x->used = 1;
    lru_unlink(s, i);
    lru_push_front(s, i);
    *gate = x->p[0]; *up = x->p[1]; *down = x->p[2];
    pthread_mutex_unlock(&s->mu);
}

void experts_release(struct expert_store *s, int l, int e) {
    pthread_mutex_lock(&s->mu);
    int i = s->map[l * s->m->c.n_expert + e];
    if (i >= 0 && s->slots[i].pins > 0) s->slots[i].pins--;
    pthread_mutex_unlock(&s->mu);
}

void experts_stats(struct expert_store *s, uint64_t *hits, uint64_t *misses, uint64_t *pf_issued,
                   uint64_t *pf_useful) {
    *hits = s->hits; *misses = s->misses; *pf_issued = s->prefetch_issued; *pf_useful = s->prefetch_useful;
}

double experts_wait_seconds(struct expert_store *s) { return s->wait_s; }

void experts_reset_stats(struct expert_store *s) {
    s->wait_s = 0;
    s->hits = s->misses = s->prefetch_issued = s->prefetch_useful = 0;
}
