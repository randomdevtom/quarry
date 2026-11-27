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
#include <sys/mman.h>
#include <sys/syscall.h>
#include <linux/aio_abi.h>
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
    int cls;                    // size class (layers whose experts have the same size)
    int urgent;                 // queued as a demand load (or promoted to one)
} slot_t;

struct expert_store {
    // what to read: per layer and part, file offset of expert 0 and bytes per expert
    int fd, n_layer, n_expert, n_parts;
    uint64_t file_size, *src_off, *src_bytes;
    uint64_t *bytes_read, *loads;
    void (*fixup)(void *ud, int layer, int part, uint8_t *data, uint64_t bytes);
    void *fixup_ud;
    int fd_direct;
    int n_slots, n_keys;
    int n_stage;                // staging slots [n_slots, n_slots + n_stage): prefill reads, never cached
    slot_t *slots;
    int *map;                   // key -> slot or -1
    int head, tail;             // LRU
    uint64_t slot_bytes;        // largest slot (staging slots use it)
    int *layer_cls;             // [layer] size class
    uint64_t *in_slot;          // [layer * n_parts + part]: offset of the part inside a slot
    uint64_t *lsz;              // [layer] slot bytes for that layer's experts
    int fd_packed;              // packed expert file (O_DIRECT), or -1
    uint64_t *pk_base;          // [layer] offset of the layer's expert 0 in the packed file
    // I/O queue (ring of slot ids)
    int *q, q_cap, q_head, q_len;
    pthread_mutex_t mu;
    pthread_cond_t cv_work, cv_done;
    pthread_t threads[4];
    int n_threads;
    int pf_inflight, pf_max;    // prefetch reads in progress; cap keeps a thread free for demand
    // stats
    uint64_t hits, misses, prefetch_issued, prefetch_useful;
    double wait_s;              // time the compute thread spent blocked on I/O
    double cache_gb;
};

// RAM the system can still give us, from /proc/meminfo (bytes), or 0
// room left under this process's cgroup v2 memory limit (bytes), or 0 if there is none
static double cgroup_room(void) {
    FILE *f = fopen("/proc/self/cgroup", "r");
    if (!f) return 0;
    char line[1024], path[1200];
    double room = 0;
    if (fgets(line, sizeof line, f) && !strncmp(line, "0::", 3)) {
        line[strcspn(line, "\n")] = 0;
        // the limit can be set on this group or any parent: take the tightest
        for (char *end = line + strlen(line); end >= line + 3; ) {
            *end = 0;
            snprintf(path, sizeof path, "/sys/fs/cgroup%s/memory.max", line + 3);
            FILE *m = fopen(path, "r");
            double mx = 0, cur = 0;
            char v[64] = "";
            if (m) { if (fgets(v, sizeof v, m) && strncmp(v, "max", 3)) mx = atof(v); fclose(m); }
            if (mx > 0) {
                snprintf(path, sizeof path, "/sys/fs/cgroup%s/memory.current", line + 3);
                if ((m = fopen(path, "r"))) { if (fgets(v, sizeof v, m)) cur = atof(v); fclose(m); }
                const double r = mx - cur;
                if (room == 0 || r < room) room = r > 1 ? r : 1;
            }
            char *slash = strrchr(line + 3, '/');
            if (!slash || slash == line + 3) break;
            end = slash;
        }
    }
    fclose(f);
    return room;
}

static double meminfo(const char *key) {
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char line[256];
    double v = 0;
    const size_t kl = strlen(key);
    while (fgets(line, sizeof line, f))
        if (!strncmp(line, key, kl) && line[kl] == ':') { v = atof(line + kl + 1) * 1024; break; }
    fclose(f);
    return v;
}

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
    const int l = key / s->n_expert, i = l * s->n_parts + part;
    return s->src_off[i] + (uint64_t)(key % s->n_expert) * s->src_bytes[i];
}
static uint64_t part_len(struct expert_store *s, int key, int part) {
    return s->src_bytes[(key / s->n_expert) * s->n_parts + part];
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
        if (r == 0 && a0 + done >= s->file_size) break;  // aligned window ran past EOF
        if (r <= 0) {
            // file tail may not be a multiple of ALIGN: finish with buffered read
            ssize_t r2 = pread(s->fd, dst + done, n - done, a0 + done);
            if (r2 <= 0) { perror("expert read"); exit(1); }
            r = r2;
        }
        done += r;
    }
    return dst + (off - a0);
}

// read all parts of an expert at once with kernel AIO (O_DIRECT): the parts are ~0.7 MB
// apart in the file, and three requests in flight read ~1.5x faster than one after another
static int aio_read_fd(struct expert_store *s, int fd, uint64_t fsize, aio_context_t ctx, int n, uint8_t **dst,
                       const uint64_t *off, const uint64_t *len, const uint8_t **out);
static int aio_read_parts(struct expert_store *s, aio_context_t ctx, int n, uint8_t **dst,
                          const uint64_t *off, const uint64_t *len, const uint8_t **out) {
    return aio_read_fd(s, s->fd_direct, s->file_size, ctx, n, dst, off, len, out);
}
static int aio_read_fd(struct expert_store *s, int fd, uint64_t fsize, aio_context_t ctx, int n, uint8_t **dst,
                       const uint64_t *off, const uint64_t *len, const uint8_t **out) {
    struct iocb cb[8], *cbs[8];
    for (int k = 0; k < n; k++) {
        const uint64_t a0 = off[k] & ~(uint64_t)(ALIGN - 1);
        uint64_t a1 = (off[k] + len[k] + ALIGN - 1) & ~(uint64_t)(ALIGN - 1);
        if (a1 > fsize) return -1;             // tail not aligned: caller falls back
        memset(&cb[k], 0, sizeof cb[k]);
        cb[k].aio_fildes = fd;
        cb[k].aio_lio_opcode = IOCB_CMD_PREAD;
        cb[k].aio_buf = (uint64_t)(uintptr_t)dst[k];
        cb[k].aio_nbytes = a1 - a0;
        cb[k].aio_offset = (int64_t)a0;
        cb[k].aio_data = (uint64_t)k;
        cbs[k] = &cb[k];
        out[k] = dst[k] + (off[k] - a0);
    }
    if (syscall(SYS_io_submit, ctx, n, cbs) != n) return -1;
    struct io_event ev[8];
    int got = 0;
    while (got < n) {
        long r = syscall(SYS_io_getevents, ctx, n - got, n - got, ev + got, NULL);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) { perror("io_getevents"); exit(1); }
        got += (int)r;
    }
    for (int k = 0; k < n; k++)
        if ((uint64_t)ev[k].res != cb[ev[k].data].aio_nbytes) { fprintf(stderr, "expert aio read short\n"); exit(1); }
    return 0;
}

static void *io_worker(void *arg) {
    struct expert_store *s = arg;
    aio_context_t ctx = 0;
    const int use_aio = !getenv("QUARRY_NO_AIO") && syscall(SYS_io_setup, 8, &ctx) == 0;
    pthread_mutex_lock(&s->mu);
    for (;;) {
        // first queued demand load; a prefetch only while fewer than pf_max are in flight
        int pick = -1;
        for (;;) {
            for (int k = 0; k < s->q_len && pick < 0; k++) {
                const int qi = s->q[(s->q_head + k) % s->q_cap];
                if (s->slots[qi].urgent || s->pf_inflight < s->pf_max) pick = k;
            }
            if (pick >= 0) break;
            pthread_cond_wait(&s->cv_work, &s->mu);
        }
        int i = s->q[(s->q_head + pick) % s->q_cap];
        for (int k = pick; k > 0; k--)   // close the gap
            s->q[(s->q_head + k) % s->q_cap] = s->q[(s->q_head + k - 1) % s->q_cap];
        s->q_head = (s->q_head + 1) % s->q_cap;
        s->q_len--;
        const int was_pf = !s->slots[i].urgent;
        if (was_pf) s->pf_inflight++;
        int key = s->slots[i].key;
        pthread_mutex_unlock(&s->mu);
        uint8_t *mem = s->slots[i].mem;
        const double t_rd = now_s();
        const uint8_t *p[3] = {NULL, NULL, NULL};
        uint64_t total = 0;
        uint8_t *dst[3];
        uint64_t off[3], len[3];
        for (int k = 0; k < s->n_parts; k++) {
            dst[k] = mem + s->in_slot[(key / s->n_expert) * s->n_parts + k];
            off[k] = part_off(s, key, k);
            len[k] = part_len(s, key, k);
            total += len[k];
        }
        if (s->fd_packed >= 0) {
            // packed file: the whole slot image in one contiguous read
            const int l = key / s->n_expert;
            const uint64_t po = s->pk_base[l] + (uint64_t)(key % s->n_expert) * s->lsz[l], n = s->lsz[l];
            // split into QUARRY_PK_SPLIT requests in flight (AIO), default 2
            static int nsplit = -1;
            if (nsplit < 0) { nsplit = getenv("QUARRY_PK_SPLIT") ? atoi(getenv("QUARRY_PK_SPLIT")) : 2; if (nsplit < 1) nsplit = 1; if (nsplit > 8) nsplit = 8; }
            uint8_t *dd[8]; uint64_t oo[8], ll[8]; const uint8_t *pp[8];
            const uint64_t chunk = (n / nsplit + ALIGN - 1) / ALIGN * ALIGN;
            int ns = 0;
            for (uint64_t at = 0; at < n; at += chunk, ns++) { dd[ns] = mem + at; oo[ns] = po + at; ll[ns] = n - at < chunk ? n - at : chunk; }
            if (!use_aio || aio_read_fd(s, s->fd_packed, s->pk_base[s->n_layer - 1] + s->lsz[s->n_layer - 1] * s->n_expert, ctx, ns, dd, oo, ll, pp) != 0) {
                uint64_t done = 0;
                while (done < n) {
                    ssize_t r = pread(s->fd_packed, mem + done, n - done, po + done);
                    if (r < 0 && errno == EINTR) continue;
                    if (r <= 0) { perror("packed expert read"); exit(1); }
                    done += r;
                }
            }
            for (int k = 0; k < s->n_parts; k++) p[k] = mem + s->in_slot[l * s->n_parts + k];
        } else if (!use_aio || aio_read_parts(s, ctx, s->n_parts, dst, off, len, p) != 0)
            for (int k = 0; k < s->n_parts; k++) p[k] = read_direct(s, dst[k], off[k], len[k]);
        if (getenv("QUARRY_IO_DEBUG")) {
            static double tsum; static uint64_t nrd, bsum;
            __atomic_fetch_add(&nrd, 1, __ATOMIC_RELAXED);
            pthread_mutex_lock(&s->mu); tsum += now_s() - t_rd; bsum += total; pthread_mutex_unlock(&s->mu);
            if (nrd % 3000 == 0) fprintf(stderr, "[io: %llu reads, avg %.2f ms, %.2f GB/s per thread]\n", (unsigned long long)nrd, 1e3 * tsum / nrd, bsum / tsum / 1e9);
        }
        if (s->fixup)
            for (int k = 0; k < s->n_parts; k++) s->fixup(s->fixup_ud, key / s->n_expert, k, (uint8_t *)p[k], len[k]);
        pthread_mutex_lock(&s->mu);
        for (int k = 0; k < 3; k++) s->slots[i].p[k] = p[k];
        s->slots[i].state = S_READY;
        *s->bytes_read += total;
        (*s->loads)++;
        if (was_pf) { s->pf_inflight--; pthread_cond_broadcast(&s->cv_work); }
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

// find a victim slot of class cls (LRU tail that is not pinned or loading). caller holds mu.
static int evict(struct expert_store *s, int cls) {
    for (int i = s->tail; i >= 0; i = s->slots[i].prev) {
        slot_t *x = &s->slots[i];
        if (x->cls != cls || x->pins || x->state == S_LOADING) continue;
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
            s->slots[i].urgent = 1;
            pthread_cond_broadcast(&s->cv_work);
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
    i = evict(s, s->layer_cls[key / s->n_expert]);
    if (i < 0) return -1;
    slot_t *x = &s->slots[i];
    x->key = key;
    x->state = S_LOADING;
    x->prefetched = !demand;
    x->urgent = demand;
    x->used = 0;
    s->map[key] = i;
    lru_unlink(s, i);
    lru_push_front(s, i);
    enqueue(s, i, demand);
    if (!demand) s->prefetch_issued++;
    return i;
}

// slot layout per layer: parts back to back, each with room for its aligned read window
static void slot_layout(const expert_src *src, uint64_t *in_slot, uint64_t *lsize) {
    for (int l = 0; l < src->n_layer; l++) {
        uint64_t o = 0;
        for (int k = 0; k < src->n_parts; k++) {
            in_slot[l * src->n_parts + k] = o;
            o += (src->bytes[l * src->n_parts + k] + 2 * ALIGN + ALIGN - 1) / ALIGN * ALIGN;
        }
        lsize[l] = o;
    }
}

int experts_write_packed(const expert_src *src, const char *path) {
    const int np = src->n_layer * src->n_parts;
    uint64_t *in_slot = malloc(np * sizeof(uint64_t)), *lsize = malloc(src->n_layer * sizeof(uint64_t));
    slot_layout(src, in_slot, lsize);
    uint64_t mx = 0;
    for (int l = 0; l < src->n_layer; l++) if (lsize[l] > mx) mx = lsize[l];
    char tmp[4096];
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    FILE *f = fopen(tmp, "wb");
    if (!f) { perror(tmp); return -1; }
    uint8_t *buf = malloc(mx);
    for (int l = 0; l < src->n_layer; l++) {
        for (int e = 0; e < src->n_expert; e++) {
            memset(buf, 0, lsize[l]);
            for (int k = 0; k < src->n_parts; k++) {
                const int i = l * src->n_parts + k;
                uint64_t off = src->off[i] + (uint64_t)e * src->bytes[i], n = src->bytes[i], done = 0;
                while (done < n) {
                    ssize_t r = pread(src->fd, buf + in_slot[i] + done, n - done, off + done);
                    if (r <= 0) { perror("pack read"); fclose(f); return -1; }
                    done += r;
                }
            }
            if (fwrite(buf, 1, lsize[l], f) != lsize[l]) { perror("pack write"); fclose(f); return -1; }
        }
        fprintf(stderr, "\rquarry: packing experts, layer %d/%d", l + 1, src->n_layer);
    }
    fprintf(stderr, "\n");
    free(buf); free(in_slot); free(lsize);
    if (fflush(f) || fsync(fileno(f)) || fclose(f)) { perror("pack close"); return -1; }
    return rename(tmp, path);
}

struct expert_store *experts_open_src(const expert_src *src) {
    struct expert_store *s = calloc(1, sizeof *s);
    s->fd = src->fd; s->file_size = src->file_size;
    s->n_layer = src->n_layer; s->n_expert = src->n_expert; s->n_parts = src->n_parts;
    s->bytes_read = src->bytes_read; s->loads = src->loads;
    s->fixup = src->fixup; s->fixup_ud = src->fixup_ud;
    const int np = s->n_layer * s->n_parts;
    s->src_off = malloc(np * sizeof(uint64_t)); s->src_bytes = malloc(np * sizeof(uint64_t));
    memcpy(s->src_off, src->off, np * sizeof(uint64_t));
    memcpy(s->src_bytes, src->bytes, np * sizeof(uint64_t));
    s->in_slot = malloc(np * sizeof(uint64_t));
    uint64_t *lsize = malloc(s->n_layer * sizeof(uint64_t));
    slot_layout(src, s->in_slot, lsize);
    s->slot_bytes = 0;
    for (int l = 0; l < s->n_layer; l++) if (lsize[l] > s->slot_bytes) s->slot_bytes = lsize[l];
    s->lsz = lsize;
    s->fd_packed = -1;
    if (src->packed) {
        s->pk_base = malloc(s->n_layer * sizeof(uint64_t));
        uint64_t o = 0;
        for (int l = 0; l < s->n_layer; l++) { s->pk_base[l] = o; o += lsize[l] * s->n_expert; }
        int fdp = open(src->packed, O_RDONLY | O_DIRECT);
        if (fdp >= 0 && (uint64_t)lseek(fdp, 0, SEEK_END) == o) s->fd_packed = fdp;
        else {
            if (fdp >= 0) close(fdp);
            fprintf(stderr, "quarry: %s does not match this model, not used\n", src->packed);
        }
    }
    // size classes: layers with the same slot size share slots
    int ncls = 0, cls_layers[64] = {0};
    uint64_t cls_size[64];
    s->layer_cls = malloc(s->n_layer * sizeof(int));
    for (int l = 0; l < s->n_layer; l++) {
        int c = 0;
        while (c < ncls && cls_size[c] != lsize[l]) c++;
        if (c == ncls) { if (ncls == 64) c = 63; else cls_size[ncls++] = lsize[l]; }
        s->layer_cls[l] = c;
        cls_layers[c]++;
    }
    // same cached fraction of experts in every class
    double all = 0;
    for (int l = 0; l < s->n_layer; l++) all += (double)lsize[l] * s->n_expert;
    double gb;
    if (getenv("QUARRY_CACHE_GB")) gb = atof(getenv("QUARRY_CACHE_GB"));
    else {
        // automatic: what is available, minus what the model still needs (part of the KV at full
        // context), minus headroom; never more than the experts that are not in VRAM. Where
        // /proc/meminfo cannot be read, the old default.
        double avail = meminfo("MemAvailable"), total = meminfo("MemTotal");
        const double lim = cgroup_room();   // under a memory limit (systemd, containers), that counts
        // headroom for the rest of the system (1 GB + 5% of RAM); under a memory limit that is
        // tighter than the system's, the limit already protects the rest, so only a little
        double headroom = 1.0e9 + 0.05 * total;
        if (lim > 0 && lim < avail) { avail = lim; headroom = 0.25e9; }
        if (avail > 0) {
            gb = (avail - (double)src->reserve - headroom) / 1e9;
            const double need = (all - (double)src->elsewhere) / 1e9;
            if (gb > need) gb = need;
            if (gb < 0.5) gb = 0.5;
            fprintf(stderr, "quarry: RAM expert cache %.1f GB (auto: %.1f GB available%s; QUARRY_CACHE_GB overrides)\n",
                    gb, avail / 1e9, src->elsewhere ? ", experts in VRAM excluded" : "");
        } else gb = 5.5;
    }
    s->cache_gb = gb;
    double frac = gb * 1e9 / all;
    if (frac > 1) frac = 1;
    int cls_slots[64], total = 0;
    uint64_t slab_bytes = 0;
    for (int c = 0; c < ncls; c++) {
        cls_slots[c] = (int)(frac * cls_layers[c] * s->n_expert);
        if (cls_slots[c] < src->min_slots) cls_slots[c] = src->min_slots;
        total += cls_slots[c];
        slab_bytes += (uint64_t)cls_slots[c] * cls_size[c];
    }
    s->n_keys = s->n_layer * s->n_expert;
    s->n_slots = total;
    s->n_stage = getenv("QUARRY_STAGE_SLOTS") ? atoi(getenv("QUARRY_STAGE_SLOTS")) : 48;
    s->slots = calloc(s->n_slots + s->n_stage, sizeof(slot_t));
    s->map = malloc(s->n_keys * sizeof(int));
    for (int k = 0; k < s->n_keys; k++) s->map[k] = -1;
    const uint64_t slab_total = (slab_bytes + (uint64_t)s->n_stage * s->slot_bytes + (1u << 21) - 1) / (1u << 21) * (1u << 21);
    uint8_t *slab = aligned_alloc(1u << 21, slab_total);   // 2MB huge pages: fewer TLB misses on expert reads
    if (slab) madvise(slab, slab_total, MADV_HUGEPAGE);
    if (!slab) { fprintf(stderr, "cannot allocate %.2f GB expert cache\n", slab_bytes / 1e9); exit(1); }
    s->head = s->tail = -1;
    uint8_t *mem = slab;
    int i = 0;
    for (int c = 0; c < ncls; c++)
        for (int j = 0; j < cls_slots[c]; j++, i++) {
            s->slots[i] = (slot_t){.key = -1, .prev = -1, .next = -1, .mem = mem, .cls = c};
            mem += cls_size[c];
            lru_push_front(s, i);
        }
    for (; i < s->n_slots + s->n_stage; i++) {
        s->slots[i] = (slot_t){.key = -1, .prev = -1, .next = -1, .mem = mem, .cls = -1};
        mem += s->slot_bytes;
    }
    s->q_cap = s->n_slots + s->n_stage + 8;
    s->q = malloc(s->q_cap * sizeof(int));
    char p[64];
    snprintf(p, sizeof p, "/proc/self/fd/%d", s->fd);
    s->fd_direct = open(p, O_RDONLY | O_DIRECT);
    if (s->fd_direct < 0) { perror("open O_DIRECT"); s->fd_direct = s->fd; }
    pthread_mutex_init(&s->mu, NULL);
    pthread_cond_init(&s->cv_work, NULL);
    pthread_cond_init(&s->cv_done, NULL);
    s->n_threads = getenv("QUARRY_IO_THREADS") ? atoi(getenv("QUARRY_IO_THREADS")) : 2;
    s->pf_max = getenv("QUARRY_PF_INFLIGHT") ? atoi(getenv("QUARRY_PF_INFLIGHT")) : 2;
    for (int t = 0; t < s->n_threads; t++) pthread_create(&s->threads[t], NULL, io_worker, s);
    fprintf(stderr, "quarry: expert cache %d slots (%.2f GB, %d size classes, %.0f%% of experts), %d io threads\n",
            s->n_slots, slab_bytes / 1e9, ncls, 100 * frac, s->n_threads);
    return s;
}

// Qwen3-MoE: three parts (gate, up, down), same layout in every layer
struct expert_store *experts_open(qmodel *m) {
    const int L = m->c.n_layer;
    uint64_t *off = malloc(L * 3 * sizeof(uint64_t)), *bytes = malloc(L * 3 * sizeof(uint64_t));
    for (int l = 0; l < L; l++) {
        const qlayer *Y = &m->layers[l];
        const qmat *M[3] = {&Y->gate_exps, &Y->up_exps, &Y->down_exps};
        for (int k = 0; k < 3; k++) {
            off[l * 3 + k] = M[k]->file_off;
            bytes[l * 3 + k] = M[k]->row_bytes * (k < 2 ? m->c.n_ff_exp : m->c.n_embd);
        }
    }
    expert_src src = {m->g.fd, m->g.file_size, L, m->c.n_expert, 3, off, bytes,
                      &m->expert_bytes_read, &m->expert_loads, 2 * m->c.n_expert_used};
    struct expert_store *s = experts_open_src(&src);
    free(off); free(bytes);
    return s;
}

// ---- API used by the forward pass ----

// start loading this layer's selected experts (demand) and pin them
void experts_begin_layer(struct expert_store *s, int l, const int *sel, int k) {
    pthread_mutex_lock(&s->mu);
    for (int j = 0; j < k; j++) {
        int key = l * s->n_expert + sel[j];
        int i = s->map[key];
        if (i >= 0 && s->slots[i].state == S_READY) s->hits++; else s->misses++;
        i = request(s, key, 1);
        if (i >= 0) s->slots[i].pins++;
    }
    pthread_mutex_unlock(&s->mu);
}

void experts_prefetch(struct expert_store *s, int l, const int *sel, int k) {
    pthread_mutex_lock(&s->mu);
    for (int j = 0; j < k; j++) request(s, l * s->n_expert + sel[j], 0);
    pthread_mutex_unlock(&s->mu);
}

int experts_ready(struct expert_store *s, int l, int e) {
    pthread_mutex_lock(&s->mu);
    int i = s->map[l * s->n_expert + e];
    int r = i >= 0 && s->slots[i].state == S_READY;
    pthread_mutex_unlock(&s->mu);
    return r;
}

// blocking get; also unpins when done=1 is called via experts_release
// block until at least one of experts e[0..n) of layer l is ready (all must be requested)
void experts_wait_any(struct expert_store *s, int l, const int *e, int n) {
    pthread_mutex_lock(&s->mu);
    double t0 = now_s();
    int waited = 0;
    for (;;) {
        int any = 0;
        for (int k = 0; k < n && !any; k++) {
            int i = s->map[l * s->n_expert + e[k]];
            if (i < 0) i = request(s, l * s->n_expert + e[k], 1);
            any = s->slots[i].state == S_READY;
        }
        if (any) break;
        waited = 1;
        pthread_cond_wait(&s->cv_done, &s->mu);
    }
    if (waited) s->wait_s += now_s() - t0;
    pthread_mutex_unlock(&s->mu);
}

void experts_get(struct expert_store *s, int l, int e, const uint8_t **gate, const uint8_t **up,
                 const uint8_t **down) {
    int key = l * s->n_expert + e;
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
    int i = s->map[l * s->n_expert + e];
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

// ---- prefill streaming: read experts that are not cached into staging slots ----

// pin a cached expert if it is resident; returns 1 and its weights, else 0
int experts_try_pin(struct expert_store *s, int l, int e, const uint8_t **gate, const uint8_t **up,
                    const uint8_t **down) {
    pthread_mutex_lock(&s->mu);
    int i = s->map[l * s->n_expert + e], ok = 0;
    if (i >= 0 && s->slots[i].state == S_READY) {
        s->slots[i].pins++;
        *gate = s->slots[i].p[0]; *up = s->slots[i].p[1]; *down = s->slots[i].p[2];
        ok = 1;
    }
    pthread_mutex_unlock(&s->mu);
    return ok;
}

// queue a read of expert e of layer l into a free staging slot; -1 if none is free
int experts_stream(struct expert_store *s, int l, int e) {
    pthread_mutex_lock(&s->mu);
    // a never-used cache slot is free to fill: load there, so a cold cache warms up
    const int cls = s->layer_cls[l];
    for (int i = s->tail; i >= 0; i = s->slots[i].prev) {
        slot_t *x = &s->slots[i];
        if (x->cls != cls || x->pins || x->state == S_LOADING) continue;
        if (x->key < 0) {
            int h = request(s, l * s->n_expert + e, 0);
            s->prefetch_issued--;   // not a prefetch
            s->slots[h].prefetched = 0;
            s->slots[h].urgent = 1;
            s->slots[h].pins++;
            pthread_mutex_unlock(&s->mu);
            return h;
        }
        break;
    }
    int h = -1;
    for (int i = s->n_slots; i < s->n_slots + s->n_stage; i++)
        if (!s->slots[i].pins) { h = i; break; }
    if (h >= 0) {
        slot_t *x = &s->slots[h];
        x->key = l * s->n_expert + e;
        x->state = S_LOADING;
        x->pins = 1;
        x->urgent = 1;          // prefill streaming is not capped like prefetch
        enqueue(s, h, 0);
    }
    pthread_mutex_unlock(&s->mu);
    return h;
}

void experts_stream_get(struct expert_store *s, int h, const uint8_t **gate, const uint8_t **up,
                        const uint8_t **down) {
    pthread_mutex_lock(&s->mu);
    if (s->slots[h].state != S_READY) {
        double t0 = now_s();
        while (s->slots[h].state != S_READY) pthread_cond_wait(&s->cv_done, &s->mu);
        s->wait_s += now_s() - t0;
    }
    *gate = s->slots[h].p[0]; *up = s->slots[h].p[1]; *down = s->slots[h].p[2];
    pthread_mutex_unlock(&s->mu);
}

void experts_stream_done(struct expert_store *s, int h) {
    pthread_mutex_lock(&s->mu);
    if (h < s->n_slots) {   // went into the cache
        if (s->slots[h].pins > 0) s->slots[h].pins--;
        pthread_mutex_unlock(&s->mu);
        return;
    }
    s->slots[h].pins = 0;
    s->slots[h].state = S_EMPTY;
    s->slots[h].key = -1;
    pthread_mutex_unlock(&s->mu);
}

// mispredicted prefetches: move to the LRU tail so they are evicted first
double experts_cache_gb(const struct expert_store *s) { return s->cache_gb; }

void experts_drop(struct expert_store *s, int l, int e) {
    pthread_mutex_lock(&s->mu);
    int i = s->map[l * s->n_expert + e];
    if (i >= 0 && !s->slots[i].pins && s->slots[i].state == S_READY && s->tail != i) {
        lru_unlink(s, i);
        slot_t *x = &s->slots[i];
        x->next = -1;
        x->prev = s->tail;
        if (s->tail >= 0) s->slots[s->tail].next = i;
        s->tail = i;
        if (s->head < 0) s->head = i;
    }
    pthread_mutex_unlock(&s->mu);
}

void experts_demote(struct expert_store *s, int l, const int *sel, int k) {
    pthread_mutex_lock(&s->mu);
    for (int j = 0; j < k; j++) {
        int i = s->map[l * s->n_expert + sel[j]];
        if (i < 0 || s->slots[i].pins || s->slots[i].used) continue;
        lru_unlink(s, i);
        slot_t *x = &s->slots[i];
        x->next = -1;
        x->prev = s->tail;
        if (s->tail >= 0) s->slots[s->tail].next = i;
        s->tail = i;
        if (s->head < 0) s->head = i;
    }
    pthread_mutex_unlock(&s->mu);
}
