#define _GNU_SOURCE
#include "qgguf.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

// Buffered sequential reader for the header (header is small, a few MB).
typedef struct {
    int fd;
    uint8_t *buf;
    size_t len, pos;
    uint64_t file_pos;   // file position of buf[0]
} rd_t;

static int rd_fill(rd_t *r, size_t need) {
    if (r->len - r->pos >= need) return 0;
    size_t keep = r->len - r->pos;
    memmove(r->buf, r->buf + r->pos, keep);
    r->file_pos += r->pos;
    r->len = keep;
    r->pos = 0;
    size_t cap = 1 << 22;
    if (need > cap) return -1;
    while (r->len < need) {
        ssize_t n = pread(r->fd, r->buf + r->len, cap - r->len, r->file_pos + r->len);
        if (n <= 0) return -1;
        r->len += n;
    }
    return 0;
}

static int rd_bytes(rd_t *r, void *dst, size_t n) {
    if (rd_fill(r, n)) return -1;
    memcpy(dst, r->buf + r->pos, n);
    r->pos += n;
    return 0;
}

#define RD(r, v) rd_bytes((r), &(v), sizeof(v))

static char *rd_str(rd_t *r) {
    uint64_t n;
    if (RD(r, n)) return NULL;
    char *s = malloc(n + 1);
    if (rd_bytes(r, s, n)) { free(s); return NULL; }
    s[n] = 0;
    return s;
}

static size_t vtype_size(int t) {
    switch (t) {
    case GV_U8: case GV_I8: case GV_BOOL: return 1;
    case GV_U16: case GV_I16: return 2;
    case GV_U32: case GV_I32: case GV_F32: return 4;
    case GV_U64: case GV_I64: case GV_F64: return 8;
    default: return 0;
    }
}

static int64_t widen_int(int t, const void *p) {
    switch (t) {
    case GV_U8: return *(uint8_t *)p;   case GV_I8: return *(int8_t *)p;
    case GV_U16: return *(uint16_t *)p; case GV_I16: return *(int16_t *)p;
    case GV_U32: return *(uint32_t *)p; case GV_I32: return *(int32_t *)p;
    case GV_U64: return (int64_t)*(uint64_t *)p; case GV_I64: return *(int64_t *)p;
    case GV_BOOL: return *(uint8_t *)p;
    default: return 0;
    }
}

int qg_open(qg_file *g, const char *path) {
    memset(g, 0, sizeof *g);
    g->fd = open(path, O_RDONLY);
    if (g->fd < 0) { perror(path); return -1; }
    struct stat st;
    fstat(g->fd, &st);
    g->file_size = st.st_size;

    rd_t r = {g->fd, malloc(1 << 22), 0, 0, 0};
    uint32_t magic = 0, version = 0;
    uint64_t n_tensors = 0, n_kv = 0;
    if (RD(&r, magic) || magic != 0x46554747) { fprintf(stderr, "not a GGUF file\n"); return -1; }
    if (RD(&r, version) || version != 3) { fprintf(stderr, "unsupported GGUF version %u\n", version); return -1; }
    RD(&r, n_tensors);
    RD(&r, n_kv);
    g->n_kv = (int)n_kv;
    g->kv = calloc(n_kv, sizeof(qg_kv));
    g->alignment = 32;

    for (uint64_t k = 0; k < n_kv; k++) {
        qg_kv *kv = &g->kv[k];
        kv->key = rd_str(&r);
        uint32_t t = 0;
        RD(&r, t);
        kv->type = t;
        if (t == GV_STR) {
            kv->s = rd_str(&r);
        } else if (t == GV_ARR) {
            uint32_t at = 0;
            uint64_t an = 0;
            RD(&r, at);
            RD(&r, an);
            kv->arr_type = at;
            kv->arr_n = an;
            if (at == GV_STR) {
                char **ss = malloc(an * sizeof(char *));
                for (uint64_t j = 0; j < an; j++) ss[j] = rd_str(&r);
                kv->arr = ss;
            } else {
                size_t es = vtype_size(at);
                kv->arr = malloc(an * es + 1);
                // arrays can be large (token types, merges are strings): read in chunks
                uint8_t *dst = kv->arr;
                uint64_t left = an * es;
                while (left) {
                    size_t chunk = left > (1 << 20) ? (1 << 20) : left;
                    if (rd_bytes(&r, dst, chunk)) return -1;
                    dst += chunk;
                    left -= chunk;
                }
            }
        } else {
            uint8_t v[8];
            size_t es = vtype_size(t);
            rd_bytes(&r, v, es);
            if (t == GV_F32) kv->f = *(float *)v;
            else if (t == GV_F64) kv->f = *(double *)v;
            else kv->i = widen_int(t, v);
        }
        if (!strcmp(kv->key, "general.alignment")) g->alignment = (uint32_t)kv->i;
    }

    g->n_tensors = (int)n_tensors;
    g->tensors = calloc(n_tensors, sizeof(qg_tensor));
    for (uint64_t j = 0; j < n_tensors; j++) {
        qg_tensor *t = &g->tensors[j];
        t->name = rd_str(&r);
        uint32_t nd = 0;
        RD(&r, nd);
        t->n_dims = nd;
        for (int d = 0; d < 4; d++) t->ne[d] = 1;
        for (uint32_t d = 0; d < nd; d++) { uint64_t ne = 0; RD(&r, ne); t->ne[d] = ne; }
        uint32_t ty = 0;
        uint64_t off = 0;
        RD(&r, ty);
        RD(&r, off);
        t->type = ty;
        t->offset = off;  // relative for now
        int64_t n = t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3];
        t->nbytes = qg_row_bytes(ty, n);
    }
    uint64_t hdr_end = r.file_pos + r.pos;
    g->data_offset = (hdr_end + g->alignment - 1) / g->alignment * g->alignment;
    for (int j = 0; j < g->n_tensors; j++) g->tensors[j].offset += g->data_offset;
    free(r.buf);
    return 0;
}

void qg_close(qg_file *g) {
    if (g->fd > 0) close(g->fd);
    // metadata intentionally leaked at exit (process-lifetime object)
}

const qg_kv *qg_find_kv(const qg_file *g, const char *key) {
    for (int k = 0; k < g->n_kv; k++)
        if (!strcmp(g->kv[k].key, key)) return &g->kv[k];
    return NULL;
}

int64_t qg_get_int(const qg_file *g, const char *key, int64_t def) {
    const qg_kv *kv = qg_find_kv(g, key);
    return kv ? kv->i : def;
}

double qg_get_float(const qg_file *g, const char *key, double def) {
    const qg_kv *kv = qg_find_kv(g, key);
    if (!kv) return def;
    return (kv->type == GV_F32 || kv->type == GV_F64) ? kv->f : (double)kv->i;
}

const char *qg_get_str(const qg_file *g, const char *key) {
    const qg_kv *kv = qg_find_kv(g, key);
    return kv && kv->type == GV_STR ? kv->s : NULL;
}

const qg_tensor *qg_find_tensor(const qg_file *g, const char *name) {
    for (int j = 0; j < g->n_tensors; j++)
        if (!strcmp(g->tensors[j].name, name)) return &g->tensors[j];
    return NULL;
}

size_t qg_type_block_bytes(int type) {
    switch (type) {
    case GT_F32: return 4;    case GT_F16: return 2;
    case GT_Q4_0: return 18;  case GT_Q4_1: return 20;  case GT_Q5_0: return 22; case GT_Q8_0: return 34;
    case GT_Q2_K: return 84;  case GT_Q3_K: return 110;
    case GT_Q4_K: return 144; case GT_Q5_K: return 176; case GT_Q6_K: return 210;
    default: return 0;
    }
}

int qg_type_block_elems(int type) {
    switch (type) {
    case GT_F32: case GT_F16: return 1;
    case GT_Q4_0: case GT_Q4_1: case GT_Q5_0: case GT_Q8_0: return 32;
    case GT_Q2_K: case GT_Q3_K: case GT_Q4_K: case GT_Q5_K: case GT_Q6_K: return 256;
    default: return 0;
    }
}

const char *qg_type_name(int type) {
    switch (type) {
    case GT_F32: return "F32";   case GT_F16: return "F16";
    case GT_Q4_0: return "Q4_0"; case GT_Q4_1: return "Q4_1"; case GT_Q5_0: return "Q5_0"; case GT_Q8_0: return "Q8_0";
    case GT_Q2_K: return "Q2_K"; case GT_Q3_K: return "Q3_K";
    case GT_Q4_K: return "Q4_K"; case GT_Q5_K: return "Q5_K"; case GT_Q6_K: return "Q6_K";
    default: return "?";
    }
}

uint64_t qg_row_bytes(int type, int64_t n) {
    int be = qg_type_block_elems(type);
    if (!be) return 0;
    return (uint64_t)(n / be) * qg_type_block_bytes(type);
}

int64_t qg_arr_int(const qg_kv *kv, uint64_t i) {
    if (kv->type != GV_ARR || i >= kv->arr_n || kv->arr_type == GV_STR) return -1;
    return widen_int(kv->arr_type, (const uint8_t *)kv->arr + i * vtype_size(kv->arr_type));
}
