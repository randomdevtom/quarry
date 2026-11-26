// Quarry: minimal GGUF v3 reader.
// Reads metadata + tensor directory; tensor data is accessed by file offset
// (dense weights get loaded into RAM, experts are streamed by the expert cache).
#pragma once
#include <stdint.h>
#include <stddef.h>

enum qg_type_id {
    GT_F32 = 0, GT_F16 = 1, GT_Q4_0 = 2, GT_Q4_1 = 3, GT_Q5_0 = 6, GT_Q8_0 = 8,
    GT_Q2_K = 10, GT_Q3_K = 11, GT_Q4_K = 12, GT_Q5_K = 13, GT_Q6_K = 14,
    GT_BF16 = 30,
};

enum qg_vtype {
    GV_U8 = 0, GV_I8, GV_U16, GV_I16, GV_U32, GV_I32, GV_F32, GV_BOOL,
    GV_STR, GV_ARR, GV_U64, GV_I64, GV_F64,
};

typedef struct {
    char *key;
    int type;            // qg_vtype
    // scalar value (ints widened), string, or array
    int64_t i;
    double f;
    char *s;
    int arr_type;
    uint64_t arr_n;
    void *arr;           // for GV_STR arrays: char** ; numeric: raw values widened to int64/double? (kept raw)
} qg_kv;

typedef struct {
    char *name;
    int type;            // qg_type_id
    int n_dims;
    int64_t ne[4];
    uint64_t offset;     // absolute file offset of tensor data
    uint64_t nbytes;
} qg_tensor;

typedef struct {
    int fd;
    uint64_t file_size;
    uint64_t data_offset;
    uint32_t alignment;
    int n_kv;
    qg_kv *kv;
    int n_tensors;
    qg_tensor *tensors;
} qg_file;

int qg_open(qg_file *g, const char *path);
void qg_close(qg_file *g);

const qg_kv *qg_find_kv(const qg_file *g, const char *key);
int64_t qg_get_int(const qg_file *g, const char *key, int64_t def);
double qg_get_float(const qg_file *g, const char *key, double def);
const char *qg_get_str(const qg_file *g, const char *key);
const qg_tensor *qg_find_tensor(const qg_file *g, const char *name);
// element i of a numeric array value
int64_t qg_arr_int(const qg_kv *kv, uint64_t i);

// bytes per block / elements per block for a ggml type
size_t qg_type_block_bytes(int type);
int qg_type_block_elems(int type);
const char *qg_type_name(int type);
uint64_t qg_row_bytes(int type, int64_t n_elems);
