// Quarry: quantized formats (GGML-compatible block layouts) and dot kernels.
#pragma once
#include <stdint.h>

#define QK_K 256

typedef struct {            // 110 bytes, 3.4375 bits/weight
    uint8_t hmask[QK_K / 8];  // high bit of each 3-bit quant
    uint8_t qs[QK_K / 4];     // low 2 bits
    uint8_t scales[12];       // 16 x 6-bit scales, packed
    uint16_t d;               // fp16 super-block scale
} block_q3_K;

typedef struct {            // 210 bytes, 6.5625 bits/weight
    uint8_t ql[QK_K / 2];     // low 4 bits
    uint8_t qh[QK_K / 4];     // high 2 bits
    int8_t scales[QK_K / 16]; // 16 x 8-bit scales
    uint16_t d;
} block_q6_K;

typedef struct {            // activations: int8 per 256, with 16-group sums
    float d;
    int8_t qs[QK_K];
    int16_t bsums[QK_K / 16];
} block_q8_K;

_Static_assert(sizeof(block_q3_K) == 110, "q3_K size");
_Static_assert(sizeof(block_q6_K) == 210, "q6_K size");

float fp16_to_fp32(uint16_t h);

void dequant_row_q3_K(const block_q3_K *x, float *y, int64_t n);
void dequant_row_q6_K(const block_q6_K *x, float *y, int64_t n);

void quantize_row_q8_K(const float *x, block_q8_K *y, int64_t n);

// dot of one weight row with quantized activations (n = row length)
float vec_dot_q3_K_q8_K_ref(int n, const block_q3_K *x, const block_q8_K *y);
float vec_dot_q6_K_q8_K_ref(int n, const block_q6_K *x, const block_q8_K *y);
float vec_dot_q3_K_q8_K(int n, const block_q3_K *x, const block_q8_K *y);
float vec_dot_q6_K_q8_K(int n, const block_q6_K *x, const block_q8_K *y);

// ---- batched path: unpack a Q3_K row once, then dot it with many activation rows ----
// Unpacked block: 256 unsigned quants u = q+4 (0..7) in natural order,
// 128 int16 scales expanded to the maddubs pair layout, 16 int16 group scales
// (for the offset correction) and the fp32 super-block scale.
typedef struct {
    uint8_t u[QK_K];
    int16_t sx[QK_K / 2];
    int16_t sc[16];
    float d;
} q3k_unpacked;

void q3k_unpack_row(const block_q3_K *x, int n, q3k_unpacked *out);
float vec_dot_q3k_unpacked(int n, const q3k_unpacked *w, const block_q8_K *y);

// ---- batched matmul path (prefill) ----
// A Q3_K block repacked to one int8 weight per value: w = (q - 4) * scale,
// saturated to int8 (only -4 * -32 = 128 clips). corr = -128 * sum(w) undoes the
// +128 offset of the unsigned activations, so one dpbusd covers 64 values.
typedef struct {
    int8_t w[QK_K];
    float d;
    int32_t corr;
} q3k_s8;

void q3k_repack_row(const block_q3_K *x, int n, q3k_s8 *out);
// activations as unsigned bytes (q8 + 128) with one scale per 256 values
void quantize_row_u8(const float *x, uint8_t *xu, float *dx, int n);
// out[t * out_stride] = dot(row, token t) for t in [0, T)
void dot_q3k_s8_multi(int n, const q3k_s8 *w, int T, const uint8_t *xu, const float *dx,
                      float *out, int out_stride);

// ---- formats used by Inkling 2 (Q8_0, Q5_0, Q4_K, Q5_K) ----
typedef struct { uint16_t d; int8_t qs[32]; } block_q8_0;                          // 34 bytes
typedef struct { uint16_t d; uint8_t qh[4]; uint8_t qs[16]; } block_q5_0;          // 22 bytes
typedef struct { uint16_t d, dmin; uint8_t scales[12]; uint8_t qs[128]; } block_q4_K;            // 144
typedef struct { uint16_t d, dmin; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128]; } block_q5_K; // 176
_Static_assert(sizeof(block_q8_0) == 34, "q8_0 size");
_Static_assert(sizeof(block_q5_0) == 22, "q5_0 size");
_Static_assert(sizeof(block_q4_K) == 144, "q4_K size");
_Static_assert(sizeof(block_q5_K) == 176, "q5_K size");

void dequant_row_q8_0(const block_q8_0 *x, float *y, int64_t n);
void dequant_row_q5_0(const block_q5_0 *x, float *y, int64_t n);
void dequant_row_q4_K(const block_q4_K *x, float *y, int64_t n);
void dequant_row_q5_K(const block_q5_K *x, float *y, int64_t n);
// any supported type -> float
int dequant_row(int type, const void *x, float *y, int64_t n);

// activations in 32-value blocks (for Q8_0/Q5_0 weights): qs[n], d[n/32], s[n/32] = sum of qs
typedef struct { int8_t *qs; float *d; int32_t *s; } act32;
void quantize_row_a32(const float *x, act32 a, int64_t n);

float vec_dot_q8_0_a32(int n, const block_q8_0 *x, act32 a);
float vec_dot_q5_0_a32(int n, const block_q5_0 *x, act32 a);
float vec_dot_q4_K_q8_K(int n, const block_q4_K *x, const block_q8_K *y);
float vec_dot_q5_K_q8_K(int n, const block_q5_K *x, const block_q8_K *y);

// two activation vectors per weight row: r0 = row . y0, r1 = row . y1 (unpacking shared)
void vec_dot2_q3_K_q8_K(int n, const block_q3_K *x, const block_q8_K *y0, const block_q8_K *y1, float *r0, float *r1);
void vec_dot2_q4_K_q8_K(int n, const block_q4_K *x, const block_q8_K *y0, const block_q8_K *y1, float *r0, float *r1);
void vec_dot2_q5_K_q8_K(int n, const block_q5_K *x, const block_q8_K *y0, const block_q8_K *y1, float *r0, float *r1);
