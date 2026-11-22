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
