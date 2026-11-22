#include "quants.h"

#include <immintrin.h>
#include <math.h>
#include <string.h>

float fp16_to_fp32(uint16_t h) { return _cvtsh_ss(h); }

// Unpack the 12-byte Q3_K scale field into 16 signed scales (value - 32).
static inline void q3k_scales(const uint8_t *packed, int8_t out[16]) {
    uint32_t aux[4];
    const uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
    memcpy(aux, packed, 12);
    uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    const int8_t *s = (const int8_t *)aux;
    for (int i = 0; i < 16; i++) out[i] = s[i] - 32;
}

// same unpack, but returns the 16 raw 6-bit scales (0..63) in a vector
static inline __m128i q3k_scales_raw(const uint8_t *packed) {
    uint32_t aux[4];
    const uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
    memcpy(aux, packed, 12);
    uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    return _mm_loadu_si128((const __m128i *)aux);
}

// Element order inside a Q3_K block (matches ggml):
//   for half n in {0,1}: for j in 0..3: for g in {0,1}: 16 values
//     value index = n*128 + j*32 + g*16 + l
//     low bits  = (qs[n*32 + g*16 + l] >> (2*j)) & 3
//     high bit  = hmask[g*16 + l] bit (n*4 + j); quant = low - (high ? 0 : 4)
//     scale     = scales[n*8 + j*2 + g]
void dequant_row_q3_K(const block_q3_K *x, float *y, int64_t n) {
    for (int64_t b = 0; b < n / QK_K; b++) {
        const block_q3_K *blk = &x[b];
        float d = fp16_to_fp32(blk->d);
        int8_t sc[16];
        q3k_scales(blk->scales, sc);
        for (int h = 0; h < 2; h++)
            for (int j = 0; j < 4; j++)
                for (int g = 0; g < 2; g++) {
                    float dl = d * sc[h * 8 + j * 2 + g];
                    for (int l = 0; l < 16; l++) {
                        int idx = g * 16 + l;
                        int low = (blk->qs[h * 32 + idx] >> (2 * j)) & 3;
                        int hi = (blk->hmask[idx] >> (h * 4 + j)) & 1;
                        y[h * 128 + j * 32 + g * 16 + l] = dl * (low - (hi ? 0 : 4));
                    }
                }
        y += QK_K;
    }
}

// Q6_K: for half n in {0,1}, l in 0..31, k in 0..3:
//   value index = n*128 + k*32 + l ; scale = scales[n*8 + l/16 + 2k]
void dequant_row_q6_K(const block_q6_K *x, float *y, int64_t n) {
    for (int64_t b = 0; b < n / QK_K; b++) {
        const block_q6_K *blk = &x[b];
        float d = fp16_to_fp32(blk->d);
        const uint8_t *ql = blk->ql, *qh = blk->qh;
        const int8_t *sc = blk->scales;
        for (int h = 0; h < 2; h++) {
            for (int l = 0; l < 32; l++) {
                int is = l / 16;
                int q1 = (int)((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int q3 = (int)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                int q4 = (int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l + 0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

void quantize_row_q8_K(const float *x, block_q8_K *y, int64_t n) {
    for (int64_t b = 0; b < n / QK_K; b++) {
        const float *xb = x + b * QK_K;
        float amax = 0;
        for (int i = 0; i < QK_K; i++) amax = fmaxf(amax, fabsf(xb[i]));
        float d = amax / 127.f, id = d ? 1.f / d : 0.f;
        y[b].d = d;
        for (int g = 0; g < 16; g++) {
            int s = 0;
            for (int l = 0; l < 16; l++) {
                int v = (int)lrintf(xb[g * 16 + l] * id);
                y[b].qs[g * 16 + l] = (int8_t)v;
                s += v;
            }
            y[b].bsums[g] = (int16_t)s;
        }
    }
}

// ---------- reference dots (integer math, same as the fast path) ----------

float vec_dot_q3_K_q8_K_ref(int n, const block_q3_K *x, const block_q8_K *y) {
    float sum = 0;
    for (int b = 0; b < n / QK_K; b++) {
        int8_t sc[16];
        q3k_scales(x[b].scales, sc);
        int32_t isum = 0;
        for (int h = 0; h < 2; h++)
            for (int j = 0; j < 4; j++)
                for (int g = 0; g < 2; g++) {
                    int32_t s = 0;
                    for (int l = 0; l < 16; l++) {
                        int idx = g * 16 + l;
                        int low = (x[b].qs[h * 32 + idx] >> (2 * j)) & 3;
                        int hi = (x[b].hmask[idx] >> (h * 4 + j)) & 1;
                        s += (low - (hi ? 0 : 4)) * y[b].qs[h * 128 + j * 32 + g * 16 + l];
                    }
                    isum += sc[h * 8 + j * 2 + g] * s;
                }
        sum += fp16_to_fp32(x[b].d) * y[b].d * isum;
    }
    return sum;
}

float vec_dot_q6_K_q8_K_ref(int n, const block_q6_K *x, const block_q8_K *y) {
    float sum = 0;
    for (int b = 0; b < n / QK_K; b++) {
        const uint8_t *ql = x[b].ql, *qh = x[b].qh;
        const int8_t *sc = x[b].scales, *q8 = y[b].qs;
        int32_t isum = 0;
        for (int h = 0; h < 2; h++) {
            int32_t s[8] = {0};
            for (int l = 0; l < 32; l++) {
                int is = l / 16;
                int q1 = (int)((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int q3 = (int)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                int q4 = (int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                s[is + 0] += q1 * q8[l + 0];
                s[is + 2] += q2 * q8[l + 32];
                s[is + 4] += q3 * q8[l + 64];
                s[is + 6] += q4 * q8[l + 96];
            }
            for (int k = 0; k < 8; k++) isum += sc[k] * s[k];
            ql += 64; qh += 32; sc += 8; q8 += 128;
        }
        sum += fp16_to_fp32(x[b].d) * y[b].d * isum;
    }
    return sum;
}

// ---------- fast paths: AVX-512VL + VNNI on 256-bit vectors ----------
//
// Both formats are handled as unsigned quants u (Q3_K: u = q+4 in 0..7,
// Q6_K: u = q+32 in 0..63) so vpdpbusd (u8 x s8) can be used directly:
//   sum(q*x) = sum(u*x) - offset * sum(x),   sum(x) per 16-group = bsums.
// Values come in 8 chunks of 32; chunk c covers 16-groups 2c and 2c+1, and
// group g always uses scale g (checked against the reference layout).

static inline int32_t hsum_i32_8(__m256i v) {
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
    return _mm_cvtsi128_si32(s);
}

// sum_g scale[g] * bsums[g]  (16 int16 pairs)
static inline int32_t scale_bsum_dot(const int8_t sc[16], const int16_t *bsums) {
    __m256i s16 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)sc));
    __m256i b16 = _mm256_loadu_si256((const __m256i *)bsums);
    return hsum_i32_8(_mm256_madd_epi16(s16, b16));
}

// per-chunk scale vector: lanes 0-3 = sc[2c], lanes 4-7 = sc[2c+1]
static inline __m256i chunk_scales(const int8_t *sc, int c) {
    return _mm256_set_m128i(_mm_set1_epi32(sc[2 * c + 1]), _mm_set1_epi32(sc[2 * c]));
}

// 512-bit Q3_K: two chunks (64 values) per step.
//   u     = low2 | hbit<<2  (0..7), built with per-lane variable shifts
//   p16   = maddubs(u, x)   -> 32 int16 (pairs of products; |p16| <= 2*7*127)
//   p32   = madd(p16, scales replicated per 16-group) -> 16 int32
// The scale applied per block happens in float at the end, one reduce per row.
float vec_dot_q3_K_q8_K(int n, const block_q3_K *x, const block_q8_K *y) {
    const __m512i m3 = _mm512_set1_epi8(3), m1 = _mm512_set1_epi8(1);
    // scale index for int16 lane k (32 lanes) of chunk pair p: group = 4p + k/8
    __m512i sidx[4];
    for (int p = 0; p < 4; p++) {
        int16_t idx[32];
        for (int k = 0; k < 32; k++) idx[k] = (int16_t)(4 * p + k / 8);
        sidx[p] = _mm512_loadu_si512(idx);
    }
    __m512 accf = _mm512_setzero_ps();
    const __m256i c32 = _mm256_set1_epi16(32);
    for (int b = 0; b < n / QK_K; b++) {
        // 16 scales as int16, minus 32 (vector op instead of a scalar loop)
        const __m256i s16 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(q3k_scales_raw(x[b].scales)), c32);
        const __m512i sc16 = _mm512_castsi256_si512(s16);
        const __m512i hm = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i *)x[b].hmask));
        __m512i acc = _mm512_setzero_si512();
        for (int h = 0; h < 2; h++) {
            const __m512i qs = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i *)(x[b].qs + h * 32)));
            for (int jp = 0; jp < 2; jp++) {
                const int j = jp * 2, c = h * 4 + j, p = c / 2;
                const __m512i sh_lo = _mm512_inserti64x4(_mm512_set1_epi16(2 * j), _mm256_set1_epi16(2 * j + 2), 1);
                const __m512i sh_hi = _mm512_inserti64x4(_mm512_set1_epi16(4 * h + j), _mm256_set1_epi16(4 * h + j + 1), 1);
                __m512i low = _mm512_and_si512(_mm512_srlv_epi16(qs, sh_lo), m3);
                __m512i hi = _mm512_and_si512(_mm512_srlv_epi16(hm, sh_hi), m1);
                __m512i u = _mm512_or_si512(low, _mm512_slli_epi16(hi, 2));
                __m512i xv = _mm512_loadu_si512((const void *)(y[b].qs + c * 32));
                __m512i p16 = _mm512_maddubs_epi16(u, xv);
                __m512i sv = _mm512_permutexvar_epi16(sidx[p], sc16);
                acc = _mm512_add_epi32(acc, _mm512_madd_epi16(p16, sv));
            }
        }
        // offset correction 4 * sum_g sc_g * bsum_g, subtracted in the integer
        // domain so each block's sum stays exact (no float cancellation)
        __m256i cb = _mm256_madd_epi16(s16, _mm256_loadu_si256((const __m256i *)y[b].bsums));
        acc = _mm512_sub_epi32(acc, _mm512_zextsi256_si512(_mm256_slli_epi32(cb, 2)));
        const float dd = fp16_to_fp32(x[b].d) * y[b].d;
        accf = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc), _mm512_set1_ps(dd), accf);
    }
    return _mm512_reduce_add_ps(accf);
}

__attribute__((unused)) static float vec_dot_q3_K_q8_K_avx2(int n, const block_q3_K *x, const block_q8_K *y) {
    const __m256i m3 = _mm256_set1_epi8(3), m1 = _mm256_set1_epi8(1);
    float sum = 0;
    for (int b = 0; b < n / QK_K; b++) {
        int8_t sc[16];
        q3k_scales(x[b].scales, sc);
        const __m256i hm = _mm256_loadu_si256((const __m256i *)x[b].hmask);
        __m256i acc = _mm256_setzero_si256();
        for (int h = 0; h < 2; h++) {
            const __m256i qs = _mm256_loadu_si256((const __m256i *)(x[b].qs + h * 32));
            for (int j = 0; j < 4; j++) {
                const int c = h * 4 + j;
                __m256i low = _mm256_and_si256(_mm256_srli_epi16(qs, 2 * j), m3);
                __m256i hi = _mm256_and_si256(_mm256_srli_epi16(hm, h * 4 + j), m1);
                __m256i u = _mm256_or_si256(low, _mm256_slli_epi16(hi, 2));
                __m256i xv = _mm256_loadu_si256((const __m256i *)(y[b].qs + c * 32));
                __m256i d = _mm256_dpbusd_epi32(_mm256_setzero_si256(), u, xv);
                acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(d, chunk_scales(sc, c)));
            }
        }
        int32_t isum = hsum_i32_8(acc) - 4 * scale_bsum_dot(sc, y[b].bsums);
        sum += fp16_to_fp32(x[b].d) * y[b].d * (float)isum;
    }
    return sum;
}

float vec_dot_q6_K_q8_K(int n, const block_q6_K *x, const block_q8_K *y) {
    const __m256i m4 = _mm256_set1_epi8(0x0F), m2 = _mm256_set1_epi8(3);
    float sum = 0;
    for (int b = 0; b < n / QK_K; b++) {
        const int8_t *sc = x[b].scales;
        __m256i acc = _mm256_setzero_si256();
        for (int h = 0; h < 2; h++) {
            const __m256i l0 = _mm256_loadu_si256((const __m256i *)(x[b].ql + h * 64));
            const __m256i l1 = _mm256_loadu_si256((const __m256i *)(x[b].ql + h * 64 + 32));
            const __m256i qh = _mm256_loadu_si256((const __m256i *)(x[b].qh + h * 32));
            __m256i u[4];
            u[0] = _mm256_or_si256(_mm256_and_si256(l0, m4),
                                   _mm256_slli_epi16(_mm256_and_si256(qh, m2), 4));
            u[1] = _mm256_or_si256(_mm256_and_si256(l1, m4),
                                   _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh, 2), m2), 4));
            u[2] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(l0, 4), m4),
                                   _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh, 4), m2), 4));
            u[3] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(l1, 4), m4),
                                   _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh, 6), m2), 4));
            for (int k = 0; k < 4; k++) {
                const int c = h * 4 + k;
                __m256i xv = _mm256_loadu_si256((const __m256i *)(y[b].qs + c * 32));
                __m256i d = _mm256_dpbusd_epi32(_mm256_setzero_si256(), u[k], xv);
                acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(d, chunk_scales(sc, c)));
            }
        }
        int32_t isum = hsum_i32_8(acc) - 32 * scale_bsum_dot(sc, y[b].bsums);
        sum += fp16_to_fp32(x[b].d) * y[b].d * (float)isum;
    }
    return sum;
}
