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
    // all in registers: the old version stored 4 x u32 and reloaded them as one 128-bit
    // value, a store-forwarding stall (~15 cycles) on every block
    // bytes 0-7: low nibbles of packed[0..7]; bytes 8-15: their high nibbles
    const __m128i lo8 = _mm_loadl_epi64((const __m128i *)packed);
    const __m128i lo = _mm_and_si128(_mm_unpacklo_epi64(lo8, _mm_srli_epi16(lo8, 4)), _mm_set1_epi8(0x0f));
    // byte k: bits 2*(k/4)..+1 of packed[8 + k%4], moved to bits 4-5
    uint32_t tmp;
    memcpy(&tmp, packed + 8, 4);
    const __m128i t = _mm_srlv_epi32(_mm_set1_epi32((int)tmp), _mm_setr_epi32(0, 2, 4, 6));
    const __m128i hi = _mm_slli_epi16(_mm_and_si128(t, _mm_set1_epi8(0x03)), 4);
    return _mm_or_si128(lo, hi);
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

// ---- constant tables, built once at startup (rebuilding them per row call was ~25% of a
// short down-projection row) ----
static int16_t T_SIDX_Q3[4][32] __attribute__((aligned(64)));   // q3: lane k of pair p -> group 4p + k/8
static int16_t T_SIDX_K4[4][32] __attribute__((aligned(64)));   // q4/q5: lane k of chunk j -> sub-block 2j + k/16
static int16_t T_SHL_Q3[4][32] __attribute__((aligned(64)));    // q3: low-bit shifts per chunk pair
static uint8_t T_HBIT_Q3[8][64] __attribute__((aligned(64)));   // q3: high-bit masks per chunk pair
__attribute__((constructor)) static void quant_tables_init(void) {
    for (int p = 0; p < 4; p++)
        for (int k = 0; k < 32; k++) {
            T_SIDX_Q3[p][k] = (int16_t)(4 * p + k / 8);
            T_SIDX_K4[p][k] = (int16_t)(2 * p + k / 16);
            T_SHL_Q3[p][k] = (int16_t)(k < 16 ? 2 * p : 2 * p + 2);
        }
    for (int b = 0; b < 8; b += 2)
        for (int k = 0; k < 64; k++) T_HBIT_Q3[b][k] = (uint8_t)(1 << (k < 32 ? b : b + 1));
}

float vec_dot_q3_K_q8_K(int n, const block_q3_K *x, const block_q8_K *y) {
    const __m512i m3 = _mm512_set1_epi8(3), c4 = _mm512_set1_epi8(4);
#define shl_tab(j) _mm512_load_si512((const void *)T_SHL_Q3[j])
#define hbit_tab(b) _mm512_load_si512((const void *)T_HBIT_Q3[b])
#define sidx(p) _mm512_load_si512((const void *)T_SIDX_Q3[p])
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
                const __m512i sh_lo = shl_tab(j);
                __m512i low = _mm512_and_si512(_mm512_srlv_epi16(qs, sh_lo), m3);
                // high bit: one mask test against this chunk pair's bits, then 4 where set
                __m512i u = _mm512_or_si512(low, _mm512_maskz_mov_epi8(_mm512_test_epi8_mask(hm, hbit_tab(h * 4 + j)), c4));
                __m512i xv = _mm512_loadu_si512((const void *)(y[b].qs + c * 32));
                __m512i p16 = _mm512_maddubs_epi16(u, xv);
                __m512i sv = _mm512_permutexvar_epi16(sidx(p), sc16);
                acc = _mm512_dpwssd_epi32(acc, p16, sv);
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
#undef shl_tab
#undef hbit_tab
#undef sidx

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

// 512-bit Q6_K: per 128 values, one 64-byte load of the low nibbles (both 32-byte halves)
// and the 32 high-bit bytes in both halves (shifted by 2 in the upper one) give two 64-value
// chunks of u = 0..63. maddubs(u, x) -> int16 pairs, dpwssd with the signed 16-value-group
// scales; the -32 offset is sum_g scale_g * bsum_g. One float accumulate per block.
float vec_dot_q6_K_q8_K(int n, const block_q6_K *x, const block_q8_K *y) {
    const __m512i m4 = _mm512_set1_epi8(0x0F), m3 = _mm512_set1_epi8(3);
    __m512 accf = _mm512_setzero_ps();
    __m256 accm = _mm256_setzero_ps();
    for (int b = 0; b < n / QK_K; b++) {
        const __m256i sc8 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)x[b].scales));
        const __m512i sc16 = _mm512_castsi256_si512(sc8);
        __m512i acc = _mm512_setzero_si512();
        for (int h = 0; h < 2; h++) {
            const __m512i l = _mm512_loadu_si512((const void *)(x[b].ql + h * 64));
            const __m256i qh = _mm256_loadu_si256((const __m256i *)(x[b].qh + h * 32));
            const __m512i hb = _mm512_inserti64x4(_mm512_castsi256_si512(qh), _mm256_srli_epi16(qh, 2), 1);
            const __m512i uA = _mm512_or_si512(_mm512_and_si512(l, m4), _mm512_slli_epi16(_mm512_and_si512(hb, m3), 4));
            const __m512i uB = _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(l, 4), m4),
                                               _mm512_slli_epi16(_mm512_and_si512(_mm512_srli_epi16(hb, 4), m3), 4));
            const __m512i pA = _mm512_maddubs_epi16(uA, _mm512_loadu_si512((const void *)(y[b].qs + h * 128)));
            const __m512i pB = _mm512_maddubs_epi16(uB, _mm512_loadu_si512((const void *)(y[b].qs + h * 128 + 64)));
            acc = _mm512_dpwssd_epi32(acc, pA, _mm512_permutexvar_epi16(_mm512_load_si512((const void *)T_SIDX_Q3[2 * h]), sc16));
            acc = _mm512_dpwssd_epi32(acc, pB, _mm512_permutexvar_epi16(_mm512_load_si512((const void *)T_SIDX_Q3[2 * h + 1]), sc16));
        }
        const float dd = fp16_to_fp32(x[b].d) * y[b].d;
        accf = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc), _mm512_set1_ps(dd), accf);
        const __m256i mp = _mm256_madd_epi16(_mm256_loadu_si256((const __m256i *)y[b].bsums), sc8);
        accm = _mm256_fmadd_ps(_mm256_cvtepi32_ps(mp), _mm256_set1_ps(32.0f * dd), accm);
    }
    const __m128 m2 = _mm_add_ps(_mm256_castps256_ps128(accm), _mm256_extractf128_ps(accm, 1));
    const __m128 m1 = _mm_add_ps(m2, _mm_movehl_ps(m2, m2));
    return _mm512_reduce_add_ps(accf) - _mm_cvtss_f32(_mm_add_ss(m1, _mm_shuffle_ps(m1, m1, 1)));
}

// ---------- batched path ----------

void q3k_unpack_row(const block_q3_K *x, int n, q3k_unpacked *out) {
    for (int b = 0; b < n / QK_K; b++) {
        const block_q3_K *blk = &x[b];
        q3k_unpacked *o = &out[b];
        int8_t sc[16];
        q3k_scales(blk->scales, sc);
        for (int h = 0; h < 2; h++)
            for (int j = 0; j < 4; j++)
                for (int i = 0; i < 32; i++) {   // i = g*16 + l
                    int low = (blk->qs[h * 32 + i] >> (2 * j)) & 3;
                    int hi = (blk->hmask[i] >> (h * 4 + j)) & 1;
                    o->u[h * 128 + j * 32 + i] = (uint8_t)(low | (hi << 2));
                }
        // maddubs pair k of 64-value chunk c covers values c*64+2k, +1 -> group c*4 + k/8
        for (int c = 0; c < 4; c++)
            for (int k = 0; k < 32; k++) o->sx[c * 32 + k] = sc[c * 4 + k / 8];
        for (int g = 0; g < 16; g++) o->sc[g] = sc[g];
        o->d = fp16_to_fp32(blk->d);
    }
}

float vec_dot_q3k_unpacked(int n, const q3k_unpacked *w, const block_q8_K *y) {
    __m512 accf = _mm512_setzero_ps();
    for (int b = 0; b < n / QK_K; b++) {
        const q3k_unpacked *o = &w[b];
        __m512i acc = _mm512_setzero_si512();
        for (int c = 0; c < 4; c++) {
            __m512i u = _mm512_loadu_si512((const void *)(o->u + c * 64));
            __m512i xv = _mm512_loadu_si512((const void *)(y[b].qs + c * 64));
            __m512i p16 = _mm512_maddubs_epi16(u, xv);
            acc = _mm512_dpwssd_epi32(acc, p16, _mm512_loadu_si512((const void *)(o->sx + c * 32)));
        }
        __m256i cb = _mm256_madd_epi16(_mm256_loadu_si256((const __m256i *)o->sc),
                                       _mm256_loadu_si256((const __m256i *)y[b].bsums));
        acc = _mm512_sub_epi32(acc, _mm512_zextsi256_si512(_mm256_slli_epi32(cb, 2)));
        accf = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc), _mm512_set1_ps(o->d * y[b].d), accf);
    }
    return _mm512_reduce_add_ps(accf);
}

// ---------- batched matmul path ----------

void q3k_repack_row(const block_q3_K *x, int n, q3k_s8 *out) {
    const __m512i m3 = _mm512_set1_epi8(3), m1 = _mm512_set1_epi8(1), c4 = _mm512_set1_epi8(4);
    const __m256i c32 = _mm256_set1_epi16(32);
    const __m512i ones = _mm512_set1_epi8(1);
    // int16 lane k of 32-value piece m uses scale group 2m + k/16
    __m512i gidx[8];
    for (int m = 0; m < 8; m++) {
        int16_t idx[32];
        for (int k = 0; k < 32; k++) idx[k] = (int16_t)(2 * m + k / 16);
        gidx[m] = _mm512_loadu_si512(idx);
    }
    for (int b = 0; b < n / QK_K; b++) {
        const __m512i sc16 = _mm512_castsi256_si512(
            _mm256_sub_epi16(_mm256_cvtepu8_epi16(q3k_scales_raw(x[b].scales)), c32));
        const __m512i hm = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i *)x[b].hmask));
        __m512i wsum = _mm512_setzero_si512();
        for (int h = 0; h < 2; h++) {
            const __m512i qs = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i *)(x[b].qs + h * 32)));
            for (int jp = 0; jp < 2; jp++) {
                const int j = jp * 2, c = h * 4 + j;   // values c*32 .. c*32+63
                const __m512i sh_lo = _mm512_inserti64x4(_mm512_set1_epi16(2 * j), _mm256_set1_epi16(2 * j + 2), 1);
                const __m512i sh_hi = _mm512_inserti64x4(_mm512_set1_epi16(4 * h + j), _mm256_set1_epi16(4 * h + j + 1), 1);
                __m512i low = _mm512_and_si512(_mm512_srlv_epi16(qs, sh_lo), m3);
                __m512i hi = _mm512_and_si512(_mm512_srlv_epi16(hm, sh_hi), m1);
                __m512i q = _mm512_sub_epi8(_mm512_or_si512(low, _mm512_slli_epi16(hi, 2)), c4);  // -4..3
                __m256i w2[2];
                for (int half = 0; half < 2; half++) {
                    const int m = c + half;  // 32-value piece index
                    __m512i q16 = _mm512_cvtepi8_epi16(half ? _mm512_extracti64x4_epi64(q, 1) : _mm512_castsi512_si256(q));
                    __m512i p16 = _mm512_mullo_epi16(q16, _mm512_permutexvar_epi16(gidx[m], sc16));
                    w2[half] = _mm512_cvtsepi16_epi8(p16);
                }
                __m512i w = _mm512_inserti64x4(_mm512_castsi256_si512(w2[0]), w2[1], 1);
                _mm512_storeu_si512((void *)(out[b].w + c * 32), w);
                wsum = _mm512_dpbusd_epi32(wsum, ones, w);
            }
        }
        out[b].d = fp16_to_fp32(x[b].d);
        out[b].corr = -128 * _mm512_reduce_add_epi32(wsum);
    }
}

void quantize_row_u8(const float *x, uint8_t *xu, float *dx, int n) {
    for (int b = 0; b < n / QK_K; b++) {
        const float *v = x + b * QK_K;
        __m512 mx = _mm512_setzero_ps();
        for (int i = 0; i < QK_K; i += 16) mx = _mm512_max_ps(mx, _mm512_abs_ps(_mm512_loadu_ps(v + i)));
        float amax = _mm512_reduce_max_ps(mx);
        float d = amax / 127.f, id = amax > 0 ? 127.f / amax : 0.f;
        dx[b] = d;
        const __m512 vid = _mm512_set1_ps(id);
        const __m512i c128 = _mm512_set1_epi32(128);
        for (int i = 0; i < QK_K; i += 16) {
            __m512i q = _mm512_cvtps_epi32(_mm512_roundscale_ps(_mm512_mul_ps(_mm512_loadu_ps(v + i), vid), _MM_FROUND_TO_NEAREST_INT));
            _mm_storeu_si128((__m128i *)(xu + b * QK_K + i), _mm512_cvtepi32_epi8(_mm512_add_epi32(q, c128)));
        }
    }
}

void dot_q3k_s8_multi(int n, const q3k_s8 *w, int T, const uint8_t *xu, const float *dx,
                      float *out, int out_stride) {
    const int nb = n / QK_K;
    int t = 0;
    for (; t + 4 <= T; t += 4) {
        __m512 f0 = _mm512_setzero_ps(), f1 = f0, f2 = f0, f3 = f0;
        const uint8_t *x0 = xu + (size_t)t * n, *x1 = x0 + n, *x2 = x1 + n, *x3 = x2 + n;
        const float *d0 = dx + (size_t)t * nb;
        for (int b = 0; b < nb; b++) {
            const __m512i cr = _mm512_zextsi128_si512(_mm_cvtsi32_si128(w[b].corr));
            __m512i a0 = cr, a1 = cr, a2 = cr, a3 = cr;
            for (int k = 0; k < QK_K; k += 64) {
                const __m512i wv = _mm512_loadu_si512((const void *)(w[b].w + k));
                const int o = b * QK_K + k;
                a0 = _mm512_dpbusd_epi32(a0, _mm512_loadu_si512((const void *)(x0 + o)), wv);
                a1 = _mm512_dpbusd_epi32(a1, _mm512_loadu_si512((const void *)(x1 + o)), wv);
                a2 = _mm512_dpbusd_epi32(a2, _mm512_loadu_si512((const void *)(x2 + o)), wv);
                a3 = _mm512_dpbusd_epi32(a3, _mm512_loadu_si512((const void *)(x3 + o)), wv);
            }
            const float dw = w[b].d;
            f0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(a0), _mm512_set1_ps(dw * d0[b]), f0);
            f1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(a1), _mm512_set1_ps(dw * d0[nb + b]), f1);
            f2 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(a2), _mm512_set1_ps(dw * d0[2 * nb + b]), f2);
            f3 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(a3), _mm512_set1_ps(dw * d0[3 * nb + b]), f3);
        }
        out[(size_t)t * out_stride] = _mm512_reduce_add_ps(f0);
        out[(size_t)(t + 1) * out_stride] = _mm512_reduce_add_ps(f1);
        out[(size_t)(t + 2) * out_stride] = _mm512_reduce_add_ps(f2);
        out[(size_t)(t + 3) * out_stride] = _mm512_reduce_add_ps(f3);
    }
    for (; t < T; t++) {
        __m512 f = _mm512_setzero_ps();
        const uint8_t *x0 = xu + (size_t)t * n;
        for (int b = 0; b < nb; b++) {
            __m512i a = _mm512_zextsi128_si512(_mm_cvtsi32_si128(w[b].corr));
            for (int k = 0; k < QK_K; k += 64)
                a = _mm512_dpbusd_epi32(a, _mm512_loadu_si512((const void *)(x0 + b * QK_K + k)),
                                        _mm512_loadu_si512((const void *)(w[b].w + k)));
            f = _mm512_fmadd_ps(_mm512_cvtepi32_ps(a), _mm512_set1_ps(w[b].d * dx[(size_t)t * nb + b]), f);
        }
        out[(size_t)t * out_stride] = _mm512_reduce_add_ps(f);
    }
}

// ======================= Inkling 2 formats =======================

static inline void k4_scale_min(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else { *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}

void dequant_row_q8_0(const block_q8_0 *x, float *y, int64_t n) {
    for (int64_t b = 0; b < n / 32; b++) {
        const float d = fp16_to_fp32(x[b].d);
        for (int i = 0; i < 32; i++) y[b * 32 + i] = d * x[b].qs[i];
    }
}

void dequant_row_q5_0(const block_q5_0 *x, float *y, int64_t n) {
    for (int64_t b = 0; b < n / 32; b++) {
        const float d = fp16_to_fp32(x[b].d);
        uint32_t qh; memcpy(&qh, x[b].qh, 4);
        for (int j = 0; j < 16; j++) {
            const int x0 = (x[b].qs[j] & 0xF) | (((qh >> j) & 1) << 4);
            const int x1 = (x[b].qs[j] >> 4) | (((qh >> (j + 16)) & 1) << 4);
            y[b * 32 + j] = d * (x0 - 16);
            y[b * 32 + j + 16] = d * (x1 - 16);
        }
    }
}

void dequant_row_q4_K(const block_q4_K *x, float *y, int64_t n) {
    for (int64_t b = 0; b < n / QK_K; b++) {
        const float d = fp16_to_fp32(x[b].d), dmin = fp16_to_fp32(x[b].dmin);
        const uint8_t *q = x[b].qs;
        float *o = y + b * QK_K;
        for (int j = 0; j < 4; j++) {
            uint8_t s1, m1, s2, m2;
            k4_scale_min(2 * j, x[b].scales, &s1, &m1);
            k4_scale_min(2 * j + 1, x[b].scales, &s2, &m2);
            for (int l = 0; l < 32; l++) o[64 * j + l] = d * s1 * (q[32 * j + l] & 0xF) - dmin * m1;
            for (int l = 0; l < 32; l++) o[64 * j + 32 + l] = d * s2 * (q[32 * j + l] >> 4) - dmin * m2;
        }
    }
}

void dequant_row_q5_K(const block_q5_K *x, float *y, int64_t n) {
    for (int64_t b = 0; b < n / QK_K; b++) {
        const float d = fp16_to_fp32(x[b].d), dmin = fp16_to_fp32(x[b].dmin);
        const uint8_t *q = x[b].qs, *qh = x[b].qh;
        float *o = y + b * QK_K;
        for (int j = 0; j < 4; j++) {
            uint8_t s1, m1, s2, m2;
            k4_scale_min(2 * j, x[b].scales, &s1, &m1);
            k4_scale_min(2 * j + 1, x[b].scales, &s2, &m2);
            for (int l = 0; l < 32; l++)
                o[64 * j + l] = d * s1 * ((q[32 * j + l] & 0xF) | (((qh[l] >> (2 * j)) & 1) << 4)) - dmin * m1;
            for (int l = 0; l < 32; l++)
                o[64 * j + 32 + l] = d * s2 * ((q[32 * j + l] >> 4) | (((qh[l] >> (2 * j + 1)) & 1) << 4)) - dmin * m2;
        }
    }
}

void dequant_row_q4_0(const block_q4_0 *x, float *y, int64_t n) {
    for (int64_t b = 0; b < n / 32; b++) {
        const float d = fp16_to_fp32(x[b].d);
        for (int j = 0; j < 16; j++) {
            y[b * 32 + j] = d * ((x[b].qs[j] & 0xF) - 8);
            y[b * 32 + j + 16] = d * ((x[b].qs[j] >> 4) - 8);
        }
    }
}
void dequant_row_q4_1(const block_q4_1 *x, float *y, int64_t n) {
    for (int64_t b = 0; b < n / 32; b++) {
        const float d = fp16_to_fp32(x[b].d), m = fp16_to_fp32(x[b].m);
        for (int j = 0; j < 16; j++) {
            y[b * 32 + j] = d * (x[b].qs[j] & 0xF) + m;
            y[b * 32 + j + 16] = d * (x[b].qs[j] >> 4) + m;
        }
    }
}

int dequant_row(int type, const void *x, float *y, int64_t n) {
    switch (type) {
    case 0: memcpy(y, x, n * 4); return 0;
    case 2: dequant_row_q4_0(x, y, n); return 0;
    case 3: dequant_row_q4_1(x, y, n); return 0;
    case 6: dequant_row_q5_0(x, y, n); return 0;
    case 8: dequant_row_q8_0(x, y, n); return 0;
    case 11: dequant_row_q3_K(x, y, n); return 0;
    case 12: dequant_row_q4_K(x, y, n); return 0;
    case 13: dequant_row_q5_K(x, y, n); return 0;
    case 14: dequant_row_q6_K(x, y, n); return 0;
    case GT_Q4_KR: dequant_row_q4_KR(x, y, n); return 0;
    }
    return -1;
}

void quantize_row_a32(const float *x, act32 a, int64_t n) {
    for (int64_t b = 0; b < n / 32; b++) {
        const __m512 v0 = _mm512_loadu_ps(x + b * 32), v1 = _mm512_loadu_ps(x + b * 32 + 16);
        const float amax = _mm512_reduce_max_ps(_mm512_max_ps(_mm512_abs_ps(v0), _mm512_abs_ps(v1)));
        const float d = amax / 127.f, id = amax > 0 ? 127.f / amax : 0.f;
        a.d[b] = d;
        const __m512 vid = _mm512_set1_ps(id);
        __m512i q0 = _mm512_cvtps_epi32(_mm512_roundscale_ps(_mm512_mul_ps(v0, vid), _MM_FROUND_TO_NEAREST_INT));
        __m512i q1 = _mm512_cvtps_epi32(_mm512_roundscale_ps(_mm512_mul_ps(v1, vid), _MM_FROUND_TO_NEAREST_INT));
        a.s[b] = _mm512_reduce_add_epi32(_mm512_add_epi32(q0, q1));
        _mm_storeu_si128((__m128i *)(a.qs + b * 32), _mm512_cvtepi32_epi8(q0));
        _mm_storeu_si128((__m128i *)(a.qs + b * 32 + 16), _mm512_cvtepi32_epi8(q1));
    }
}

static inline __m512 two_scales(float a, float b) {
    return _mm512_insertf32x8(_mm512_set1_ps(a), _mm256_set1_ps(b), 1);
}

// s8 x s8 via |w| (u8) and x with w's sign: dpbusd lanes 0-7 = block 0, 8-15 = block 1
float vec_dot_q8_0_a32(int n, const block_q8_0 *x, act32 a) {
    const int nb = n / 32;
    __m512 acc = _mm512_setzero_ps();
    int b = 0;
    for (; b + 2 <= nb; b += 2) {
        const __m512i w = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)x[b].qs)),
                                             _mm256_loadu_si256((const __m256i *)x[b + 1].qs), 1);
        const __m512i xv = _mm512_loadu_si512((const void *)(a.qs + b * 32));
        const __m512i sx = _mm512_mask_sub_epi8(xv, _mm512_movepi8_mask(w), _mm512_setzero_si512(), xv);
        const __m512i dp = _mm512_dpbusd_epi32(_mm512_setzero_si512(), _mm512_abs_epi8(w), sx);
        acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(dp),
                              two_scales(fp16_to_fp32(x[b].d) * a.d[b], fp16_to_fp32(x[b + 1].d) * a.d[b + 1]), acc);
    }
    float sum = _mm512_reduce_add_ps(acc);
    for (; b < nb; b++) {
        int s = 0;
        for (int i = 0; i < 32; i++) s += x[b].qs[i] * a.qs[b * 32 + i];
        sum += fp16_to_fp32(x[b].d) * a.d[b] * s;
    }
    return sum;
}

static inline __m256i q5_0_u(const block_q5_0 *x) {
    const __m128i q = _mm_loadu_si128((const __m128i *)x->qs);
    const __m256i nib = _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(q, 4), _mm_set1_epi8(0xF)),
                                         _mm_and_si128(q, _mm_set1_epi8(0xF)));
    uint32_t qh; memcpy(&qh, x->qh, 4);
    return _mm256_or_si256(nib, _mm256_maskz_set1_epi8((__mmask32)qh, 0x10));
}

// 32 nibbles of a Q4_0/Q4_1 block in natural order (low nibbles = values 0..15, high = 16..31)
static inline __m256i q4_nib(const uint8_t *qs) {
    const __m128i q = _mm_loadu_si128((const __m128i *)qs);
    return _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(q, 4), _mm_set1_epi8(0xF)), _mm_and_si128(q, _mm_set1_epi8(0xF)));
}
// Q4_0: value = d * (u - 8): sum(u*x) - 8*sum(x). Four blocks per step: scales converted
// together (F16C), spread to their 8 lanes with one permute; correction kept as a vector.
float vec_dot_q4_0_a32(int n, const block_q4_0 *x, act32 a) {
    const int nb = n / 32;
    __m512 acc = _mm512_setzero_ps(), acc2 = _mm512_setzero_ps();
    __m128 cv = _mm_setzero_ps();
    const __m512i lo2 = _mm512_set_epi32(1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0);
    const __m512i hi2 = _mm512_set_epi32(3,3,3,3,3,3,3,3, 2,2,2,2,2,2,2,2);
    float corr = 0;
    int b = 0;
    for (; b + 4 <= nb; b += 4) {
        const __m512i u0 = _mm512_inserti64x4(_mm512_castsi256_si512(q4_nib(x[b].qs)), q4_nib(x[b + 1].qs), 1);
        const __m512i u1 = _mm512_inserti64x4(_mm512_castsi256_si512(q4_nib(x[b + 2].qs)), q4_nib(x[b + 3].qs), 1);
        const __m512i d0 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), u0, _mm512_loadu_si512((const void *)(a.qs + b * 32)));
        const __m512i d1 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), u1, _mm512_loadu_si512((const void *)(a.qs + b * 32 + 64)));
        const __m128i h = _mm_setr_epi16(x[b].d, x[b + 1].d, x[b + 2].d, x[b + 3].d, 0, 0, 0, 0);
        const __m128 s4 = _mm_mul_ps(_mm_cvtph_ps(h), _mm_loadu_ps(a.d + b));
        cv = _mm_fmadd_ps(s4, _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i *)(a.s + b))), cv);
        const __m512 s16 = _mm512_castps128_ps512(s4);
        acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(d0), _mm512_permutexvar_ps(lo2, s16), acc);
        acc2 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(d1), _mm512_permutexvar_ps(hi2, s16), acc2);
    }
    {   // fold the vector correction in, then the 2-block tail as before
        __m128 t = _mm_add_ps(cv, _mm_movehl_ps(cv, cv));
        t = _mm_add_ss(t, _mm_shuffle_ps(t, t, 1));
        corr = _mm_cvtss_f32(t);
        acc = _mm512_add_ps(acc, acc2);
    }
    for (; b + 2 <= nb; b += 2) {
        const __m512i u = _mm512_inserti64x4(_mm512_castsi256_si512(q4_nib(x[b].qs)), q4_nib(x[b + 1].qs), 1);
        const __m512i dp = _mm512_dpbusd_epi32(_mm512_setzero_si512(), u, _mm512_loadu_si512((const void *)(a.qs + b * 32)));
        const float s0 = fp16_to_fp32(x[b].d) * a.d[b], s1 = fp16_to_fp32(x[b + 1].d) * a.d[b + 1];
        acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(dp), two_scales(s0, s1), acc);
        corr += s0 * a.s[b] + s1 * a.s[b + 1];
    }
    float sum = _mm512_reduce_add_ps(acc) - 8 * corr;
    for (; b < nb; b++) {
        float y[32];
        dequant_row_q4_0(&x[b], y, 32);
        for (int i = 0; i < 32; i++) sum += y[i] * a.d[b] * a.qs[b * 32 + i];
    }
    return sum;
}
// Q4_1: value = d * u + m: d*sum(u*x) + m*sum(x)
float vec_dot_q4_1_a32(int n, const block_q4_1 *x, act32 a) {
    const int nb = n / 32;
    __m512 acc = _mm512_setzero_ps();
    float mins = 0;
    int b = 0;
    for (; b + 2 <= nb; b += 2) {
        const __m512i u = _mm512_inserti64x4(_mm512_castsi256_si512(q4_nib(x[b].qs)), q4_nib(x[b + 1].qs), 1);
        const __m512i dp = _mm512_dpbusd_epi32(_mm512_setzero_si512(), u, _mm512_loadu_si512((const void *)(a.qs + b * 32)));
        const float s0 = fp16_to_fp32(x[b].d) * a.d[b], s1 = fp16_to_fp32(x[b + 1].d) * a.d[b + 1];
        acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(dp), two_scales(s0, s1), acc);
        mins += fp16_to_fp32(x[b].m) * a.d[b] * a.s[b] + fp16_to_fp32(x[b + 1].m) * a.d[b + 1] * a.s[b + 1];
    }
    float sum = _mm512_reduce_add_ps(acc) + mins;
    for (; b < nb; b++) {
        float y[32];
        dequant_row_q4_1(&x[b], y, 32);
        for (int i = 0; i < 32; i++) sum += y[i] * a.d[b] * a.qs[b * 32 + i];
    }
    return sum;
}

// u = q + 16 in 0..31: sum(u*x) - 16*sum(x)
float vec_dot_q5_0_a32(int n, const block_q5_0 *x, act32 a) {
    const int nb = n / 32;
    __m512 acc = _mm512_setzero_ps();
    float corr = 0;
    int b = 0;
    for (; b + 2 <= nb; b += 2) {
        const __m512i u = _mm512_inserti64x4(_mm512_castsi256_si512(q5_0_u(&x[b])), q5_0_u(&x[b + 1]), 1);
        const __m512i dp = _mm512_dpbusd_epi32(_mm512_setzero_si512(), u, _mm512_loadu_si512((const void *)(a.qs + b * 32)));
        const float s0 = fp16_to_fp32(x[b].d) * a.d[b], s1 = fp16_to_fp32(x[b + 1].d) * a.d[b + 1];
        acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(dp), two_scales(s0, s1), acc);
        corr += s0 * a.s[b] + s1 * a.s[b + 1];
    }
    float sum = _mm512_reduce_add_ps(acc) - 16 * corr;
    for (; b < nb; b++) {
        float y[32];
        dequant_row_q5_0(&x[b], y, 32);
        for (int i = 0; i < 32; i++) sum += y[i] * a.d[b] * a.qs[b * 32 + i];
    }
    return sum;
}

// dot of 64 unsigned quants (two 32-value sub-blocks) with x, scaled per sub-block
static inline __m512i k4_chunk(__m512i u, __m512i xv, int sa, int sb) {
    const __m512i dp = _mm512_dpbusd_epi32(_mm512_setzero_si512(), u, xv);
    const __m512i sc = _mm512_inserti64x4(_mm512_set1_epi32(sa), _mm256_set1_epi32(sb), 1);
    return _mm512_mullo_epi32(dp, sc);
}

// 8 scales and 8 mins (6-bit) of a Q4_K/Q5_K block, branch-free (same packing as ggml)
static inline void k4_unpack(const uint8_t *scales, uint32_t ut[4]) {
    const uint32_t km1 = 0x3f3f3f3f, km2 = 0x0f0f0f0f, km3 = 0x03030303;
    memcpy(ut, scales, 12);
    ut[3] = ((ut[2] >> 4) & km2) | (((ut[1] >> 6) & km3) << 4);
    const uint32_t a = ut[1] & km1;
    ut[1] = (ut[2] & km2) | (((ut[0] >> 6) & km3) << 4);
    ut[2] = a;
    ut[0] &= km1;
}

// the same, in registers: storing 4 x u32 and reloading them as one 128-bit value stalls
// store forwarding on every block. Reads 16 bytes from `scales` (the 12-byte field is always
// followed by more of the block).
static inline __m128i k4_scales_v(const uint8_t *scales) {
    const __m128i a = _mm_loadu_si128((const __m128i *)scales);
    const __m128i P = _mm_shuffle_epi32(a, _MM_SHUFFLE(2, 1, 2, 0));   // u0 u2 u1 u2
    const __m128i Q = _mm_shuffle_epi32(a, _MM_SHUFFLE(1, 1, 0, 0));   // u0 u0 u1 u1
    const __m128i lo = _mm_and_si128(P, _mm_set1_epi32(0x3f3f3f3f));
    const __m128i hi = _mm_or_si128(_mm_and_si128(_mm_srlv_epi32(P, _mm_setr_epi32(0, 0, 0, 4)), _mm_set1_epi32(0x0f0f0f0f)),
                                    _mm_slli_epi32(_mm_and_si128(_mm_srli_epi32(Q, 6), _mm_set1_epi32(0x03030303)), 4));
    return _mm_blend_epi32(lo, hi, 0xA);
}

// u (0..15 or 0..31) x q8 via maddubs, scaled per 32-value sub-block with madd; one
// float accumulate per block, one reduction per row. mins: madd(bsums, mins per pair).
float vec_dot_q4_K_q8_K(int n, const block_q4_K *x, const block_q8_K *y) {
    const __m256i m4 = _mm256_set1_epi8(0xF);
    // int16 lane k of chunk j (64 values = 32 pairs): sub-block 2j for k < 16, else 2j+1
    const __m128i dup = _mm_setr_epi8(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7);
    __m512 accf = _mm512_setzero_ps();
    __m256 accm = _mm256_setzero_ps();
    for (int b = 0; b < n / QK_K; b++) {
        const __m128i sm = k4_scales_v(x[b].scales);          // bytes 0-7 scales, 8-15 mins
        const __m512i sc16 = _mm512_castsi256_si512(_mm256_cvtepu8_epi16(sm));
        __m512i acc = _mm512_setzero_si512();
        for (int j = 0; j < 4; j++) {
            const __m256i q = _mm256_loadu_si256((const __m256i *)(x[b].qs + 32 * j));
            const __m512i u = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_and_si256(q, m4)),
                                                 _mm256_and_si256(_mm256_srli_epi16(q, 4), m4), 1);
            const __m512i p16 = _mm512_maddubs_epi16(u, _mm512_loadu_si512((const void *)(y[b].qs + 64 * j)));
            acc = _mm512_dpwssd_epi32(acc, p16, _mm512_permutexvar_epi16(_mm512_load_si512((const void *)T_SIDX_K4[j]), sc16));
        }
        const float dy = y[b].d;
        accf = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc), _mm512_set1_ps(fp16_to_fp32(x[b].d) * dy), accf);
        // mins: sum_j m_j * (bsum_2j + bsum_2j+1)
        const __m256i mn = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(_mm_srli_si128(sm, 8), dup));
        const __m256i mp = _mm256_madd_epi16(_mm256_loadu_si256((const __m256i *)y[b].bsums), mn);
        accm = _mm256_fmadd_ps(_mm256_cvtepi32_ps(mp), _mm256_set1_ps(fp16_to_fp32(x[b].dmin) * dy), accm);
    }
    const __m128 m2 = _mm_add_ps(_mm256_castps256_ps128(accm), _mm256_extractf128_ps(accm, 1));
    const __m128 m1 = _mm_add_ps(m2, _mm_movehl_ps(m2, m2));
    return _mm512_reduce_add_ps(accf) - _mm_cvtss_f32(_mm_add_ss(m1, _mm_shuffle_ps(m1, m1, 1)));
}

float vec_dot_q5_K_q8_K(int n, const block_q5_K *x, const block_q8_K *y) {
    const __m256i m4 = _mm256_set1_epi8(0xF), c16 = _mm256_set1_epi8(0x10);
    const __m128i dup = _mm_setr_epi8(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7);
    __m512 accf = _mm512_setzero_ps();
    __m256 accm = _mm256_setzero_ps();
    for (int b = 0; b < n / QK_K; b++) {
        const __m128i sm = k4_scales_v(x[b].scales);
        const __m512i sc16 = _mm512_castsi256_si512(_mm256_cvtepu8_epi16(sm));
        const __m256i hm = _mm256_loadu_si256((const __m256i *)x[b].qh);
        __m512i acc = _mm512_setzero_si512();
        for (int j = 0; j < 4; j++) {
            const __m256i q = _mm256_loadu_si256((const __m256i *)(x[b].qs + 32 * j));
            const __m256i lo = _mm256_or_si256(_mm256_and_si256(q, m4),
                _mm256_maskz_mov_epi8(_mm256_test_epi8_mask(hm, _mm256_set1_epi8((char)(1 << (2 * j)))), c16));
            const __m256i hi = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q, 4), m4),
                _mm256_maskz_mov_epi8(_mm256_test_epi8_mask(hm, _mm256_set1_epi8((char)(1 << (2 * j + 1)))), c16));
            const __m512i u = _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
            const __m512i p16 = _mm512_maddubs_epi16(u, _mm512_loadu_si512((const void *)(y[b].qs + 64 * j)));
            acc = _mm512_dpwssd_epi32(acc, p16, _mm512_permutexvar_epi16(_mm512_load_si512((const void *)T_SIDX_K4[j]), sc16));
        }
        const float dy = y[b].d;
        accf = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc), _mm512_set1_ps(fp16_to_fp32(x[b].d) * dy), accf);
        const __m256i mn = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(_mm_srli_si128(sm, 8), dup));
        const __m256i mp = _mm256_madd_epi16(_mm256_loadu_si256((const __m256i *)y[b].bsums), mn);
        accm = _mm256_fmadd_ps(_mm256_cvtepi32_ps(mp), _mm256_set1_ps(fp16_to_fp32(x[b].dmin) * dy), accm);
    }
    const __m128 m2 = _mm_add_ps(_mm256_castps256_ps128(accm), _mm256_extractf128_ps(accm, 1));
    const __m128 m1 = _mm_add_ps(m2, _mm_movehl_ps(m2, m2));
    return _mm512_reduce_add_ps(accf) - _mm_cvtss_f32(_mm_add_ss(m1, _mm_shuffle_ps(m1, m1, 1)));
}



// ---------- two activation vectors per weight row (speculative verify) ----------
// Same math as the single-vector kernels; the weight unpacking is shared.

static inline float hsum256_ps(__m256 v) {
    const __m128 m2 = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    const __m128 m1 = _mm_add_ps(m2, _mm_movehl_ps(m2, m2));
    return _mm_cvtss_f32(_mm_add_ss(m1, _mm_shuffle_ps(m1, m1, 1)));
}

void vec_dot2_q3_K_q8_K(int n, const block_q3_K *x, const block_q8_K *y0, const block_q8_K *y1, float *r0, float *r1) {
    const __m512i m3 = _mm512_set1_epi8(3), c4 = _mm512_set1_epi8(4);
    __m512 f0 = _mm512_setzero_ps(), f1 = f0;
    const __m256i c32 = _mm256_set1_epi16(32);
    for (int b = 0; b < n / QK_K; b++) {
        const __m256i s16 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(q3k_scales_raw(x[b].scales)), c32);
        const __m512i sc16 = _mm512_castsi256_si512(s16);
        const __m512i hm = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i *)x[b].hmask));
        __m512i a0 = _mm512_setzero_si512(), a1 = a0;
        for (int h = 0; h < 2; h++) {
            const __m512i qs = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i *)(x[b].qs + h * 32)));
            for (int jp = 0; jp < 2; jp++) {
                const int j = jp * 2, c = h * 4 + j, p = c / 2;
                __m512i low = _mm512_and_si512(_mm512_srlv_epi16(qs, _mm512_load_si512((const void *)T_SHL_Q3[j])), m3);
                __m512i u = _mm512_or_si512(low, _mm512_maskz_mov_epi8(_mm512_test_epi8_mask(hm, _mm512_load_si512((const void *)T_HBIT_Q3[h * 4 + j])), c4));
                const __m512i sv = _mm512_permutexvar_epi16(_mm512_load_si512((const void *)T_SIDX_Q3[p]), sc16);
                a0 = _mm512_dpwssd_epi32(a0, _mm512_maddubs_epi16(u, _mm512_loadu_si512((const void *)(y0[b].qs + c * 32))), sv);
                a1 = _mm512_dpwssd_epi32(a1, _mm512_maddubs_epi16(u, _mm512_loadu_si512((const void *)(y1[b].qs + c * 32))), sv);
            }
        }
        const __m256i cb0 = _mm256_madd_epi16(s16, _mm256_loadu_si256((const __m256i *)y0[b].bsums));
        const __m256i cb1 = _mm256_madd_epi16(s16, _mm256_loadu_si256((const __m256i *)y1[b].bsums));
        a0 = _mm512_sub_epi32(a0, _mm512_zextsi256_si512(_mm256_slli_epi32(cb0, 2)));
        a1 = _mm512_sub_epi32(a1, _mm512_zextsi256_si512(_mm256_slli_epi32(cb1, 2)));
        const float dx = fp16_to_fp32(x[b].d);
        f0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(a0), _mm512_set1_ps(dx * y0[b].d), f0);
        f1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(a1), _mm512_set1_ps(dx * y1[b].d), f1);
    }
    *r0 = _mm512_reduce_add_ps(f0);
    *r1 = _mm512_reduce_add_ps(f1);
}

// Q4_K (hbits = 0) and Q5_K (hbits = 1)
static inline void k45_dot2(int n, const void *xv, int hbits, const block_q8_K *y0, const block_q8_K *y1, float *r0, float *r1) {
    const __m256i m4 = _mm256_set1_epi8(0xF), c16 = _mm256_set1_epi8(0x10);
    const __m128i dup = _mm_setr_epi8(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7);
    __m512 f0 = _mm512_setzero_ps(), f1 = f0;
    __m256 g0 = _mm256_setzero_ps(), g1 = g0;
    const block_q4_K *x4 = xv;
    const block_q5_K *x5 = xv;
    for (int b = 0; b < n / QK_K; b++) {
        const uint8_t *scales = hbits ? x5[b].scales : x4[b].scales;
        const uint8_t *qs = hbits ? x5[b].qs : x4[b].qs;
        const uint16_t d16 = hbits ? x5[b].d : x4[b].d, dm16 = hbits ? x5[b].dmin : x4[b].dmin;
        const __m128i sm = k4_scales_v(scales);
        const __m512i sc16 = _mm512_castsi256_si512(_mm256_cvtepu8_epi16(sm));
        const __m256i hm = hbits ? _mm256_loadu_si256((const __m256i *)x5[b].qh) : _mm256_setzero_si256();
        __m512i a0 = _mm512_setzero_si512(), a1 = a0;
        for (int j = 0; j < 4; j++) {
            const __m256i q = _mm256_loadu_si256((const __m256i *)(qs + 32 * j));
            __m256i lo = _mm256_and_si256(q, m4), hi = _mm256_and_si256(_mm256_srli_epi16(q, 4), m4);
            if (hbits) {
                lo = _mm256_or_si256(lo, _mm256_maskz_mov_epi8(_mm256_test_epi8_mask(hm, _mm256_set1_epi8((char)(1 << (2 * j)))), c16));
                hi = _mm256_or_si256(hi, _mm256_maskz_mov_epi8(_mm256_test_epi8_mask(hm, _mm256_set1_epi8((char)(1 << (2 * j + 1)))), c16));
            }
            const __m512i u = _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
            const __m512i sv = _mm512_permutexvar_epi16(_mm512_load_si512((const void *)T_SIDX_K4[j]), sc16);
            a0 = _mm512_dpwssd_epi32(a0, _mm512_maddubs_epi16(u, _mm512_loadu_si512((const void *)(y0[b].qs + 64 * j))), sv);
            a1 = _mm512_dpwssd_epi32(a1, _mm512_maddubs_epi16(u, _mm512_loadu_si512((const void *)(y1[b].qs + 64 * j))), sv);
        }
        const float d = fp16_to_fp32(d16), dmin = fp16_to_fp32(dm16);
        f0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(a0), _mm512_set1_ps(d * y0[b].d), f0);
        f1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(a1), _mm512_set1_ps(d * y1[b].d), f1);
        const __m256i mn = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(_mm_srli_si128(sm, 8), dup));
        g0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_loadu_si256((const __m256i *)y0[b].bsums), mn)), _mm256_set1_ps(dmin * y0[b].d), g0);
        g1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_loadu_si256((const __m256i *)y1[b].bsums), mn)), _mm256_set1_ps(dmin * y1[b].d), g1);
    }
    *r0 = _mm512_reduce_add_ps(f0) - hsum256_ps(g0);
    *r1 = _mm512_reduce_add_ps(f1) - hsum256_ps(g1);
}
void vec_dot2_q4_K_q8_K(int n, const block_q4_K *x, const block_q8_K *y0, const block_q8_K *y1, float *r0, float *r1) { k45_dot2(n, x, 0, y0, y1, r0, r1); }
void vec_dot2_q5_K_q8_K(int n, const block_q5_K *x, const block_q8_K *y0, const block_q8_K *y1, float *r0, float *r1) { k45_dot2(n, x, 1, y0, y1, r0, r1); }

// ---------- Q4_K repacked (see quants.h) ----------

void q4k_repack_kr(void *blocks, uint64_t n_blocks) {
    block_q4_K *x = blocks;
    for (uint64_t b = 0; b < n_blocks; b++) {
        const uint8_t *qs = x[b].qs;
        uint8_t nib[256], nq[128];
        for (int j = 0; j < 4; j++)
            for (int l = 0; l < 32; l++) { nib[64 * j + l] = qs[32 * j + l] & 15; nib[64 * j + 32 + l] = qs[32 * j + l] >> 4; }
        for (int h = 0; h < 2; h++)
            for (int i = 0; i < 64; i++) {
                const int s = i / 8, r = i % 8;
                nq[64 * h + i] = (uint8_t)(nib[s * 32 + 16 * h + r] | nib[s * 32 + 16 * h + 8 + r] << 4);
            }
        memcpy(x[b].qs, nq, 128);
    }
}

void q8k_permute_kr(const block_q8_K *y, block_q8_K *yp, int n) {
    // chunk c (= 2h for low nibbles, 2h + 1 for high), bytes 8s..8s+7 <- values 32s + 8c ..
    for (int b = 0; b < n / QK_K; b++) {
        yp[b].d = y[b].d;
        memcpy(yp[b].bsums, y[b].bsums, sizeof y[b].bsums);
        for (int c = 0; c < 4; c++)
            for (int sb = 0; sb < 8; sb++) memcpy(yp[b].qs + 64 * c + 8 * sb, y[b].qs + 32 * sb + 8 * c, 8);
    }
}

static int16_t T_SIDX_KR[32] __attribute__((aligned(64)));
__attribute__((constructor)) static void kr_tables_init(void) { for (int k = 0; k < 32; k++) T_SIDX_KR[k] = (int16_t)(k / 4); }

float vec_dot_q4_KR_q8_KR(int n, const block_q4_K *x, const block_q8_K *y) {
    const __m512i m4 = _mm512_set1_epi8(0xF);
    const __m128i dup = _mm_setr_epi8(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7);
    const __m512i sidx = _mm512_load_si512((const void *)T_SIDX_KR);
    __m512 accf = _mm512_setzero_ps();
    __m256 accm = _mm256_setzero_ps();
    for (int b = 0; b < n / QK_K; b++) {
        const __m128i sm = k4_scales_v(x[b].scales);
        const __m512i sc = _mm512_permutexvar_epi16(sidx, _mm512_castsi256_si512(_mm256_cvtepu8_epi16(sm)));
        const __m512i L0 = _mm512_loadu_si512((const void *)x[b].qs), L1 = _mm512_loadu_si512((const void *)(x[b].qs + 64));
        // |pair| <= 2 * 15 * 127, four chunks summed in int16 stay below 2^15
        __m512i p = _mm512_maddubs_epi16(_mm512_and_si512(L0, m4), _mm512_loadu_si512((const void *)y[b].qs));
        p = _mm512_add_epi16(p, _mm512_maddubs_epi16(_mm512_and_si512(_mm512_srli_epi16(L0, 4), m4), _mm512_loadu_si512((const void *)(y[b].qs + 64))));
        p = _mm512_add_epi16(p, _mm512_maddubs_epi16(_mm512_and_si512(L1, m4), _mm512_loadu_si512((const void *)(y[b].qs + 128))));
        p = _mm512_add_epi16(p, _mm512_maddubs_epi16(_mm512_and_si512(_mm512_srli_epi16(L1, 4), m4), _mm512_loadu_si512((const void *)(y[b].qs + 192))));
        const __m512i acc = _mm512_madd_epi16(p, sc);
        const float dy = y[b].d;
        accf = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc), _mm512_set1_ps(fp16_to_fp32(x[b].d) * dy), accf);
        const __m256i mn = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(_mm_srli_si128(sm, 8), dup));
        const __m256i mp = _mm256_madd_epi16(_mm256_loadu_si256((const __m256i *)y[b].bsums), mn);
        accm = _mm256_fmadd_ps(_mm256_cvtepi32_ps(mp), _mm256_set1_ps(fp16_to_fp32(x[b].dmin) * dy), accm);
    }
    const __m128 m2 = _mm_add_ps(_mm256_castps256_ps128(accm), _mm256_extractf128_ps(accm, 1));
    const __m128 m1 = _mm_add_ps(m2, _mm_movehl_ps(m2, m2));
    return _mm512_reduce_add_ps(accf) - _mm_cvtss_f32(_mm_add_ss(m1, _mm_shuffle_ps(m1, m1, 1)));
}

void dequant_row_q4_KR(const void *xv, float *y, int64_t n) {
    const block_q4_K *x = xv;
    for (int64_t b = 0; b < n / QK_K; b++) {
        block_q4_K t = x[b];
        uint8_t nib[256];
        for (int v = 0; v < 256; v++) {
            const int s = v / 32, w = v % 32, h = w / 16, rem = w % 16;
            const uint8_t byte = x[b].qs[64 * h + 8 * s + (rem & 7)];
            nib[v] = rem < 8 ? byte & 15 : byte >> 4;
        }
        for (int j = 0; j < 4; j++)
            for (int l = 0; l < 32; l++) t.qs[32 * j + l] = (uint8_t)(nib[64 * j + l] | nib[64 * j + 32 + l] << 4);
        dequant_row_q4_K(&t, y + b * QK_K, QK_K);
    }
}

// ---------- batched exact dot: many rows x many tokens (prefill) ----------
// int32 lane k of every 64-value chunk belongs to one scale group, so u8 weights (as stored,
// unpacked in registers) x int8 activations accumulate with dpbusd and the scales apply in
// float once per block (Q4_KR) or per chunk (Q6_K). Offsets (Q4_K mins, the Q6_K -32) use the
// activation's 16-value sums. Activations: q8_K, KR-permuted for Q4_KR, natural for Q6_K.
//   xs[t][b * 16 + k] = bsums[k] * d of token t, block b
static inline void q4kr_row_block(const block_q4_K *x, __m512i u[4], __m512 *sv, __m512 *ov) {
    const __m512i m4 = _mm512_set1_epi8(0xF);
    const __m128i sm = k4_scales_v(x->scales);
    // lane k -> sub-block k / 2
    const __m128i dup = _mm_setr_epi8(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7);
    const __m512 sc = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_shuffle_epi8(sm, dup)));
    const __m512 mn = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_shuffle_epi8(_mm_srli_si128(sm, 8), dup)));
    *sv = _mm512_mul_ps(sc, _mm512_set1_ps(fp16_to_fp32(x->d)));
    *ov = _mm512_mul_ps(mn, _mm512_set1_ps(fp16_to_fp32(x->dmin)));
    const __m512i L0 = _mm512_loadu_si512((const void *)x->qs), L1 = _mm512_loadu_si512((const void *)(x->qs + 64));
    u[0] = _mm512_and_si512(L0, m4); u[1] = _mm512_and_si512(_mm512_srli_epi16(L0, 4), m4);
    u[2] = _mm512_and_si512(L1, m4); u[3] = _mm512_and_si512(_mm512_srli_epi16(L1, 4), m4);
}

static int32_t T_Q6_LANE[4][16] __attribute__((aligned(64)));   // chunk c, lane k -> group 4c + k/4
__attribute__((constructor)) static void q6_lane_init(void) { for (int c = 0; c < 4; c++) for (int k = 0; k < 16; k++) T_Q6_LANE[c][k] = 4 * c + k / 4; }

static inline void q6k_row_block(const block_q6_K *x, __m512i u[4], __m512 sv[4], __m512 *ov) {
    const __m512i m4 = _mm512_set1_epi8(0x0F), m3 = _mm512_set1_epi8(3);
    const __m512 sg = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)x->scales))),
                                    _mm512_set1_ps(fp16_to_fp32(x->d)));
    for (int c = 0; c < 4; c++) sv[c] = _mm512_permutexvar_ps(_mm512_load_si512((const void *)T_Q6_LANE[c]), sg);
    *ov = _mm512_mul_ps(sg, _mm512_set1_ps(32.0f));
    for (int h = 0; h < 2; h++) {
        const __m512i l = _mm512_loadu_si512((const void *)(x->ql + h * 64));
        const __m256i qh = _mm256_loadu_si256((const __m256i *)(x->qh + h * 32));
        const __m512i hb = _mm512_inserti64x4(_mm512_castsi256_si512(qh), _mm256_srli_epi16(qh, 2), 1);
        u[2 * h] = _mm512_or_si512(_mm512_and_si512(l, m4), _mm512_slli_epi16(_mm512_and_si512(hb, m3), 4));
        u[2 * h + 1] = _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(l, 4), m4),
                                       _mm512_slli_epi16(_mm512_and_si512(_mm512_srli_epi16(hb, 4), m3), 4));
    }
}

// out[t * ldy + r] for rows r in [0, nr) (nr <= 2) and tokens t in [0, nt) (nt <= 4)
static inline __attribute__((always_inline)) void kq_tile(int type, const uint8_t *const *rows, int nr, int n, int nt, const block_q8_K *const *x,
                    const float *const *xs, float *out, int ldy) {
    const int nb = n / QK_K;
    __m512 F[2][4];
    for (int a = 0; a < 2; a++) for (int t = 0; t < 4; t++) F[a][t] = _mm512_setzero_ps();
    for (int b = 0; b < nb; b++) {
        __m512i u[2][4];
        __m512 sv[2][4], ov[2];
        for (int a = 0; a < nr; a++) {
            if (type == GT_Q4_KR) q4kr_row_block((const block_q4_K *)rows[a] + b, u[a], &sv[a][0], &ov[a]);
            else q6k_row_block((const block_q6_K *)rows[a] + b, u[a], sv[a], &ov[a]);
        }
        for (int t = 0; t < nt; t++) {
            const int8_t *q = x[t][b].qs;
            const __m512i x0 = _mm512_loadu_si512((const void *)q), x1 = _mm512_loadu_si512((const void *)(q + 64));
            const __m512i x2 = _mm512_loadu_si512((const void *)(q + 128)), x3 = _mm512_loadu_si512((const void *)(q + 192));
            const __m512 dx = _mm512_set1_ps(x[t][b].d), xsv = _mm512_loadu_ps(xs[t] + b * 16);
            for (int a = 0; a < nr; a++) {
                __m512 f;
                if (type == GT_Q4_KR) {
                    __m512i acc = _mm512_dpbusd_epi32(_mm512_setzero_si512(), u[a][0], x0);
                    acc = _mm512_dpbusd_epi32(acc, u[a][1], x1);
                    acc = _mm512_dpbusd_epi32(acc, u[a][2], x2);
                    acc = _mm512_dpbusd_epi32(acc, u[a][3], x3);
                    f = _mm512_mul_ps(_mm512_cvtepi32_ps(acc), sv[a][0]);
                } else {
                    f = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_dpbusd_epi32(_mm512_setzero_si512(), u[a][0], x0)), sv[a][0]);
                    f = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_dpbusd_epi32(_mm512_setzero_si512(), u[a][1], x1)), sv[a][1], f);
                    f = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_dpbusd_epi32(_mm512_setzero_si512(), u[a][2], x2)), sv[a][2], f);
                    f = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_dpbusd_epi32(_mm512_setzero_si512(), u[a][3], x3)), sv[a][3], f);
                }
                F[a][t] = _mm512_fmadd_ps(f, dx, F[a][t]);
                F[a][t] = _mm512_fnmadd_ps(ov[a], xsv, F[a][t]);
            }
        }
    }
    for (int a = 0; a < nr; a++)
        for (int t = 0; t < nt; t++) out[(size_t)t * ldy + a] = _mm512_reduce_add_ps(F[a][t]);
}

int kq_supported(int type) { return type == GT_Q4_KR || type == 14 /* Q6_K */; }

void kq_xs(const block_q8_K *x, int n, float *xs) {
    for (int b = 0; b < n / QK_K; b++) {
        const __m512 v = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm256_loadu_si256((const __m256i *)x[b].bsums)));
        _mm512_storeu_ps(xs + b * 16, _mm512_mul_ps(v, _mm512_set1_ps(x[b].d)));
    }
}

void kq_matmul(int type, const uint8_t *base, uint64_t rb, int rows, int cols, int T,
               const block_q8_K *const *x, const float *const *xs, float *Y, int ldy) {
    const int npair = (rows + 1) / 2;
    #pragma omp parallel for schedule(dynamic, 4)
    for (int p = 0; p < npair; p++) {
        const int r0 = 2 * p, nr = rows - r0 < 2 ? rows - r0 : 2;
        const uint8_t *rw[2] = {base + (uint64_t)r0 * rb, base + (uint64_t)(r0 + nr - 1) * rb};
        for (int t = 0; t < T; t += 4) {
            const int nt = T - t < 4 ? T - t : 4;
            float *o = Y + (size_t)t * ldy + r0;
            // full tiles with compile-time sizes (accumulators stay in registers)
            if (nr == 2 && nt == 4) {
                if (type == GT_Q4_KR) kq_tile(GT_Q4_KR, rw, 2, cols, 4, x + t, xs + t, o, ldy);
                else kq_tile(14, rw, 2, cols, 4, x + t, xs + t, o, ldy);
            } else kq_tile(type, rw, nr, cols, nt, x + t, xs + t, o, ldy);
        }
    }
}
