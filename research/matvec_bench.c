// Phase 1: can a hand-written 4-bit matvec reach RAM bandwidth on this CPU?
// Layout (our own, not GGUF): per row, blocks of 32 weights:
//   uint8 qs[16]  (two 4-bit weights per byte, values 0..15, centered at 8)
//   float16-ish scale stored separately as float (fine for a bandwidth test)
// Activations quantized to int8 per 32-block (like llama.cpp Q8_0 path).
// Kernel: AVX-512 VNNI (vpdpbusd): u8 x s8 -> s32 dot products.
//
// Build: gcc -O3 -march=native -fopenmp matvec_bench.c -o matvec_bench
#include <immintrin.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define QK 32

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

// rows x cols weight matrix, cols % 64 == 0
typedef struct {
    int rows, cols;
    uint8_t *qs;    // rows * cols/2 bytes
    float *scale;   // rows * cols/QK
} q4mat;

// x: int8 activations (cols), xs: per-block activation scales (cols/QK)
// sum_x: per-block sum of int8 activations (to undo the +8 offset)
static void matvec_q4_vnni(const q4mat *m, const int8_t *x, const float *xs,
                           const float *xcorr, float *y) {
    // xcorr[b] = 8 * sum(x in block b) * xs[b]  (undoes the +8 weight offset)
    const int nb = m->cols / QK;
    const __m512i idx01 = _mm512_set_epi32(1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0);
    const __m512i idx23 = _mm512_set_epi32(3,3,3,3,3,3,3,3, 2,2,2,2,2,2,2,2);
    const __m512i lowmask = _mm512_set1_epi8(0x0F);
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < m->rows; r++) {
        const uint8_t *q = m->qs + (size_t)r * (m->cols / 2);
        const float *s = m->scale + (size_t)r * nb;
        __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
        for (int b = 0; b < nb; b += 4) {   // 64 bytes = 128 weights = 4 blocks
            __m512i packed = _mm512_loadu_si512((const void *)(q + b * 16));
            __m512i lo = _mm512_and_si512(packed, lowmask);
            __m512i hi = _mm512_and_si512(_mm512_srli_epi16(packed, 4), lowmask);
            __m512i xlo = _mm512_loadu_si512((const void *)(x + b * QK));
            __m512i xhi = _mm512_loadu_si512((const void *)(x + b * QK + 64));
            __m512i dlo = _mm512_dpbusd_epi32(_mm512_setzero_si512(), lo, xlo);
            __m512i dhi = _mm512_dpbusd_epi32(_mm512_setzero_si512(), hi, xhi);
            __m512 sc = _mm512_castps128_ps512(_mm_mul_ps(_mm_loadu_ps(s + b), _mm_loadu_ps(xs + b)));
            acc0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(dlo), _mm512_permutexvar_ps(idx01, sc), acc0);
            acc1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(dhi), _mm512_permutexvar_ps(idx23, sc), acc1);
        }
        __m512 corr = _mm512_setzero_ps();
        for (int b = 0; b < nb; b += 16)
            corr = _mm512_fmadd_ps(_mm512_loadu_ps(s + b), _mm512_loadu_ps(xcorr + b), corr);
        y[r] = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1)) - _mm512_reduce_add_ps(corr);
    }
}

// scalar reference for correctness
static void matvec_ref(const q4mat *m, const int8_t *x, const float *xs, float *y) {
    const int nb = m->cols / QK;
    for (int r = 0; r < m->rows; r++) {
        const uint8_t *q = m->qs + (size_t)r * (m->cols / 2);
        double acc = 0;
        for (int c4 = 0; c4 < nb; c4 += 4) {           // 128-weight chunk
            for (int j = 0; j < 64; j++) {
                uint8_t byte = q[c4 * 16 + j];
                int wlo = (byte & 15) - 8, whi = (byte >> 4) - 8;
                int clo = c4 * QK + j, chi = c4 * QK + 64 + j;
                acc += (double)wlo * x[clo] * m->scale[r * nb + clo / QK] * xs[clo / QK];
                acc += (double)whi * x[chi] * m->scale[r * nb + chi / QK] * xs[chi / QK];
            }
        }
        y[r] = (float)acc;
    }
}

int main(int argc, char **argv) {
    // ~1 GiB of 4-bit weights: 16384 x 131072/2 bytes... use rows=32768, cols=65536 -> 1 GiB
    int rows = argc > 1 ? atoi(argv[1]) : 32768;
    int cols = argc > 2 ? atoi(argv[2]) : 65536;
    q4mat m = {rows, cols};
    size_t qbytes = (size_t)rows * cols / 2, nb = cols / QK;
    m.qs = aligned_alloc(64, qbytes);
    m.scale = aligned_alloc(64, (size_t)rows * nb * sizeof(float));
    int8_t *x = aligned_alloc(64, cols);
    float *xs = aligned_alloc(64, nb * sizeof(float));
    float *xcorr = aligned_alloc(64, nb * sizeof(float));
    float *y = aligned_alloc(64, rows * sizeof(float)), *yr = malloc(rows * sizeof(float));
    srand(1);
    for (size_t i = 0; i < qbytes; i++) m.qs[i] = rand();
    for (size_t i = 0; i < (size_t)rows * nb; i++) m.scale[i] = (rand() % 1000) / 1e5f;
    for (int i = 0; i < cols; i++) x[i] = (rand() % 255) - 127;
    for (size_t b = 0; b < nb; b++) {
        xs[b] = 0.01f;
        int s = 0;
        for (int k = 0; k < QK; k++) s += x[b * QK + k];
        xcorr[b] = 8.0f * s * xs[b];
    }
    double total = qbytes + (double)rows * nb * 4;  // bytes read per matvec
    printf("matrix %d x %d, %.2f GB read per matvec (incl. fp32 scales)\n", rows, cols, total / 1e9);

    matvec_q4_vnni(&m, x, xs, xcorr, y);  // warm
    // correctness on first 64 rows
    q4mat small = m; small.rows = 64;
    matvec_ref(&small, x, xs, yr);
    double maxerr = 0;
    for (int r = 0; r < 64; r++) { double e = y[r] - yr[r]; if (e < 0) e = -e; if (e > maxerr) maxerr = e; }
    printf("max abs err vs reference (64 rows): %.3e\n", maxerr);

    double best = 1e9;
    for (int it = 0; it < 5; it++) {
        double t0 = now();
        matvec_q4_vnni(&m, x, xs, xcorr, y);
        double dt = now() - t0;
        if (dt < best) best = dt;
    }
    printf("best %.1f ms  ->  %.1f GB/s effective (threads=%d)\n", best * 1e3, total / 1e9 / best,
           omp_get_max_threads());
    return 0;
}
