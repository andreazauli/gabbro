// SPDX-License-Identifier: MIT
// The CPU row dot products of the ggml formats the native experts use, portable form, for a build without the
// AVX2 ggml_cpu_dots.c (aarch64: DGX Spark / GB10). Same table as ggml_cpu_dots.c - the same types, the same
// activation (vec_dot) types, the same reference activation quantizers - but every dot is the weight row
// dequantized by ggml_quants.c's reference to_float, against the activation row dequantized, in fp32. Correct and
// slow: it exists so a native pack loads and an expert the planner puts on the CPU still computes; on GB10 the
// experts are meant to stay on the GPU. The rounding is not ggml-cpu's integer dot's, so a CPU-computed expert
// differs from the x86 build in the last bits.
#define GGML_COMMON_DECL_C
#include "basalt/ggml/ggml_common.h"

#include "ggml_impl.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

void quantize_row_q8_0_ref(const float * GGML_RESTRICT x, block_q8_0 * GGML_RESTRICT y, int64_t k);
void quantize_row_q8_1_ref(const float * GGML_RESTRICT x, block_q8_1 * GGML_RESTRICT y, int64_t k);
void quantize_row_q8_K_ref(const float * GGML_RESTRICT x, block_q8_K * GGML_RESTRICT y, int64_t k);

static void from_q8_0(const float * x, void * y, int64_t k) { quantize_row_q8_0_ref(x, (block_q8_0 *) y, k); }
static void from_q8_1(const float * x, void * y, int64_t k) { quantize_row_q8_1_ref(x, (block_q8_1 *) y, k); }
static void from_q8_K(const float * x, void * y, int64_t k) { quantize_row_q8_K_ref(x, (block_q8_K *) y, k); }

// one growing scratch per thread: [0, n) the weight row, [n, 2n) the activation row
static _Thread_local float * g_buf = NULL;
static _Thread_local size_t g_cap = 0;
static float * scratch(size_t n) {
    if (g_cap < 2 * n) {
        free(g_buf);
        g_cap = 2 * n;
        g_buf = (float *) malloc(g_cap * sizeof(float));
        if (g_buf == NULL) GGML_ABORT("ggml_cpu_dots_portable: out of memory (%zu floats)", g_cap);
    }
    return g_buf;
}

static void act_to_float(enum ggml_type t, const void * vy, float * y, int n) {
    if (t == GGML_TYPE_Q8_0) {
        const block_q8_0 * b = (const block_q8_0 *) vy;
        for (int i = 0; i < n / QK8_0; ++i) {
            const float d = ggml_fp16_to_fp32(b[i].d);
            for (int j = 0; j < QK8_0; ++j) y[i * QK8_0 + j] = d * (float) b[i].qs[j];
        }
    } else if (t == GGML_TYPE_Q8_1) {
        const block_q8_1 * b = (const block_q8_1 *) vy;
        for (int i = 0; i < n / QK8_1; ++i) {
            const float d = ggml_fp16_to_fp32(b[i].d);
            for (int j = 0; j < QK8_1; ++j) y[i * QK8_1 + j] = d * (float) b[i].qs[j];
        }
    } else {   // Q8_K
        const block_q8_K * b = (const block_q8_K *) vy;
        for (int i = 0; i < n / QK_K; ++i)
            for (int j = 0; j < QK_K; ++j) y[i * QK_K + j] = b[i].d * (float) b[i].qs[j];
    }
}

static inline void dot_dequant(enum ggml_type wt, enum ggml_type at, int n, float * s, const void * vx, const void * vy) {
    float * w = scratch((size_t) n);
    float * a = w + n;
    ggml_get_type_traits(wt)->to_float(vx, w, n);
    act_to_float(at, vy, a, n);
    double acc = 0.0;
    for (int i = 0; i < n; ++i) acc += (double) w[i] * (double) a[i];
    *s = (float) acc;
}

#define DOT(name, WT, AT)                                                                                          \
    static void name(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx,          \
                     const void * GGML_RESTRICT vy, size_t by, int nrc) {                                          \
        GGML_UNUSED(bs); GGML_UNUSED(bx); GGML_UNUSED(by); GGML_UNUSED(nrc);                                       \
        dot_dequant(WT, AT, n, s, vx, vy);                                                                         \
    }
DOT(dot_q2_0, GGML_TYPE_Q2_0, GGML_TYPE_Q8_0)
DOT(dot_q5_0, GGML_TYPE_Q5_0, GGML_TYPE_Q8_0)
DOT(dot_q5_1, GGML_TYPE_Q5_1, GGML_TYPE_Q8_1)
DOT(dot_q8_0, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
DOT(dot_q2_K, GGML_TYPE_Q2_K, GGML_TYPE_Q8_K)
DOT(dot_q3_K, GGML_TYPE_Q3_K, GGML_TYPE_Q8_K)
DOT(dot_q4_K, GGML_TYPE_Q4_K, GGML_TYPE_Q8_K)
DOT(dot_q5_K, GGML_TYPE_Q5_K, GGML_TYPE_Q8_K)
DOT(dot_q6_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_K)
DOT(dot_iq2_xxs, GGML_TYPE_IQ2_XXS, GGML_TYPE_Q8_K)
DOT(dot_iq2_xs, GGML_TYPE_IQ2_XS, GGML_TYPE_Q8_K)
DOT(dot_iq3_xxs, GGML_TYPE_IQ3_XXS, GGML_TYPE_Q8_K)
DOT(dot_iq4_nl, GGML_TYPE_IQ4_NL, GGML_TYPE_Q8_0)
DOT(dot_iq3_s, GGML_TYPE_IQ3_S, GGML_TYPE_Q8_K)
DOT(dot_iq2_s, GGML_TYPE_IQ2_S, GGML_TYPE_Q8_K)
DOT(dot_iq4_xs, GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_K)
DOT(dot_iq1_s, GGML_TYPE_IQ1_S, GGML_TYPE_Q8_K)
DOT(dot_iq1_m, GGML_TYPE_IQ1_M, GGML_TYPE_Q8_K)
DOT(dot_nvfp4, GGML_TYPE_NVFP4, GGML_TYPE_Q8_0)
DOT(dot_mxfp8, GGML_TYPE_MXFP8, GGML_TYPE_Q8_0)
#undef DOT

static const struct ggml_type_traits_cpu type_traits_cpu[GGML_TYPE_COUNT] = {
    [GGML_TYPE_Q2_0]    = {NULL, dot_q2_0, GGML_TYPE_Q8_0, 1},
    [GGML_TYPE_Q5_0]    = {NULL, dot_q5_0, GGML_TYPE_Q8_0, 1},
    [GGML_TYPE_Q5_1]    = {NULL, dot_q5_1, GGML_TYPE_Q8_1, 1},
    [GGML_TYPE_Q8_0]    = {from_q8_0, dot_q8_0, GGML_TYPE_Q8_0, 1},
    [GGML_TYPE_Q8_1]    = {from_q8_1, NULL, GGML_TYPE_Q8_1, 1},
    [GGML_TYPE_Q2_K]    = {NULL, dot_q2_K, GGML_TYPE_Q8_K, 1},
    [GGML_TYPE_Q3_K]    = {NULL, dot_q3_K, GGML_TYPE_Q8_K, 1},
    [GGML_TYPE_Q4_K]    = {NULL, dot_q4_K, GGML_TYPE_Q8_K, 1},
    [GGML_TYPE_Q5_K]    = {NULL, dot_q5_K, GGML_TYPE_Q8_K, 1},
    [GGML_TYPE_Q6_K]    = {NULL, dot_q6_K, GGML_TYPE_Q8_K, 1},
    [GGML_TYPE_Q8_K]    = {from_q8_K, NULL, GGML_TYPE_Q8_K, 1},
    [GGML_TYPE_IQ2_XXS] = {NULL, dot_iq2_xxs, GGML_TYPE_Q8_K, 1},
    [GGML_TYPE_IQ2_XS]  = {NULL, dot_iq2_xs, GGML_TYPE_Q8_K, 1},
    [GGML_TYPE_IQ3_XXS] = {NULL, dot_iq3_xxs, GGML_TYPE_Q8_K, 1},
    [GGML_TYPE_IQ4_NL]  = {NULL, dot_iq4_nl, GGML_TYPE_Q8_0, 1},
    [GGML_TYPE_IQ3_S]   = {NULL, dot_iq3_s, GGML_TYPE_Q8_K, 1},
    [GGML_TYPE_IQ2_S]   = {NULL, dot_iq2_s, GGML_TYPE_Q8_K, 1},
    [GGML_TYPE_IQ4_XS]  = {NULL, dot_iq4_xs, GGML_TYPE_Q8_K, 1},
    [GGML_TYPE_IQ1_S]   = {NULL, dot_iq1_s, GGML_TYPE_Q8_K, 1},
    [GGML_TYPE_IQ1_M]   = {NULL, dot_iq1_m, GGML_TYPE_Q8_K, 1},
    [GGML_TYPE_NVFP4]   = {NULL, dot_nvfp4, GGML_TYPE_Q8_0, 1},
    [GGML_TYPE_MXFP8]   = {NULL, dot_mxfp8, GGML_TYPE_Q8_0, 1},
};

const struct ggml_type_traits_cpu * ggml_get_type_traits_cpu(enum ggml_type type) {
    if ((int) type < 0 || (int) type >= GGML_TYPE_COUNT) return NULL;
    const struct ggml_type_traits_cpu * t = &type_traits_cpu[type];
    return t->vec_dot == NULL && t->from_float == NULL ? NULL : t;
}

// MXFP8 rows (native_expert.cpp's multi-token path). The "prep" here is the activation dequantized in natural order
// (n floats, within the caller's n / 128 * GGML_MXFP8_PREP_BLOCK).
void ggml_mxfp8_q8_0_prep(int n, const void * GGML_RESTRICT vy, float * GGML_RESTRICT out) {
    act_to_float(GGML_TYPE_Q8_0, vy, out, n);
}

void ggml_mxfp8_q8_0_rows(int n, const void * GGML_RESTRICT vw, size_t stride, const float * const * prep, int nt,
                          float * const * out, int r0, int r1) {
    float * w = scratch((size_t) n);
    const ggml_to_float_t to_float = ggml_get_type_traits(GGML_TYPE_MXFP8)->to_float;
    for (int r = r0; r < r1; ++r) {
        to_float((const char *) vw + (size_t) r * stride, w, n);
        for (int t = 0; t < nt; ++t) {
            double acc = 0.0;
            for (int i = 0; i < n; ++i) acc += (double) w[i] * (double) prep[t][i];
            out[t][r] = (float) acc;
        }
    }
}
