// src/kernels/cpu/portable.cpp - the CPU kernel symbols on a non-x86 build (aarch64: DGX Spark / GB10, Grace).
//
// Replaces expert.cpp, q2_avx2*.cpp, iq_avx2*.cpp, iq_avx512.cpp, kq_avx2.cpp, kq_avx1.cpp and nvfp4_avx*.cpp, which
// are AVX-only. Ported from the same file in Strata (engine/src/kernels/cpu/portable.cpp), the project basalt forked:
//
//   * the activation quantizer is expert.cpp's scalar loop (bitwise the same result as its AVX-512 form);
//   * the BF16 router dot is NEON (fp32 accumulation, eight columns a step);
//   * the Q8_K quantizer is ggml's reference quantizer;
//   * every multi-token expert row kernel (Q2_0, i-quant, k-quant, NVFP4) reports itself unsupported and aborts with
//     a message if reached. cpu_avx2_ok() and friends read false on this CPU and ggml_cpu_dispatch.cpp reports no
//     ggml-cpu dot products, so native_expert.cpp refuses to plan experts in RAM: on GB10 the 128 GB of unified memory
//     keeps every expert on the GPU ("engine": {"resident-experts": true}).
#include "basalt/kernels/cpu/expert.hpp"
#include "basalt/kernels/cpu/iq_avx2.hpp"
#include "basalt/kernels/cpu/iq_avx512.hpp"
#include "basalt/kernels/cpu/kq_avx1.hpp"
#include "basalt/kernels/cpu/kq_avx2.hpp"
#include "basalt/kernels/cpu/nvfp4_avx2.hpp"
#include "basalt/kernels/cpu/nvfp4_avx512.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

extern "C" void quantize_row_q8_K_ref(const float* x, void* y, int64_t k);

namespace basalt::kernels::cpu {
namespace {

[[noreturn]] void unsupported(const char* what) {
    std::fprintf(stderr,
                 "basalt: %s is x86-only (AVX2/AVX-512) and this build (%s) has no portable version.\n"
                 "        Keep the experts on the GPU (\"engine\": {\"resident-experts\": true}).\n",
                 what,
#if defined(__aarch64__)
                 "aarch64"
#else
                 "non-x86"
#endif
    );
    std::abort();
}

inline float bf16f(uint16_t h) {
    const uint32_t b = (uint32_t) h << 16;
    float f;
    std::memcpy(&f, &b, 4);
    return f;
}

}  // namespace

// ---- expert.hpp ----

const char* CpuFeatures::reason() const {
    return usable() ? "ok" : "not an x86 CPU (the Q2_0 expert kernel is AVX-512 only)";
}

CpuFeatures cpu_features() { return CpuFeatures{}; }

void act_quant_q8_1(const float* x, int n, ActQ& a) {
    // expert.cpp's scalar loop: max |x|, one multiply, +-0.5 away from zero, truncation, clamp to +-127.
    a.nchunks = n / QKA;
    for (int k = 0; k < a.nchunks; ++k) {
        const float* xb = x + k * QKA;
        float amax = 0.f;
        for (int j = 0; j < QKA; ++j) amax = std::fmax(amax, std::fabs(xb[j]));
        const float s = amax > 0.f ? amax / 127.f : 0.f;
        const float inv = s > 0.f ? 1.f / s : 0.f;
        int32_t sum = 0;
        int8_t* q = a.q + k * QKA;
        for (int j = 0; j < QKA; ++j) {
            const float t = xb[j] * inv;
            const float r = t + (t >= 0.f ? 0.5f : -0.5f);
            int v = (int) r;
            v = v < -127 ? -127 : (v > 127 ? 127 : v);
            q[j] = (int8_t) v;
            sum += v;
        }
        a.scale[k] = s;
        a.sum[k] = sum;
        a.hx[k] = s * (float) sum;
    }
}

void act_quant_q8_1_avx2(const float* x, int n, ActQ& a) { act_quant_q8_1(x, n, a); }

void q2_0_gguf_rows_multi(const uint8_t*, size_t, int, const ActQ* const*, int, float* const*, int, int) {
    unsupported("the Q2_0 row kernel");
}
void q2_0_gguf_rows_multi_avx2(const uint8_t*, size_t, int, const ActQ* const*, int, float* const*, int, int) {
    unsupported("the Q2_0 row kernel");
}
void q2_0_gguf_rows_multi_avx2_base(const uint8_t*, size_t, int, const ActQ* const*, int, float* const*, int, int) {
    unsupported("the Q2_0 row kernel");
}
void q2_0_gguf_rows_multi_avx2_vnni(const uint8_t*, size_t, int, const ActQ* const*, int, float* const*, int, int) {
    unsupported("the Q2_0 row kernel");
}

// ---- iq_avx2.hpp / iq_avx512.hpp ----

void quantize_q8_K_avx2(const float* x, void* dst, int64_t k) { quantize_row_q8_K_ref(x, dst, k); }
void quantize_q8_K_avx512(const float* x, void* dst, int64_t k) { quantize_row_q8_K_ref(x, dst, k); }

bool iq256_supported(int) noexcept { return false; }
bool iq4nl256_fits(int) noexcept { return false; }
bool iq512_supported(int) noexcept { return false; }
bool iq1s_512_fits(int) noexcept { return false; }
bool iq_exact_512_supported(int) noexcept { return false; }

void iq256_gu_rows(int, const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int) {
    unsupported("the AVX2 i-quant kernel");
}
void iq256_rows(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    unsupported("the AVX2 i-quant kernel");
}
void iq256_gu_rows_base(int, const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int) {
    unsupported("the AVX2 i-quant kernel");
}
void iq256_gu_rows_vnni(int, const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int) {
    unsupported("the AVX2 i-quant kernel");
}
void iq256_rows_base(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    unsupported("the AVX2 i-quant kernel");
}
void iq256_rows_vnni(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    unsupported("the AVX2 i-quant kernel");
}
void iq256_gu_rows_mode_base(int, int, const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int,
                             int) {
    unsupported("the AVX2 i-quant kernel");
}
void iq256_gu_rows_mode_vnni(int, int, const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int,
                             int) {
    unsupported("the AVX2 i-quant kernel");
}
void iq256_rows_mode_base(int, int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    unsupported("the AVX2 i-quant kernel");
}
void iq256_rows_mode_vnni(int, int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    unsupported("the AVX2 i-quant kernel");
}
void iq4nl256_down_rows(const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    unsupported("the AVX2 IQ4_NL kernel");
}
void iq4nl256_down_rows_base(const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    unsupported("the AVX2 IQ4_NL kernel");
}
void iq4nl256_down_rows_vnni(const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    unsupported("the AVX2 IQ4_NL kernel");
}
void iq512_gu_rows(int, const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int) {
    unsupported("the AVX-512 i-quant kernel");
}
void iq512_rows(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    unsupported("the AVX-512 i-quant kernel");
}
void iq1s_512_gu_rows(const uint8_t*, size_t, size_t, int, float, float, bool, float, const void* const*, int,
                      float* const*, int, int) {
    unsupported("the AVX-512 IQ1_S kernel");
}
void iq1s_512_rows(const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    unsupported("the AVX-512 IQ1_S kernel");
}
void iq_exact_512_gu_rows(int, const uint8_t*, size_t, size_t, int, float, float, bool, float, const void* const*, int,
                          float* const*, int, int) {
    unsupported("the AVX-512 exact i-quant kernel");
}
void iq_exact_512_rows(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    unsupported("the AVX-512 exact i-quant kernel");
}

// ---- kq_avx2.hpp / kq_avx1.hpp ----

bool kq256_supported(int) noexcept { return false; }
void kq256_gu_rows(int, const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int) {
    unsupported("the AVX2 k-quant kernel");
}
void kq256_rows(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    unsupported("the AVX2 k-quant kernel");
}

/// out[t * rows + r] = dot(w row r as bf16, x token t), fp32 accumulation (Strata's NEON form).
void bf16_rows_dot_multi(const uint16_t* w, int rows, int cols, const float* x, int nt, float* out) {
    for (int r = 0; r < rows; ++r) {
        const uint16_t* wr = w + (size_t) r * (size_t) cols;
        for (int t = 0; t < nt; ++t) {
            const float* xt = x + (size_t) t * cols;
            int c = 0;
            float s = 0.f;
#if defined(__ARM_NEON)
            float32x4_t acc0 = vdupq_n_f32(0.f), acc1 = vdupq_n_f32(0.f);
            for (; c + 8 <= cols; c += 8) {
                const uint16x8_t h = vld1q_u16(wr + c);
                const float32x4_t w0 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(h), 16));
                const float32x4_t w1 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(h), 16));
                acc0 = vfmaq_f32(acc0, w0, vld1q_f32(xt + c));
                acc1 = vfmaq_f32(acc1, w1, vld1q_f32(xt + c + 4));
            }
            s = vaddvq_f32(vaddq_f32(acc0, acc1));
#endif
            for (; c < cols; ++c) s += bf16f(wr[c]) * xt[c];
            out[(size_t) t * rows + r] = s;
        }
    }
}
void bf16_rows_dot_multi_avx1(const uint16_t* w, int rows, int cols, const float* x, int nt, float* out) {
    bf16_rows_dot_multi(w, rows, cols, x, nt, out);
}

// ---- nvfp4_avx2.hpp / nvfp4_avx512.hpp ----

bool nvfp4_256_fits(int, int) noexcept { return false; }
bool nvfp4_512_fits(int, int) noexcept { return false; }
void nvfp4_256_gu_rows(const uint8_t*, size_t, size_t, int, float, float, const SwiGlu&, const void* const*, int,
                       float* const*, int, int) {
    unsupported("the AVX2 NVFP4 kernel");
}
void nvfp4_256_down_rows(const uint8_t*, size_t, int, float, const void* const*, int, float* const*, int, int) {
    unsupported("the AVX2 NVFP4 kernel");
}
void nvfp4_256_gu_rows_base(const uint8_t*, size_t, size_t, int, float, float, const SwiGlu&, const void* const*, int,
                            float* const*, int, int) {
    unsupported("the AVX2 NVFP4 kernel");
}
void nvfp4_256_gu_rows_vnni(const uint8_t*, size_t, size_t, int, float, float, const SwiGlu&, const void* const*, int,
                            float* const*, int, int) {
    unsupported("the AVX2 NVFP4 kernel");
}
void nvfp4_256_down_rows_base(const uint8_t*, size_t, int, float, const void* const*, int, float* const*, int, int) {
    unsupported("the AVX2 NVFP4 kernel");
}
void nvfp4_256_down_rows_vnni(const uint8_t*, size_t, int, float, const void* const*, int, float* const*, int, int) {
    unsupported("the AVX2 NVFP4 kernel");
}
void nvfp4_512_gu_rows(const uint8_t*, size_t, size_t, int, float, float, const SwiGlu&, const void* const*, int,
                       float* const*, int, int) {
    unsupported("the AVX-512 NVFP4 kernel");
}
void nvfp4_512_down_rows(const uint8_t*, size_t, int, float, const void* const*, int, float* const*, int, int) {
    unsupported("the AVX-512 NVFP4 kernel");
}

}  // namespace basalt::kernels::cpu
