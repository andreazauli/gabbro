# Basalt on DGX Spark (GB10, aarch64, sm_121) — port notes

Based on upstream [Basalt](https://github.com/jesdga95/basalt) a7e22c1 (2026-11-23, v0.5.6). All five models build
(qwen4exp, deepseek41, qwen35, hy4flash, inkling2) with CUDA 13.0 and gcc 13 for `sm_121a` on aarch64.

## Build

    git clone <this repository> gabbro && cd gabbro && git checkout gb10-arm64
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=121a
    cmake --build build -j 16          # binaries land in ./bin; about 8 minutes on a Spark (20 cores)
    python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

`bin/basalt serve` runs the Python server (`python -m serve.server`) from the checkout root, with `.venv/bin/python`
if present. Any environment with CUDA 13.0, CMake 3.24+ and gcc 13 works, on the host or in a container.

## What changed

1. **CMake**
   - `CMAKE_CUDA_ARCHITECTURES=121a` accepted (any other non-120 arch still FATAL). New `BASALT_CC_MINOR` (0 or 1) compile
     definition and `BASALT_ARCH_A` (120a / 121a); the 7 targets that hardcoded `CUDA_ARCHITECTURES 120a-real`
     (prefill_fp8, window_gemm, prefill_fp4, prefill_mxfp8, prefill_q8, narrow_bf16_parity, silu_parity) now use
     `${BASALT_ARCH_A}-real` — arch-specific `a` SASS does not run on another minor CC, so without this the FP4/FP8 prompt
     kernels would fail with "no kernel image" on GB10.
   - `BASALT_CPU_X86` from `CMAKE_SYSTEM_PROCESSOR` (same test as Strata's `STRATA_CPU_X86`). Off x86: all `-mavx*` /
     `-mf16c` / `-mfma` / `-mbmi2` ISA flags empty; x86 TUs not built (`expert.cpp`, `q2_avx2*.cpp`, `iq_avx2*.cpp`,
     `iq_avx512.cpp`, `kq_avx1/2.cpp`, `nvfp4_avx*.cpp`, the two `ggml_cpu_dots.c` objects); `portable.cpp` added instead.
   - `tools/vision/CMakeLists.txt`: `gemm_cpu_avx512.cpp` and the AVX flags only on x86.
2. **Runtime compute-capability checks** (`src/core/device.cu`, `src/program/gpu_select.cpp`, `src/models/deepseek41/model.cpp`)
   compare against `12.BASALT_CC_MINOR` instead of hardcoded 12.0.
3. **`include/basalt/platform/compiler.hpp`**: `immintrin.h`/`cpuid.h` only on x86 (`BASALT_X86`); on aarch64 `cpuid()` returns
   zeros and `xgetbv0()` 0, so every `cpu_avx*_ok()` probe reads false and the x86 kernels are never selected;
   `cpu_relax()` = `yield`; new `store_fence()` (x86 `sfence`, aarch64 `dmb sy`) and `prefetch_t1()`
   (`_mm_prefetch T1` / `__builtin_prefetch(p,0,2)`); `cpu_has_avx512_bf16()` false.
4. **Spin-wait / fence intrinsics** replaced by those helpers: `src/core/graph.cpp`, `src/kernels/cpu/pool.cpp`,
   `src/models/inkling2/cpu_experts.cpp`, `src/models/qwen4exp/decode/{session.cpp,verify/{batch_execution,diagnostics,execution,pipeline,cpu_warm}.cpp}`.
5. **`src/kernels/cpu/portable.cpp`** (new, ported from Strata `engine/src/kernels/cpu/portable.cpp`):
   - real implementations: `act_quant_q8_1` / `_avx2` (expert.cpp's scalar loop, bitwise identical to the AVX-512 one),
     `bf16_rows_dot_multi` / `_avx1` (Strata's NEON fp32-accumulate dot), `quantize_q8_K_avx2/512` (ggml's
     `quantize_row_q8_K_ref`), `cpu_features()` (all false);
   - stubs that `abort()` with a message: every Q2_0 / i-quant / k-quant / NVFP4 multi-token CPU expert row kernel;
     the `*_supported` / `*_fits` predicates return false.
6. **`ggml_cpu_dispatch.cpp`**: off x86 `ggml_get_type_traits_cpu()` returns nullptr (native_expert.cpp then refuses to plan
   CPU experts) and the MXFP8 CPU rows abort with a message.
7. **`qwen4exp/decode/verify/cpu_warm.cpp`**: scalar Q4×Q8_1 router-logit fallback for the CPU expert pre-warm.
8. **`src/vision/gemm_cpu.cpp`**: NEON 4×4 tile (two float32x4 accumulators per AVX lane), plus ARM `cpu_has_bf16()=false`.
9. **`src/platform/unified_memory.cpp`** (new, aarch64 only, linked into `basalt`): interposes `cudaMemGetInfo`. On GB10 the
   driver reports as free only unused pages, not reclaimable page cache; right after reading/downloading a 20-70 GB pack
   the "card" looked 1-5 GB free and every model refused to plan ("spill plan: ... card is 2884 MiB short"). On an
   integrated GPU (cudaDevAttrIntegrated, or device total ~= MemTotal) `free` is raised to `/proc/meminfo` MemAvailable,
   capped at total. `BASALT_UMA_FREE=0` disables it. No effect on discrete GPUs / x86 builds.
10. **`src/kernels/cpu/ggml/ggml_cpu_dots_portable.c`** (new, aarch64 only): the same `ggml_get_type_traits_cpu` table as
    the AVX2 `ggml_cpu_dots.c` (Q2_0, Q5_0/1, Q8_0, Q2_K..Q6_K, IQ1_S/M, IQ2_XXS/XS/S, IQ3_XXS/S, IQ4_NL/XS, NVFP4,
    MXFP8; activation types Q8_0/Q8_1/Q8_K with ggml's reference quantizers) implemented as dequantize (ggml_quants.c
    reference to_float) + fp32/fp64 dot, plus the MXFP8 multi-row entry points. Correct but slow scalar; needed because
    Flash-Next (qwen4exp) refuses to load a native pack without CPU dot products for every expert type, even when every
    expert ends up GPU-resident.
11. **`src/models/qwen4exp/load/qfn_run.cpp`**: the hard "this CPU does not support AVX2" refusal is a warning off x86.

## Test results (DGX Spark GB10, driver 580.178.04, CUDA 13.0; `--kv int8`; basalt-bench seeded)

Chat smoke test ("What is the capital of Italy? One word.", temperature 0): all three answered `Rome`.

| model / pack | load (weights on GPU) | pp16384 tok/s | decode conv / struct / prose tok/s, default drafter (acc %) | decode, drafter off |
| --- | --- | ---: | --- | --- |
| Inkling 2 Mini, Inkling-2-Mini-NVFP4-MX8 (19.6 GB), DSpark | 17.0 GiB, 19 s | 3582 (3618 no-draft) | 69 (55) / 130 (96) / 57 (38) | 44 / 42 / 44 |
| Qwen3.8-27B, Qwen3.8-27B-QUASAR-NVFP4 (21.1 GB), DFlash 2 | 18.4 GiB, 20 s | 3178 (3249) | 34 (39) / 117 (85) / 25 (28) | 11 / 11 / 11 |
| Qwen3.8-Flash-Next, Qwen3.8-FN-UD-IQ1_S (70.4 GB file), MTP | 35.6 GiB, 37 s, every expert resident | 2077 (2055) | 83 (67) / 152 (97) / 72 (63) | 42 / 44 / 42 |

Chat request (63/36 tokens): Inkling 79-83 tok/s, 27B 38.8 tok/s, Flash-Next 107 tok/s. System memory in use during a run:
~26 GB (Inkling), ~28 GB (27B), ~49 GB (Flash-Next IQ1_S). Test script: `tools/gb10/run_test.sh`; community-preset rows: [gb10-results](gb10-results/community-SUMMARY.md).
deepseek41 and hy4flash build but were not run: their smallest packs (260 GB and 159 GB) do not fit one Spark.

The default Flash-Next pack (NVFP4-DQ6_K, 131.7 GB) is larger than GB10 memory and was not tried: its spilled experts
would run on the slow portable CPU dots.

## Limitations

- **No fast CPU expert computation on aarch64.** Keep all MoE experts on the GPU. On GB10 (one 121 GB unified pool) this is
  the natural setup; packs larger than what the "card" can hold (e.g. Flash-Next NVFP4-DQ6_K, 131.7 GB) cannot run, since
  the spilled experts would need the x86 CPU kernels. Native-type experts spilled to RAM fall back to the portable
  scalar dots (correct, very slow); the canonical Q2_0 pack's AVX-512 kernels and the x86-only row kernels abort with a
  clear message (`basalt: ... is x86-only ...`). Porting the CPU expert rows would mean NEON versions of the Q2_0/IQ/K-quant/NVFP4/MXFP8
  row kernels (Strata uses ggml-cpu's NEON vec_dots for this; Basalt's extracted `ggml_cpu_dots.c` is AVX2-only) —
  estimated 2-4 days for correctness + parity tests.
- Tests (`-DBUILD_TESTING=ON`) were not ported: several test targets are x86-kernel-specific (avx_bench, iq_avx2_test, ...).
- PEER_ADA / multi-GPU peer paths not exercised (single GB10).
