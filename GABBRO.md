# gabbro

Port of [Basalt](https://github.com/jesdga95/basalt) (MIT) to the NVIDIA DGX Spark (GB10: arm64, CUDA compute
capability 12.1, 121 GB unified memory). Gabbro has the same composition as basalt, cooled differently.

- Upstream: remote `upstream` (fetch only, push disabled). Branch `main` tracks upstream; `gb10-arm64` holds the port.
- Port strategy: the CUDA kernels build for `sm_121a` as they are; the x86-only CPU code (AVX2/AVX-512 expert and
  quant kernels) is replaced on aarch64 by the portable/NEON code of [Strata](https://github.com/Niko1221/Strata)
  (MIT), which Basalt forks, or by scalar fallbacks. On GB10 the MoE experts can stay resident in unified memory.
- Licences: MIT (Basalt, Strata); see LICENSE and THIRD_PARTY_LICENSES.

## Status (2026-11-24, port of upstream a7e22c1)

All five models build for `sm_121a` on aarch64 (CUDA 13.0, gcc 13). Tested on one DGX Spark, `--kv int8`,
all answering correctly; `basalt-bench --seed 1 -p 16384`, decode tok/s per fixture conv / struct / prose:

| Model, pack | Prefill 16K | Decode with drafter (accept %) | Drafter off | RAM |
|---|---:|---|---|---:|
| Inkling 2 Mini, NVFP4-MX8 (DSpark) | 3582 | 69 (55) / 130 (96) / 57 (38) | 44 / 42 / 44 | ~26 GB |
| Qwen3.8-27B, QUASAR-NVFP4 (DFlash 2) | 3178 | 34 (39) / 117 (85) / 25 (28) | 11 / 11 / 11 | ~28 GB |
| Qwen3.8-Flash-Next, UD-IQ1_S (MTP) | 2077 | 83 (67) / 152 (97) / 72 (63) | 42 / 44 / 42 | ~49 GB |

Community preset (`basalt-bench --community`, int8 KV, default drafter, cold), prefill 16K / 64K / decode tok/s:
Inkling 2 Mini NVFP4 3585 / 3013 / 107; Qwen3.8-27B NVFP4 3220 / 2442 / 69; Flash-Next UD-IQ1_S 1696 / 2213 / 125, i.e.
19-25 % of an RTX 5090 at decode (GB10 has ~15 % of its memory bandwidth). 4 parallel requests: Inkling 89, Flash-Next
96 tok/s aggregate. Rows and comparison: [docs/gb10-results](docs/gb10-results/community-SUMMARY.md). DeepSeek-V4.1 (smallest pack 260 GB) and
Hy4-flash (159 GB) do not fit one Spark.

What the port changes and its limits: [docs/gb10-arm64.md](docs/gb10-arm64.md). Test script: `tools/gb10/run_test.sh`.
On GB10 keep every MoE expert on the GPU (unified memory): CPU-placed experts run on slow portable code.

## Licences

Basalt and Strata are MIT, ggml/llama.cpp MIT, Infernix (kernels adapted by Basalt) Apache-2.0: all kept in
`LICENSE` and `THIRD_PARTY_LICENSES`. Files added or changed by the port say so in their header; `portable.cpp` is
adapted from Strata's `engine/src/kernels/cpu/portable.cpp` (MIT, (c) Niko1221 and the Strata contributors).
