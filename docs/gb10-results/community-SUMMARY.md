# gabbro (Basalt a7e22c1 + GB10/arm64 port) — community-1 rows on DGX Spark (GB10)

Runs: `basalt-bench --community --model PACK --submitter gabbro --community-dir <here>`
(one DGX Spark, the whole GPU, 16 CPUs), 2026-11-24. Preset community-1: int8 KV, default drafter, each number cold.
Row files: `runs/<model>/<id>.json`.

GB10 caveats written into the rows: NVML reports no framebuffer and no power limit on GB10, so `src/program/bench_main.cpp` (this port) records
`vram_gib` = MemTotal (122) and `power_w` = 140 (GB10's nominal SoC budget, CPU+GPU, not an NVML limit;
override with BASALT_GPU_POWER_W). `pcie` "1.0 x1" is what NVML reports for the C2C-attached GPU (h2d 59.3 GB/s measured).
`cpu` is missing ("unknown": the CPU name is read via CPUID). RAM read 85 GB/s.

| model / quant | prefill 16K | prefill 64K | decode_tps | conv / struct / prose tok/s | accept % conv/struct/prose | peak RAM used |
| --- | ---: | ---: | ---: | --- | --- | ---: |
| Inkling 2 Mini NVFP4 (MX8 pack) | 3584.5 | 3012.9 | 106.9 | 66.7 / 130.5 / 56.5 | 53.5 / 96.6 / 37.7 | 27.4 GiB |
| Qwen3.8-27B QUASAR NVFP4 | 3220.2 | 2442.3 | 68.5 | 36.9 / 116.3 / 25.5 | 41.6 / 84.4 / 30.2 | 30.4 GiB |
| Qwen3.8-Flash-Next UD-IQ1_S | 1696.0 | 2212.6 | 125.1 | 90.8 / 153.0 / 72.6 | 69.3 / 96.7 / 62.6 | 55.1 GiB |

Concurrency (`basalt serve --kv int8 --parallel 4`, 4 slots x 2048 ctx; 256-token story replies, thinking off, temp 0.7):

| model | 1 request | 4 simultaneous: per request | 4 simultaneous: aggregate |
| --- | ---: | ---: | ---: |
| Inkling 2 Mini | 58.4 tok/s (56.3 wall) | 23.8-25.5 tok/s | 89.2 tok/s (1024 tok / 11.5 s) |
| Flash-Next IQ1_S | 64.2 tok/s (61.4 wall) | 24.5-28.6 tok/s | 95.5 tok/s (1024 tok / 10.7 s) |

## Against the best published single-GPU rows (docs/community/runs at a7e22c1)

| model / quant | machine | prefill 16K | prefill 64K | decode_tps | GB10 as % (pp16 / pp64 / decode) |
| --- | --- | ---: | ---: | ---: | --- |
| Inkling NVFP4 | GB10 (this) | 3584.5 | 3012.9 | 106.9 | |
| | RTX 5090 @400 W, kv fp16 (48f0bea00e49) | 16417.4 | 12981.3 | 555.1 | 22 / 23 / 19 |
| | RTX 5090 @400 W, kv int8 (a71c4a807263) | 16624.6 | 12809.7 | 550.5 | 22 / 24 / 19 |
| | RTX PRO 6000 WS @550 W, fp16 (bb0ef8128691) | 17175.4 | 14105.6 | 423.3 | 21 / 21 / 25 |
| | RTX PRO 4500 @200 W, fp16 (28766f22c5e6) | 9311.5 | 7519.1 | 304.6 | 38 / 40 / 35 |
| Qwen3.8-27B NVFP4 | GB10 (this) | 3220.2 | 2442.3 | 68.5 | |
| | RTX 5090 @800 W (e34cf252a72d) | 18709.7 | 13322.1 | 575.0 | 17 / 18 / 12 |
| | RTX 5090 @400 W (bae46cf53542) | 12715.2 | 8916.7 | 481.5 | 25 / 27 / 14 |
| | RTX PRO 6000 | — no row — | | | |
| | RTX PRO 4500 @200 W (9855077e4bac) | 6559.2 | 4770.4 | 239.9 | 49 / 51 / 29 |
| Flash-Next IQ1_S | GB10 (this) | 1696.0 | 2212.6 | 125.1 | |
| | no RTX 5090 / PRO 6000 / PRO 4500 IQ1_S row; same quant elsewhere: | | | | |
| | RTX PRO 4000 @145 W (54fa4d69c839) | 3498.4 | 3536.0 | 257.1 | 48 / 63 / 49 |
| | RTX 5070 Ti @300 W (88bb8f0a8249) | 3303.9 | 3892.5 | 230.7 | 51 / 57 / 54 |
| | nearest bigger cards, other quants: RTX 5090 @500 W IQ3_XXS (1d28ef5f8e80) | 8784.2 | 9326.3 | 564.5 | 19 / 24 / 22 |
| | RTX PRO 6000 WS @600 W NVFP4-DQ8 (631f514c709d) | 10070.0 | 12033.9 | 538.7 | 17 / 18 / 23 |
| | RTX PRO 4500 @200 W NVFP4 (6546e1df6233) | 5530.6 | 5931.5 | 306.3 | 31 / 37 / 41 |

Reading: decode is memory-bandwidth bound; GB10's LPDDR5x (273 GB/s) vs RTX 5090 GDDR7 (1792 GB/s) is 15 %, and
decode lands at 12-25 % of a 5090. Prefill (tensor-core bound) lands at 17-27 % of a 5090 (GB10 ~ 1/5 of its SMs/clock).

## Not run
- DeepSeek-V4.1-Flash-1017: smallest Basalt pack UD-IQ1_M = 260.0 GB (IQ2_XXS 285.5, NVFP4 428.4) > 121 GiB unified
  memory: most experts would stream from the NFS-backed SSD path and compute on the scalar portable CPU dots. Not downloaded.
- Hy4-flash: only pack Hy4-flash-NVFP4 = 158.7 GB (routed experts 83.6 GB NVFP4 + 63.4 GB E4M3 engram table read by row
  from the file + ~7 GB dense + 1.7 GB drafter). Over the ~100 GB bound; the experts alone (83.6 GB) plus dense/KV could
  fit the GPU side, but Hy4's tier keeps the non-card experts in page-locked RAM computed on the CPU (portable scalar
  NVFP4 dot here) and on GB10 both draw from the same 121 GiB. Not downloaded.
