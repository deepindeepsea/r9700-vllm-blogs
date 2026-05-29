# Raw benchmark data

The data backing the three-part blog series in this repo.

| File | What it is |
|---|---|
| `bench_2026-05-28.md` | Dense Qwen2.5-14B-AWQ on vLLM — single-stream baseline ("Part 1" data) |
| `bench_2026-05-28.json` | Same as above, machine-readable |
| `bench_moe_2026-05-28.md` | Qwen3-30B-A3B-AWQ (MoE) on vLLM — concurrency sweep N=1..16 ("Part 2" data) + MoE tuning negative-result addendum |
| `bench_moe_2026-05-28.json` | Same as above, machine-readable |

All runs: single Radeon AI PRO R9700 (32 GB, gfx1201/RDNA4), `rocm/vllm-dev:rocm7.2.1_navi_ubuntu24.04_py3.12_pytorch_2.9_vllm_0.16.0`, `--gpu-memory-utilization 0.92`, `--max-model-len 32768`, distinct prompts per request (no prefix-cache hit), 512 output tokens, temperature 0.

The llama.cpp Vulkan number (108 tok/s tg128) referenced in Part 3 is reproducible via the `docker run` snippet in `../part3-the-vulkan-plot-twist.md` — no separate JSON for that one.

Internal hostnames, IPs, usernames, and HF tokens have been redacted from the markdown writeups.
