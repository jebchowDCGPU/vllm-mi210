# fp8-anywhere

**FP8 inference on AMD Instinct MI210 (gfx90a/CDNA2) — hardware that doesn't have FP8 matrix cores.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

## What This Is

A complete deployment stack for serving **Qwen3.8-27B** (and similar models) with **FP8 quantization** on a single AMD Instinct MI210 — a GPU with **no native FP8 support**. Every FP8 operation runs through software emulation, delivering **28–49× the speed of stock vLLM** on this hardware.

| Metric | Stock vLLM | This Stack |
|---|---|---|
| FP8 decode (single-stream) | 1.4 tok/s | **45–69 tok/s** |
| KV cache capacity (262K ctx) | — | **601K tokens** (FP8 KV) |
| MTP acceptance | — | ~46% (n=5, probabilistic) |

## How It Works

The MI210 (gfx90a/CDNA2) has no FP8 matrix instructions — AMD introduced those in CDNA3 (MI300). This stack makes FP8 work anyway:

1. **Custom Triton W8A8/W8A16 GEMM kernel** — decodes FP8 weights to BF16 in-loop, uses the BF16 MFMA that gfx90a does have. Includes split-K, a tuned tile ladder, and a W8A16 path that skips activation quantization at decode (M≤16).
2. **FP8 KV cache** — stores K/V as e4m3 (halving KV memory), decodes in the attention kernel. Works via Triton `unified_attention` (HIP kernel fix documented in `docs/FP8_HIP_KERNEL_GUIDE.md`).
3. **MTP speculative decoding optimizations** — draft-vocab truncation (40K-row draft head instead of 248K), probabilistic draft sampling, sort-free top-k/top-p sampler.
4. **AITER attention** — repatched for gfx90a (davetha's `aiter-cdna2` binary repatcher + assertion fixes).

## Quick Start

```bash
# Prerequisites: MI210 (gfx90a), ROCm 7.2+, Docker, the base image
# (local/fp8-anywhere:mi210.6-aiter — built from davetha/mi210-vllm)

# 1. Place the model
#    Qwen/Qwen3.8-27B-FP8 → /home/tai/models/qwen38-27b-fp8/

# 2. Build the draft head (one-time, ~2 min)
bash mtp-opt/port-and-build.sh

# 3. Launch
bash start-fp8-aiter.sh
# Server on :8001 after ~5 min

# 4. Benchmark
cd gemm-harness && ./bench-e2e.sh
```

## Repository Structure

```
├── start-fp8-aiter.sh       # Launch script (recreates container, installs all patches)
├── kernel/                   # Custom Triton W8A8/W8A16 GEMM kernel + tests
├── patches/                  # All vLLM/AITER patches (drop-in replacements)
│   ├── fp8_utils_fork.py     #   bf16-cast fix for the stock block-FP8 kernel
│   ├── linear_init_fork.py   #   W8A8 kernel registration
│   ├── qwen3_5_mtp_patched.py#   Draft-vocab truncation for MTP
│   ├── rocm_aiter_fa_patched.py#  FP8 KV decode routing
│   ├── unified_attention_patched.py # AITER assertion fix
│   ├── envs_fork.py          #   Draft top-k/top-p + temp scale env vars
│   └── sampler-stage/        #   Sort-free sampler (7 files)
├── mtp-opt/                  # MTP optimization kit
│   ├── build-draft-head.py   #   Build the 40K-row draft head
│   ├── draft_vocab_ids.json  #   Pre-built vocab list (95% coverage)
│   └── sampler.patch         #   HyperQwen's sort-free sampler patch
├── gemm-harness/             # Test & benchmark harness
│   ├── verify.sh             #   Kernel gate (correctness + benchmark)
│   ├── bench-e2e.sh          #   E2E benchmark (warmup + 6 prompts + acceptance)
│   └── run-test.sh           #   One-command container test runner
```

## Performance Ladder

| Step | tok/s | What Changed |
|---|---|---|
| Stock vLLM 0.30 | 1.4 | Generic FP8 auto-emulation (11,997-instruction decoder) |
| + bf16-native cast | 7.8 | 5.6× — replace the generic decoder with the native cast |
| + tuned configs | 9.7 | Per-shape tile configs for MI210's 104 CUs |
| + MTP n=3 | 27–39 | Speculative decoding (62% acceptance) |
| + AITER FA + fork patches | 30–42 | Repatched AITER attention + davetha's stack |
| + W8A8 kernel (split-K) | 35–48 | Custom Triton kernel (1.24–1.61× vs stock per shape) |
| + MTP n=5 + draft-vocab + probabilistic | 44–65 | 40K draft head, sampled drafts, sort-free sampler |
| + W8A16 decode path | **45–69** | Skip activation quant at M≤16 (the other agent's +25%) |

## Key Innovations

- **FP8 on CDNA2**: software decode (e4m3→BF16, exact on all 254 finite codes) + BF16 MFMA. No native FP8 needed.
- **W8A16 decode path**: at M≤16 (decode), skip activation quantization entirely — the MI210 has no FP8 MFMA, so the quant round-trip was pure overhead.
- **Draft-vocab truncation**: the MTP drafter scores only 40,960 lm_head rows (419 MB) instead of 248,320 (2.54 GB) — a 6× cut in draft cost.
- **FP8 KV cache**: 2× KV capacity via e4m3 storage + in-loop decode in the attention kernel.

## Credits

- **davetha** — the base stack ([mi210-vllm](https://github.com/davetha/mi210-vllm), [mi210-llm-stack](https://github.com/davetha/mi210-llm-stack), [aiter-cdna2](https://github.com/davetha/aiter-cdna2))
- **wu1w** — the anti-fork approach ([fp8-anywhere](https://github.com/wu1w/fp8-anywhere))
- **HyperQwen** ([syv-ai/HyperQwen](https://github.com/syv-ai/HyperQwen)) — draft-vocab truncation, sort-free sampler, probabilistic draft sampling
- **rlrs** — AITER gfx90a attention PRs ([#4387](https://github.com/ROCm/aiter/pull/4387), [#4388](https://github.com/ROCm/aiter/pull/4388), [#4389](https://github.com/ROCm/aiter/pull/4389))

## License

Apache 2.0 (same as vLLM)
