# Scout Report — Qwen3.8-27B (FP8 + W4A16) on AMD Instinct MI210

**Date:** 2026-09-26 · **Phase:** Scout complete (MI210 system still powered off — awaiting power-on + SSH for deployment) · **Status:** final

**Mission:** Bring up the user's MI210 system to serve **Qwen/Qwen3.8-27B** with **FP8 W8A8** (the official block-FP8 checkpoint — user explicitly wants this served on gfx90a via emulated FP8 GEMM; a development task, not a deployment-only task) and **W4A16** (Quark-Qronos INT4). **Serving framework: vLLM primary, SGLang explicitly acceptable** (user already runs SGLang on ROCm MI35x).

---

## 1. Target model — Qwen3.8-27B (confirmed facts)

Source: [model card](https://huggingface.co/Qwen/Qwen3.8-27B), [config.json](https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/config.json), [vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B).

| Property | Value |
|---|---|
| Released | 2026-08-05 (FP8 variant 2026-08-13) |
| Size | Dense **27B**, Apache-2.0 |
| Architecture | `Qwen3_5ForConditionalGeneration`, model_type `qwen3_5` (same family as Qwen3.5/3.6) |
| Attention | **Hybrid**: 64 layers = 48 × Gated DeltaNet (linear attn) + 16 × full attention (`full_attention_interval: 4`) |
| Full-attn details | GQA 24 Q heads / 4 KV heads, head_dim 256, partial RoPE (0.25), interleaved mrope |
| GDN details | 48 V heads / 16 QK heads, head_dim 128, conv kernel 4, `mamba_ssm_dtype: float32` |
| Hidden / FFN / vocab | 5120 / 17408 / 248,320 (padded) |
| Vision | Native VLM: 27-layer ViT (hidden 1152, patch 16, spatial-merge 2, deepstack mergers) — image + video |
| MTP | Built-in multi-token-prediction draft head (1 extra layer) for speculative decoding |
| Context | 262,144 native → 1M via YaRN (`rope_parameters` override) |
| Chat behavior | Thinking mode default, `reasoning_effort` (xhigh/medium/low), `preserve_thinking` |

**Official quantized checkpoints:**
- [`Qwen/Qwen3.8-27B-FP8`](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) — e4m3, **dynamic activation scheme, 128×128 block-wise weight scales** (DeepSeek-style block-FP8). Vision tower, GDN internals (`in_proj_a/b/ba`, `conv1d`, `A_log`, `dt_bias`), embeddings, `lm_head`, norms, and the MTP head are all kept unquantized.
- NVFP4: [`nvidia/Qwen3.8-27B-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4), [`unsloth/Qwen3.8-27B-NVFP4`](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) (NVIDIA-only kernels)
- GGUF: unsloth / bartowski / ggml-org (+ MTP variants)
- **No official GPTQ-Int4/AWQ of Qwen3.8-27B exists** (Qwen shipped GPTQ-Int4 for Qwen3.5-27B but not for 3.8-27B) → the W4A16 answer is AMD's own Quark-Qronos INT4 checkpoint (§5), already cached locally.

**Memory math on MI210 (64 GB HBM2e):**
- BF16: ~54 GB weights → does not fit practically
- FP8: ~28 GB weights (+ bf16 vision/GDN/MTP parts) → fits, room for KV cache
- W4A16: ~15–16 GB → fits easily
- KV: only 16 full-attn layers cache KV ≈ 64 KB/token (bf16); GDN layers hold constant-size recurrent state per sequence

## 2. Official framework support (vLLM recipe, 2026-09-14)

Source: [recipes.vllm.ai/Qwen/Qwen3.8-27B](https://recipes.vllm.ai/Qwen/Qwen3.8-27B).

- Requires **vLLM ≥ 0.17.0** (recipe verified on 0.26.1rc1; DFlash2 drafting needs ≥ 0.28.0) and **transformers ≥ 5.8.0**.
- Official FP8 checkpoint: on Blackwell, vLLM auto-disables DeepGEMM for `qwen3_5_text` and falls back to CUTLASS — loads unaided. (DeepGEMM is the default block-FP8 GEMM path elsewhere — CUDA/MI300 oriented; the MI210 question is exactly here.)
- MTP speculative decoding works in every precision (`{"method":"mtp","num_speculative_tokens":3}`; spelled `qwen3_5_mtp` on Ascend).
- `--language-model-only` flag exists to skip the vision tower (text-only serving) — a memory lever for MI210.
- `--kv-cache-dtype fp8` supported.
- Precedent for non-CUDA backends: vLLM-Ascend serves the official block-FP8 checkpoint by expanding `weight_scale_inv` tiles at load and re-quantizing to MXFP8 ([vllm-ascend#14852](https://github.com/vllm-project/vllm-ascend/pull/14852), merged 2026-08-26).
- NVFP4/MXFP4 paths are NVIDIA-specific (MXFP4 currently fails to load on NVIDIA; NVFP4 needs cutlass/FlashInfer kernels).

## 3. MI210 hardware & software support reality (gfx90a / CDNA2)

- Single AMD Instinct MI210: Aldebaran, gfx90a, CDNA2, 64 GB HBM2e, ~1.6 TB/s, MFMA FP16/BF16/INT8 — **no native FP8 matrix support** (FP8 MFMA arrived with CDNA3/MI300); `torch._scaled_mm` FP8 paths are MI300+.
- **ROCm 10.0.0 still fully supports MI210/MI250/MI250X** (compatibility matrix; new MI210 KVM configs even added; TheRock since 7.14). No "last ROCm for MI200" exists yet.
- **vLLM still lists MI200s (gfx90a) as supported** (ROCm 6.3+); no deprecation through v0.30. v0.28.0 enabled Qwen3.8 on ROCm ([#50068](https://github.com/vllm-project/vllm/pull/50068)). Wheels: Python 3.12, `rocm723` variant (0.28.0/0.29.0/0.30.0); official Docker `vllm/vllm-openai-rocm:v0.30.0` / `nightly` / `nightly-rocm100`. AMD's `rocm/vllm` images deprecated.
- **AMD's validated AI-ecosystem stack covers only gfx942/gfx950+** — MI200 is excluded from AMD's *validated* vLLM/PyTorch stack even though ROCm itself supports it. The community carries MI210 (see §6).
- Watch item: vLLM MRV1 deprecated (removal in v0.32) and still used by a few ROCm models.

## 4. FP8 W8A8 on MI210 — the core development task, now precisely scoped

**Hardware reality:** gfx90a/CDNA2 has no FP8 MFMA; `torch._scaled_mm` is MI300+. Stock vLLM crashes on **per-tensor/dynamic FP8** (compressed-tensors) via an ironic accident: `_check_scheme_supported` compares a CUDA cc threshold (89) against ROCm's reported capability (gfx90a → (9,0) = 90), which *passes* → routes W8A8 into `torch._scaled_mm` → crash. (davetha's fix: gate the W8A8 branch on `get_cdna_version() <= 2 and not on_rdna4()`, commit `d74ce3db1d`.) **The block-FP8 path — the official checkpoint — does not hit this crash** (see routing reality below).

**What exists — davetha fork PR #7, `TritonW8A16Fp8LinearKernel` (750 lines, read in full):**
- Loads BF16 activations + uint8 (e4m3) weights; upcasts weights to BF16 **in the inner loop** via `b_u8.to(tl.float8e4nv, bitcast=True).to(tl.bfloat16)` (~4.5 VALU ops/value, exact on all 254 codes incl. denormals); BF16 `tl.dot` with FP32 accumulator. An fp16 bit-trick variant (`decode="fp16"`) loses 26/27 rows — gfx90a's f16 MFMA flushes subnormals (probed on hardware).
- **Scales: per-channel [N] FP32 only, applied once in the epilogue.** `can_implement` explicitly rejects 128×128 block quant, fnuz, and e5m2.
- Registered in `_POSSIBLE_WFP8A16_KERNELS[ROCM]` (upstream ships the list empty); davetha's upstream PR [#52985](https://github.com/vllm-project/vllm/pull/52985) open since 2026-08-19, unmerged.
- Measured: **45.9 tok/s e2e** on Qwen3-8B-FP8-dynamic (compressed-tensors per-channel — the proven-working shape); on the 27B, kernel-level: ties W4A16 at M=1/8, **~611 GB/s weight-stream at M=1 (~37% of the 1.6 TB/s peak — conversion-throughput-bound, not bandwidth-bound)**. davetha's own framing: "a capacity-and-compatibility feature, not a speed feature."

**Routing reality (corrected by Triton-source findings — better than first scoped):**
- In **stock vLLM**, the official checkpoint's Fp8Config block-quant path routes through `init_fp8_linear_kernel` → `_POSSIBLE_FP8_BLOCK_KERNELS[ROCM]` = [AITER (rejects gfx90a), **TritonFp8BlockScaledMMKernel (accepts all ROCm)**] → `w8a8_triton_block_scaled_mm` — `tl.dot(fp8, fp8)` with per-token-group × 128×128-block scales in the FP32 accumulator, exactly the checkpoint semantics.
- **Triton ≥3.5 auto-emulates FP8 dots on gfx90a**: `tl.dot` with e4m3fn operands on MFMA version ≤3 is automatically promoted to FP16 MFMA with software conversions ([triton#7186](https://github.com/triton-lang/triton/pull/7186), landed 2025-06-14; compiler remark: "emulated with fp16 so low performance"). Triton ≤3.4 blocked FP8 dtypes on gfx90a — that was the old error (vLLM [#16394](https://github.com/vllm-project/vllm/issues/16394)). **The stock block path plausibly loads and runs on gfx90a unmodified — untested publicly (vLLM CI skips FP8 model tests on gfx90a), not proven broken.**
- Safety detail: e4m3's min magnitude 2⁻⁹ is an FP16 normal → gfx90a's fp16-MFMA subnormal flush cannot bite (e5m2 would be at risk; the checkpoint is e4m3).
- In **davetha's fork**, fp8 checkpoints route to his W8A16 kernel (per-channel only, rejects 128×128 blocks) — a fork-specific choice, not a hardware limit.
- Activation quant: `per_token_group_fp8_quant` has a HIP software fallback on gfx90a (the fast path is gfx942/1200/1201/950/1250-only) — works, slower.

**The W8A8 route (ranked, per the reframed research):**
0. **Try stock first**: vLLM 0.28+/0.30 ROCm (Triton ≥3.5) + the official checkpoint — verify `Selected TritonFp8BlockScaledMMKernel` in the logs + the "emulated with fp16" Triton remark, check numerics vs BF16, benchmark. This may just work, slowly.
1. **Tune the Triton block-kernel configs for MI210** (104 CUs; the default 64×128×128 / 2-stage ≈ 32 KB LDS fits) — the auto-emulation's generic software fp8→fp16 conversion is the slow part.
2. **Replace the generic conversion with the bit-trick** (davetha's 3-instruction `h = ((u & 0x80) << 8) | ((u & 0x7f) << 7)` — bit-exact 254/254 codes; or the user's FP8-KV variant) inside the block kernel.
3. **Fallback**: extend davetha's PR #52985 W8A16 kernel to W8A8 (add per-token-group quant + in-loop block scales; the decode machinery is documented to the bit level).
- CK route: dead end for this (no F8×F8 emulated instance — FP8 instances hang LLVM on CDNA2; no block-scaled FP8 for gfx90a).
- Bonus: PR [#54341](https://github.com/vllm-project/vllm/pull/54341) measured **fp8_e4m3 KV cache bit-correct on MI250/gfx90a** (fp8_e5m2 KV silently mis-stored, being removed upstream) — FP8 KV is available if wanted.
- **The conversion trick is known and measured**: stock vLLM's block-FP8 Triton kernel on gfx90a compiles the generic fp8→f32 lowering into an **11,997-instruction software decoder** (7,106 of them `v_cmp`/`v_cndmask`) — that is the 2.7 tok/s catastrophe. davetha's **3-instruction bit-trick** (`h = ((u & 0x80) << 8) | ((u & 0x7f) << 7)`, fold 2^16 into the accumulator) is bit-exact on all 254 non-NaN byte patterns and took Qwen3-14B-FP8 from 2.7 → 29.2 tok/s (10.8×); tuned, FP8-emulated GEMMs run at **0.67–0.85× of bf16 time with 1.75× less weight memory**. Same family as the user's FP8-KV bit-trick.
- **Hard avoid**: CK's FP8 blockscale path is **numerically wrong on gfx90a** (its `amd_xdlops.hpp` FP8 fallback computes exactly ¼ of the sum) — Triton only.
- Performance target math unchanged: ~28 GB FP8 weight reads → ~57 tok/s ceiling → ~100 tok/s with MTP (acceptance ≥ ~0.8). The conversion-optimization work decides whether the ceiling is approached.
- **Alternatives resolved (FP8 agent):** the "F8Emulation" library **does not exist** (ROCm/F8Emulation and amd/F8Emulation are 404; zero GitHub hits; the similar-shaped project was Intel's discontinued FP8-Emulation-Toolkit). AMD's real FP8-emulation vehicle is **Composable Kernel**: `CK_USE_FP8_ON_UNSUPPORTED_ARCH` + `gemm_xdl_fp16_fp8` examples (FP8 storage, FP16 compute) — GEMM/conv only, "functional support only" on MI100/MI200 per AMD's own README, and wu1w measured **CK FP8 instances hanging LLVM on CDNA2** (INT8 JITs fine) → CK FP8 is a risky side-route; **Triton is the primary route**. Even HIP-level FP8 *types/conversions* are gated off CDNA2 in ROCm's precision tables — but Triton-level e4m3→bf16 conversion demonstrably works (davetha's kernel does it).
- **Key dev doc:** davetha's repo ships an 806-line `FP8-GFX90A.md` — bf16 MFMA measured at **165.5 TFLOP/s** on MI210, plus the MFMA subnormal-flush hardware finding. Read before writing the W8A8 kernel.
- **Performance reference (the bar to match):** INT8 W8A8 on the same model class measures **54.32 tok/s (MTP n=3)** / **~80 tok/s (DFlash2 N=12)** on 1× MI210. Emulated FP8 W8A8 reads the same ~28 GB of 8-bit weights → same memory-bound ceiling → the target is to match INT8's numbers while serving the official FP8 checkpoint unmodified (INT8 would require re-quantization; the user's goal is the official W8A8 checkpoint). Note the FP8 agent's caveat: for *compute-bound* prefill, emulated FP8 gains nothing over BF16 (165.5 TFLOP/s ceiling) — the W8A8 win is checkpoint fidelity + activation traffic, not FLOPs.
- Memory check (measured): 27B @ 8-bit ≈ 27.5–31 GB weights → ~29 GB left for KV on 64 GB → 262–375K KV tokens measured on this hybrid model. Fits comfortably.
- Full details: [`fp8-w8a16-davetha-details.md`](fp8-w8a16-davetha-details.md) · [`fp8-mi210-research.md`](fp8-mi210-research.md).

## 5. W4A16 on MI210 — verdict: works on stock vLLM via the Triton kernel; the cached Quark checkpoint is the right one

**The W4A16 checkpoint exists and is already cached on the scout machine (19 GB): [`amd/Qwen3.8-27B-Quark-Qronos-INT4-W4A16`](https://huggingface.co/amd/Qwen3.8-27B-Quark-Qronos-INT4-W4A16)**

- INT4 weight-only (W4A16), **group size 128**, BF16 activations; produced with AMD Quark's **Qronos** algorithm (Hessian-based PTQ, [ICLR 2026](https://arxiv.org/abs/2505.11695)), native `qwen3_5` support
- Quality: GSM8K 101.4% / 99.2% recovery (thinking/non-thinking), Wikitext ppl 97.2%, BFCL 97.6% — essentially lossless
- **Kernel verdict (gfx90a):** vLLM's `QuarkW4A16Int4` scheme routes through `TritonW4A16LinearKernel` on ROCm — the same kernel GPTQ-Int4 and compressed-tensors int4 use. On gfx90a the RDNA kernels are gated out, so **Triton W4A16 is the path for all of them**; it is directly MI210-validated (maintainer-run W4A16 Triton benchmarks on 2× MI210, [vllm#52983](https://github.com/vllm-project/vllm/pull/52983); MI250 is a vLLM CI agent pool running these kernel tests).
- **Version requirement: vLLM ≥ 0.30.0** — native Quark W4A16Int4 support ([#48606](https://github.com/vllm-project/vllm/pull/48606), merged 2026-09-18, first shipped in v0.30.0). Older vLLM — including the 0.28.x community MI210 stacks — cannot load Quark int4 exports at all. Stock `vllm==0.30.0+rocm723` wheel / `vllm/vllm-openai-rocm:v0.30.0` image serves it on gfx90a; W4A16 needs no MI210-specific patches. (The model card's other PR ref, [#46110](https://github.com/vllm-project/vllm/pull/46110), is ROCm platform *detection* — only relevant when amdsmi fails.)
- Precedent: a user served **this exact checkpoint on MI100** (gfx908) — hit a config-mapping bug on pre-#48606 main ([#52454](https://github.com/vllm-project/vllm/issues/52454), `apply_vllm_mapper` AttributeError); the #48606 rewrite fixes that failure mode. No MI210/MI250/MI300 success reports yet — we would be first.
- Fallbacks if the Quark loader misbehaves on gfx90a: sibling `amd/Qwen3.8-27B-Quark-AWQ-INT4-W4A16` (same format, isolates checkpoint-vs-loader); GGUF via llama.cpp/Ollama (both officially support gfx90a); re-quantize the BF16 base with GPTQModel/AutoAWQ; or `Qwen/Qwen3.5-27B-GPTQ-Int4`. Note: the Quark export does NOT load via GPTQ/AWQ kernels (proprietary pack format), and Quark cannot re-export to GPTQ/AWQ (only quark-HF/ONNX/GGUF).
- Kernel landscape on gfx90a: Marlin/Machete = CUDA-only; Exllama now works on ROCm (fp16-act only, `--linear-backend exllama`); **AWQ is slow on ROCm** (legacy `awq_gemm` path; fix in draft [#55098](https://github.com/vllm-project/vllm/pull/55098), 6–8× TPOT) → prefer GPTQ/Quark/CT int4 over AWQ.
- **davetha's fork adds gfx90a-optimized W4A16**: 104-CU tile table (stock tiles assume MI300's 304 CUs), magic-bias dequant (gfx90a lacks bf16 VALU arithmetic), int4 interleave packing (1.45–4.8× on MoE kernels) — use if stock Triton W4A16 underperforms.
- **The fast W4A16 kernel exists — in SGLang**: hyl64's [`mi210-qwen3.8-inference`](https://github.com/hyl64/mi210-qwen3.8-inference) ships a custom **int8-MFMA Marlin-style W4A16 GEMM for gfx90a** (`kernels/marlin_epi2.hip`): int4 weights stay packed, int8 activations via 2-pass even/odd separation, fused group-128 scale/zero-point epilogue — **709 GB/s vs Triton's 327 GB/s (2.2×)**, rel err 1e-6. Qwen3.8-27B AWQ W4A16 + DFlash2 on SGLang: **59 → 70–113 tok/s, peak 122.45 tok/s**. Porting this kernel into vLLM is the highest-leverage W4A16 work item. Guardrail finding: fp16 MFMA overflows GDN output (Inf); bf16 is safe — applies to all custom kernels on this model.
- Memory: ~15–16 GB weights → ~105 tok/s memory-bound ceiling before MTP — large headroom vs the 100 tok/s target.
- **Risk flag (full multimodal):** ROCm bench recipes for the multimodal 27B Qwens use `--language-model-only` ([vllm#52391](https://github.com/vllm-project/vllm/pull/52391)) — whether the ViT tower works on gfx90a needs on-box verification, since the user wants vision serving.
- Full details: [`w4a16-mi210-research.md`](w4a16-mi210-research.md) in this workspace.

## 5b. INT8 W8A8 on MI210 — verdict: feasible, native, and the best-measured path (the performance baseline)

- **Hardware: native.** gfx90a has INT8 MFMA (`v_mfma_i32_16x16x16i8`, 181 TOPS) — no emulation needed, unlike FP8.
- **Kernels: working today.** AITER CK INT8 `gemm_a8w8` runs on gfx90a two ways: wu1w's JIT patch on stock vllm 0.28.0+rocm723 (move the wheel's gfx942-only `module_gemm_a8w8.so` aside first; ~4 min JIT) or davetha's repatched AITER inside his image. **2.9× over the silent Triton fallback** (18.71 → 54.32 tok/s).
- **Checkpoints (Qwen3.8-27B W8A8 INT8, all compressed-tensors):**
  - [`Freaksterz/Qwen3.8-27B-SmoothQuant-W8A8-INT8`](https://huggingface.co/Freaksterz/Qwen3.8-27B-SmoothQuant-W8A8-INT8) (~31 GB, 35.8K downloads) — wu1w's reference checkpoint
  - [`davetha/Qwen3.8-27B-ABLITERATED-W8A8-gdnint8`](https://huggingface.co/davetha/Qwen3.8-27B-ABLITERATED-W8A8-gdnint8) (28 GB) — **fully quantized: INT8 GDN projections + INT8 lm_head** (stock W8A8 leaves them BF16 ≈ 40% of decode bytes), tagged gfx90a/mi210/rocm — the 80.6 tok/s build
  - [`RukaRat/Qwen3.8-27B-INT8-W8A8-imatrix-MTP`](https://huggingface.co/RukaRat/Qwen3.8-27B-INT8-W8A8-imatrix-MTP) — **multimodal + MTP included** (candidate for full-vision INT8 serving)
  - [`Avesed/Qwen3.8-27B-INT8-W8A8`](https://huggingface.co/Avesed/Qwen3.8-27B-INT8-W8A8) — vision-language tagged
  - No *official* Qwen INT8 exists (official = BF16/FP8/NVFP4/GGUF) — community SmoothQuant/llm-compressor exports, or self-quantize (Quark supports W8A8 INT8; davetha ships a convert image)
- **Measured on 1× MI210, Qwen3.8-27B W8A8 (262K ctx):**

  | Setup | Decode tok/s |
  |---|---:|
  | Stock Triton INT8 (silent fallback) | 18.71 |
  | AITER CK INT8 + MTP n=3 (wu1w, Freaksterz ckpt) | 54.32 |
  | davetha gdnint8, no speculation | 36.4 |
  | + MTP N=1 / N=2 | 57.1 / 68.2 |
  | + dflash N=8 / **N=12 (optimum)** / N=16 | 78.0 / **80.6** / 74.9 |
  | Aggregate @ 8 / 16 concurrent streams | 312 / 413 |
  | Prefill @ 3.5K ctx | ~2,070 tok/s |
  | KV capacity | ~375K tokens (28–31 GB weights) |

- **Key levers (measured):** CK INT8 GEMM ≈2.9×; quantize GDN+lm_head (40.5→57.2); speculation is the largest decode lever (36.4→80.6); FULL_DECODE_ONLY graphs 1.7×; keep the DFlash2 draft in BF16 (INT8 draft is net −6.5%).
- **davetha's production ladder ([mi210-llm-stack](https://github.com/davetha/mi210-llm-stack)) goes higher**: 18.3 → 42.3 (+AITER CK) → 80.0 (+MTP n2) → **108.5 tok/s (MTP n5)**; **DFlash2 (block-diffusion drafting, [vllm#52816](https://github.com/vllm-project/vllm/pull/52816)) ported to gfx90a: 168 tok/s decode, 2.2×, 98.6% draft acceptance** — both exceed the 100 tok/s target on INT8. Operational traps that make the difference: vLLM silently downgrades cudagraph_mode to PIECEWISE under spec decode (force `FULL_DECODE_ONLY` — worth ~60% of decode); `--max-model-len` above 128K costs 10× decode (gfx9 attention gate evaluated at graph capture); power cap 199 W → 300 W is worth +8.4% prefill.
- **Caveats:** 100 tok/s not yet reached on INT8 (80.6 max — the gap needs the post-tag fused-GDN-decode work and/or better acceptance); wu1w measured dflash acceptance 0 on the *Freaksterz* checkpoint ("do not use") while davetha's own gdnint8 build works — checkpoint-dependent; silent-failure greps mandatory (AiterInt8 kernel selected, FULL graphs captured, dflash present); version pins: stock 0.28.0+rocm723 + wu1w patches, or davetha's image.
- **Role in the mission:** INT8 W8A8 is the Phase-1 baseline deployment and the performance bar (~54–81 tok/s) the W8A8 FP8 development must match — same ~28–31 GB weight reads, same memory-bound ceiling, but native MFMA and zero kernel development required.

## 6. Community prior art on MI210 — two convergent one-person projects, both targeting Qwen3.8-27B

- **[`davetha/mi210-vllm`](https://github.com/davetha/mi210-vllm)** (created 2026-08-04, active through 2026-09-10): "Pinned, gate-verified vLLM deployment stack for AMD MI210". Pins fork `davetha/vllm` @ v0.28.1rc0+mi210.7 — 14 gfx90a patches on upstream main — atop AMD's `rocm/vllm:rocm10.0.0_…vllm_0.27.0` base (ROCm 10.0.0 / torch 2.12.0), AITER v0.1.21 + **aiter-cdna2 v1.6** (binary repatcher rewriting AITER's gfx942-only code objects for gfx90a). Carries: W4A16 gfx90a tile/dequant patches, int4 interleave packing (1.45–4.8× on MoE), **FP8 W8A16 Triton kernel**, NVFP4, fused-GDN-decode. 3-tier hardware-verified build gates. **Target model: Qwen3.8-27B.**
- **[`wu1w/vllm-mi210`](https://github.com/wu1w/vllm-mi210)** (created 2026-08-31, single push, explicitly anti-fork): site-packages patches for **stock vllm 0.28.0+rocm723** on ROCm 7.2.x — opens AITER's gfx90a gate + JITs CK INT8 `gemm_a8w8` (2.9× over Triton: 18.71 → **54.32 tok/s on Qwen3.8-27B W8A8, MTP n=3, 262K ctx**), plus a HIP fused GDN decode kernel with runtime GQA (numerics-correct; no extra speed once GEMM is CK).
- Both converge on the same root cause: **AITER gates on CDNA3 and ships no gfx90a binaries** — the patches open that gate.
- Caveat: both are one-person efforts with near-zero external user base — plan for self-support.
- Full details, benchmark tables, install commands: [`mi210-vllm-research.md`](mi210-vllm-research.md) in this workspace.

### 6.1 User's own prior work — local assets on gfx1100 (W7900), directly reusable

From `~/vllm_qwen/` (Sept 2026, vLLM 0.28.0+rocm723 + aiter 0.1.19, same Quark-Qronos checkpoint, WSL2 — see `SESSION_FINDINGS.md`, `AITER_W4A16_GFX1100.md`):

- **`gemm_a16w4` — custom AITER Triton W4A16 kernel** (fused INT4 nibble unpack via `tl.interleave` + WMMA `tl.dot`, split-K, per-M-band tuned configs) + `AiterW4A16LinearKernel` vLLM MPLinear integration + `quark_aiter_gemm` custom op. Verified exact vs fp32 and vs the production RDNA3 HIP kernel; E2E greedy outputs identical. **But E2E slower than the RDNA3 HIP kernel on gfx1100 (16.3 vs 23.8 tok/s)** — demoted. Lessons: microbench wins don't transfer E2E; per-M Triton compile storms (`GRID_MN` in the cache key); tune against real serving shapes (M=2/4/8/16 MTP verify batches). On gfx90a the baseline is the generic TritonW4A16 kernel (no RDNA3 HIP kernel exists there) — porting `gemm_a16w4` (new gfx90a config JSON + gate relax) is a plausible optimization, with realistic expectations.
- **FP8 KV cache bit-trick conversions** — branchless e4m3→f32/f16 bit-placement in Triton paged-attention kernels (decode 2.82→1.02 ms @32K, verify 2.94→1.52 ms; fp8 KV decode +6–12% at all ctx, 2× KV capacity). **Directly applicable to the emulated W8A8 GEMM's in-kernel fp8→bf16 upcast.**
- Operational playbooks: stale `~/.cache/vllm/torch_compile_cache` crashes on kernel switch (clear it); autotune compile-storm deadlocks; GPU-faulting ad-hoc tests kill co-resident servers; `serve-qwen.sh` production config: `--reasoning-parser qwen3 --tool-call-parser qwen3_coder`, MTP via `--speculative-config '{"method":"qwen3_5_mtp","model":"<same>","num_speculative_tokens":3}'`, gpu_util 0.88, max_len 262144, max_seqs 8, batched_tokens 24576; KV math: ~64 KB/token (16 full-attn layers) + ~151 MB/seq GDN state.
- Remote 2× W7900 TP2 box (`tai@192.168.12.14`) currently blocked by a failed DRAM DIMM (hardware).

### 6.2 AITER & the gfx90a kernel landscape — full deep-dive in [`aiter-mi210-kernels-research.md`](aiter-mi210-kernels-research.md)

- **AITER upstream (v0.1.23, 2026-09-26) will not help gfx90a**: healthy project (bi-weekly releases; tuned configs for Qwen3.8-27B MXFP4 on gfx950), but ships ASM only as 2,863 prebuilt `.co` blobs (gfx942/gfx950/gfx1250, zero assembly sources), omits gfx90a from its supported-hardware table, has no gfx90a tuned configs (Triton config lookup fails with `KeyError: 'default'`), and its FP8-bearing JIT modules don't compile for gfx90a unpatched. Every gfx90a win comes from the **source-JIT layers (HIP and CK)** + community gate-patching.
- **davetha/aiter-cdna2 proves the ASM ceiling**: 242 of 1,422 gfx942 kernels binary-translate to gfx90a — a hard hardware ceiling (FP8, INT8-K32 MFMA spellings, `global_atomic_pk_add_bf16` all absent on CDNA2). Useful subset: `fmha_v3_fwd` bf16 prefill (1.36–1.86× vs SDPA) and `pa_fwd_asm` decode (1.7× over HIP, ~1% e2e). AMD ask ([aiter#4524](https://github.com/ROCm/aiter/issues/4524)) unanswered.
- **rlrs stack (all open drafts)**: AITER HIP paged-attention decode 3.94×/3.81× vs Triton at 128K/256K + CK batch prefill 1.70× with 2048-token pages for hybrid models ([aiter#4387](https://github.com/ROCm/aiter/pull/4387)/[#4388](https://github.com/ROCm/aiter/pull/4388)/[#4389](https://github.com/ROCm/aiter/pull/4389), [CK#3760](https://github.com/ROCm/composable_kernel/pull/3760), [vllm#49888](https://github.com/vllm-project/vllm/pull/49888)) — **but measured at head_dim 128 only; hd-256 coverage for Qwen3.8-27B's 16 full-attention layers is the make-or-break verification item**. FA2-CK officially supports MI200 with head dims ≤ 256 — the fallback library.
- **GDN decode is not the bottleneck** once GEMMs are on CK (wu1w: HIP fused GDN 53.54 vs 54.32 tok/s — parity). AITER's Triton GDN kernels are one `@if_aiter_supported` decorator away from gfx90a — an untested one-line experiment.
- **Gap matrix for this model on this card**: hd-256 attention path (Triton today), zero upstreamed gfx90a patches, fast W4A16 kernel port (hyl64 → vLLM), AITER Triton GDN unblock, ViT profiling cap. Every gap has a working prototype somewhere in the 12-repo gfx90a ecosystem.

### 6.3 HyperQwen (syv-ai) — the performance-engineering reference: same model, vLLM, 127 tok/s on a 3090

[`syv-ai/HyperQwen`](https://github.com/syv-ai/HyperQwen) (vLLM 0.29.0 + 38-patch series, Apache-2.0): **Qwen3.8-27B on one RTX 3090 (24 GB, 936 GB/s, sm86)** — **127 tok/s single-stream**, 381 tok/s when the answer quotes its own prompt, ~1,035 tok/s aggregate @64 concurrent, 150K context. The measured, cumulative ladder:

| step | single-stream tok/s |
|---|---|
| W4A16 AutoRound body + fp8 KV, no speculation | 46 |
| MTP-2 as shipped (bf16 drafter, full 248k head, fp32 GDN state) | 66–79 |
| MTP-4 + int8 drafter + 40k draft head + fp16 GDN state | 78–99 |
| + probabilistic draft sampling, sampler patch, split-KV verify attention | 93–99 |
| + draft vocab counted over the model's own outputs (97.5% coverage) | 107–109 |
| + GPTQ-int4 lm_head + int4 MTP module | 114–124 |
| DFlash2 block drafter (int4-requantized, 3.85 → 1.19 GB) | 118–126 |
| + lookup drafting (n-gram from context) + 16-token verify block | **127–133 (381 quoting)** |

Techniques and MI210 portability:
- **Model-prep pipeline (portable as-is, card-agnostic)**: requantize `lm_head` + `embed_tokens` to int8 group-128 (public W4A16 quants leave two 2.5 GB bf16 matrices alone — 2.6 GB back); int4-requantize the DFlash2 drafter; **40k-token draft vocabulary counted over 5.4M tokens of the model's own outputs** (97.5% coverage vs 92% for a web-text list; every out-of-vocab draft token is a forced rejection — the vocab alone was worth 10%).
- **fp16 GDN recurrent state** (`--mamba-ssm-cache-dtype float16`): halves the ~150 MB/seq state footprint and traffic; perplexity unchanged (fp16 keeps 10 mantissa bits vs bf16's 7). Portable flag.
- **W4A8 int8-activation Marlin path** (int4 weights + int8 activations on int8 tensor cores, 4× fp16 rate on sm86; negative-scales bug fix + per-layer select) — the MI210 analog is **hyl64's int8-MFMA Marlin kernel extended to int8 activations**. Caveat: gfx90a's INT8 MFMA is same-rate as BF16 (~181), so the 4× is Ampere-specific; the win on MI210 is activation-traffic reduction, not MMA rate.
- **Split-KV verify attention** (Triton): FA2 leaves 58/82 SMs idle on the multi-query verify step; 57 µs → 23 µs per layer. Portable Triton, needs gfx90a tuning.
- **Lookup drafting + long verify blocks**: draft from the request's own context (n-gram), verify 15 tokens/step while copying — lossless (point-mass drafts, greedy verify). vLLM-level feature, portable.
- **int8-QK prefill attention** (SageAttention-style, gated on this model's exact 24Q/4KV/hd-256 geometry): 1.27–1.35× on attention — **but the 2× int8 tensor-core rate is sm86-specific; on gfx90a INT8 ≈ BF16 rate, so this does NOT transfer as a win**.
- **Hybrid prefix caching** (recurrent state resumes from cached block boundary): follow-up turn on a 24K doc 23 s → 1 s. Portable. KVarN 4/2-bit KV (240K ctx), hybrid KV-groups fix, vision-tower CPU offload, memory-profiling/cudagraph-accounting fixes — vLLM patches, portable with rebase (series cut against 0.29.0, `--fuzz 0` discipline).
- **Implication for MI210 targets**: the 3090's 936 GB/s over ~15 GB of int4 weights gives a ~62 tok/s no-spec ceiling; HyperQwen reaches 127 = ~2.05× via speculation engineering (3.3–3.4 tokens/step). MI210 has 1.75× the bandwidth → same playbook on a W4A16 body implies **~109 tok/s no-spec ceiling, ~180–220 tok/s with the full drafter engineering**. davetha's 80.6 (dflash N=12) is less than half of what this model supports with better speculation.

## 7. SGLang on MI210 — verdict: a porting project, not a deployment → **framework = vLLM**

- Qwen3.8-27B requires SGLang ≥ v0.5.19 ([PR #34859](https://github.com/sgl-project/sglang/pull/34859)); current is v0.5.20. But on MI210:
  - `setup_rocm.py` **hard-exits on gfx90a** (only gfx942/gfx950/gfx1250 allowed); FP8 macros compiled only for gfx942/gfx950/gfx1250.
  - **Zero MI200 docker images** exist (all 897 `lmsysorg/sglang-rocm` tags are mi30x/mi35x/mi45x); `rocm.Dockerfile` builds gfx942/gfx950/gfx1250 only and bakes `SGLANG_USE_AITER=1`.
  - **AITER does not support MI2xx** (SGLang only fails gracefully since [PR #7187](https://github.com/sgl-project/sglang/pull/7187)); the non-AITER path historically required co-installed vLLM ROCm wheels.
  - Historical MI210/MI250 runs needed patched stacks (triton attention, hipblaslt fix); **TP>1 on MI250 produced garbage output — unresolved** ([#7641](https://github.com/sgl-project/sglang/issues/7641)).
- FP8: the only MI210 FP8 work is **unmerged** [PR #17082](https://github.com/sgl-project/sglang/pull/17082) (pure-PyTorch FP8 kernels, Triton MoE, AITER off). No block-FP8 serving success on MI200 found anywhere in SGLang.
- W4A16: **GPTQ is dead on AMD in SGLang** (gptq_marlin CUDA-only; non-Marlin GPTQ deleted in v0.5.20 [PR #32114](https://github.com/sgl-project/sglang/pull/32114); ROCm misroute crashes hipcc [#33015](https://github.com/sgl-project/sglang/issues/33015)); AWQ works but slow (Triton dequant+matmul); no Qwen3.8-27B GPTQ/AWQ checkpoints. `petit_nvfp4` works on MI250 but the model's NVFP4 export is W4A4 with FP8 projections (same FP8 blocker).
- The SGLang Qwen3.8-27B cookbook verifies only NVIDIA hardware (H200 / RTX PRO 6000 / 5090 / DGX Spark).
- **Conclusion: vLLM (stock + wu1w patches, or davetha fork) is the serving stack.** SGLang only if we later want to port it — a bigger project than the W8A8 kernel work itself. **Nuance:** hyl64's [`mi210-qwen3.8-inference`](https://github.com/hyl64/mi210-qwen3.8-inference) *did* port SGLang to MI210 for this exact model (2 sgl-kernel patches + the custom int8-MFMA Marlin W4A16 kernel, 122 tok/s peak) — proving the port is doable and producing the fastest published W4A16 stack on this card; its kernel is the port target for vLLM W4A16. Full details: [`sglang_mi210_qwen38_research.md`](sglang_mi210_qwen38_research.md).

## 8. Requirements & performance feasibility (user-confirmed)

- **Access:** user powers the box on and provides SSH; **Ubuntu + ROCm already installed** (version TBD after boot).
- **Serving scope:** **full multimodal** (vision tower must work, not text-only).
- **Kernel quality:** **fully fused kernels** required (fused GDN scan, fused attention, dequant-fused GEMMs — no eager/unfused fallbacks).
- **Performance target:** **≥ 100 tok/s single-stream decode with FP8 + MTP** (user's own calculation).

Feasibility sanity check (single MI210, ~1.6 TB/s HBM2e):
- FP8 weights ≈ 28 GB → memory-bound decode ceiling ≈ 1600/28 ≈ **57 tok/s** without speculation (weights re-read every token; GDN state is constant-size, KV only on 16 layers).
- vLLM recipe reports MTP acceptance 0.77–0.90 for this model → effective speedup ~1.5–1.9× → **~85–110 tok/s**.
- Conclusion (updated): the 100 tok/s target is **demonstrably beatable** — davetha's INT8 stack measures 108.5 (MTP n5) / 168 tok/s (DFlash2) on MI210, and HyperQwen proves 127 tok/s on a 3090 with 43% less bandwidth via pure speculation engineering (§6.3). Applied to MI210's 1,638 GB/s, the same playbook implies **~180–220 tok/s** on a W4A16 body. That is the bar the FP8 W8A8 work should be benchmarked against.
- W4A16 (~15 GB reads → ~105 tok/s ceiling before MTP) has far more headroom, but the user's stated target is the FP8 path.

## 9. Bring-up & development plan (final)

**Phase 0 — box on (user provides SSH):** inventory — ROCm version, `rocm-smi`/`rocminfo` (gfx90a visible; single GPU must be physical device 0 — vLLM resolves arch once from amdsmi device 0 and ignores `HIP_VISIBLE_DEVICES`), disk, docker, internet; check power cap (199 W default → 250/300 W is worth +2.9/+8.4% prefill).

**Phase 1 — INT8 W8A8 baseline (day one, zero kernel dev):**
- Stack: stock `vllm==0.28.0+rocm723` (Py3.12) + wu1w patches (`apply_gfx90a.py`, `jit_gemm_a8w8.py` — move the wheel's gfx942 `module_gemm_a8w8.so` aside first), or davetha's pinned image
- Checkpoint: `davetha/Qwen3.8-27B-ABLITERATED-W8A8-gdnint8` (fully quantized) or `Freaksterz/...-SmoothQuant-W8A8-INT8`
- Spec decode: MTP n=3, then the n2→n5 ladder; DFlash2 N=12 per davetha's recipe
- Ops traps: force `FULL_DECODE_ONLY` graphs (silent PIECEWISE downgrade under spec decode costs ~60%); `--max-model-len` ≤ 128K initially (>128K costs 10× decode); clear `~/.cache/vllm/torch_compile_cache` after JIT; `--safetensors-load-strategy eager`; no fp8 KV on 0.28; grep that `AiterInt8ScaledMMLinearKernel` was selected
- Text-only first (`--limit-mm-per-prompt '{"image":0,"video":0}'` — the ViT profiling path attempts a 256 GiB allocation and OOMs 64 GB cards)
- Expected: 54–80 tok/s; davetha's ladder reaches 108.5 (MTP n5) / 168 (DFlash2)

**Phase 1b — HyperQwen playbook (the speculation engineering — the biggest single lever):**
- Port syv-ai/HyperQwen's card-agnostic work (§6.3): requantize `lm_head`/`embed_tokens` to int8 group-128; int4-requantize the DFlash2 drafter (3.85 → 1.19 GB); build the 40k draft vocabulary from the model's own outputs; fp16 GDN state; lookup drafting + long verify blocks; hybrid prefix caching
- Kernel-side ports for gfx90a: split-KV verify attention (Triton, tune for 104 CUs); W4A8 int8-activation variant of hyl64's Marlin kernel
- Target: **>127 tok/s** (beat the 3090 reference on MI210's 1.75× bandwidth); realistic ceiling ~180–220 tok/s on the W4A16 body

**Phase 2 — W4A16 (Quark-Qronos, the cached 19 GB checkpoint):**
- Stock `vllm==0.30.0+rocm723` (Quark W4A16Int4 needs ≥ 0.30.0) — Triton W4A16 kernel works on gfx90a (~327 GB/s)
- Optimization: port hyl64's int8-MFMA Marlin kernel (709 GB/s) from SGLang to vLLM — the biggest W4A16 lever; or davetha's W4A16 tile/dequant patches (1.37–1.45×)
- Version note: INT8 stack pins 0.28.x, Quark needs 0.30.0 — run two venvs/containers initially; consolidate later (verify wu1w patches apply on 0.30)

**Phase 3 — FP8 W8A8 (the user-assigned development work — ranked route):**
- **Step 0 — try stock first**: vLLM 0.28+/0.30 ROCm (Triton ≥3.5 auto-emulates fp8 `tl.dot` as fp16 MFMA on gfx90a) + the official `Qwen/Qwen3.8-27B-FP8` checkpoint; verify `Selected TritonFp8BlockScaledMMKernel`, numerics vs BF16, benchmark
- **Step 1 — tune** the Triton block-kernel configs for MI210's 104 CUs; replace the generic software fp8→fp16 conversion with the **3-instruction bit-trick** (never the stock 11,997-instruction decoder); **bf16 dot + fp32 accum** (fp16 MFMA flushes subnormals; GDN outputs overflow fp16)
- **Step 2 — fallback**: extend davetha's PR #52985 W8A16 kernel to W8A8 (per-token-group quant + in-loop 128×128 block scales)
- Avoid: CK FP8 blockscale (numerically wrong — ¼ of the sum; FP8 instances also hang LLVM on CDNA2), AITER FP8 (gated), fp8_e5m2 KV (mis-stored; e4m3 KV is bit-correct)
- Validate: fp32 reference on all layer shapes (the `~/vllm_qwen` test pattern) → e2e greedy vs BF16 base → MTP acceptance
- Bar: the HyperQwen-adjusted target (§6.3) — INT8-class decode at minimum, ideally the full speculation-engineering numbers

**Phase 4 — full multimodal + fused kernels:**
- ViT tower: cap `--limit-mm-per-prompt` / video-preprocessor `longest_edge` (256 GiB profiling OOM trap); FA2-CK (MI200, hd ≤ 256) for the tower if SDPA is slow
- Full-attention hd-256 layers: Triton today; verify the rlrs stack at hd 256 (make-or-break item) or wire FA2-CK; davetha's attention-partitioning fix for long context
- GDN: Triton chunked prefill (fine); fused GDN decode optional (not a bottleneck); AITER Triton GDN unblock = one-decorator experiment
- Benchmark: TTFT / ITL / tok/s vs targets; concurrency scaling (312 tok/s @8 expected on INT8)

**Open decisions for the user:** (1) two-stack version split (0.28 INT8 / 0.30 W4A16) vs consolidate on 0.30; (2) port hyl64's Marlin kernel to vLLM now or after the W8A8 work; (3) FP8 W8A8 acceptance criteria (match INT8 within X%?).

## 10. Sources

- Qwen3.8-27B model card: https://huggingface.co/Qwen/Qwen3.8-27B
- Qwen3.8-27B-FP8: https://huggingface.co/Qwen/Qwen3.8-27B-FP8
- Quark-Qronos INT4 W4A16: https://huggingface.co/amd/Qwen3.8-27B-Quark-Qronos-INT4-W4A16
- vLLM recipe: https://recipes.vllm.ai/Qwen/Qwen3.8-27B
- Community repos: https://github.com/davetha/mi210-vllm · https://github.com/wu1w/vllm-mi210 · https://github.com/davetha/mi210-llm-stack · https://github.com/davetha/aiter-cdna2 · https://github.com/hyl64/mi210-qwen3.8-inference
- Detailed research files (this workspace): `mi210-vllm-research.md` · `w4a16-mi210-research.md` · `sglang_mi210_qwen38_research.md` · `fp8-w8a16-davetha-details.md` · `fp8-mi210-research.md` · `fp8-w8a8-emulated-mi210.md` · `aiter-mi210-kernels-research.md`
