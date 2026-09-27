# GEMM Optimization Handoff — W8A8 FP8 Block Kernel on AMD MI210 (gfx90a)

**For: the next agent continuing GEMM optimization.** Everything here is measured on the target hardware. Read `BRINGUP_LOG.md` for the full mission history; this doc is the working kit.

**UPDATE 2026-09-27 (evening) — W8A16 decode path SHIPPED, +25% e2e.** The
M≤16 path runs A as raw bf16 (no activation quant — the MI210 has no fp8
MFMA; the quant round-trip was pure overhead) on a new kernel with ONE config
everywhere: (16,64,128)@4w/2s + split-K=4. Kernel-level vs the W8A8 ladder
below: gate_up +21%, gdn +29/35%, down +22%, o +19%, qkv +46%, mtp_fc +26%
≈ **31% GEMM-sum cut**. Measured e2e (same-config A/B, `KERNEL_BENCHMARK.md`):
**53.0 → 66.4 tok/s avg = +25.3%** (acceptance unchanged ~46%). All 42
correctness checks pass (rel ~2–3e-3).

**⚠️ THE BUG THAT ALMOST SANK IT (learn this one):** the first deploy was
e2e-NEUTRAL despite the kernel wins — the W8A16 path never executed. The
model is compiled ONCE for the range (1, max_num_batched_tokens) and traced
at the profile shape (M=4096); a Python-level `if M <= 16` in
`apply_block_scaled_mm` is evaluated at TRACE time and specialized away —
the compiled graph baked in quant→W8A8 for EVERY M. The M-branch must live
INSIDE the opaque custom op (`_w8a16_block_gemm_dispatch`), which runs
eagerly per call including during CUDA-graph capture. Diagnose dispatch
questions with one-shot prints inside the custom op (never in traced Python —
dynamo rejects `print`).

## Mission context (30 seconds)

Serve **Qwen/Qwen3.8-27B-FP8** (official checkpoint, e4m3, 128×128 weight blocks, dynamic per-token-group activations) on **one AMD Instinct MI210** (gfx90a/CDNA2, 64 GB HBM2e, 104 CUs, ~1.6 TB/s, **no native FP8 MFMA**). The custom Triton W8A8 kernel below replaced the stock `TritonFp8BlockScaledMMKernel`. Current e2e: **~40–56 tok/s** single-stream with MTP n=3 (~62% acceptance). The GEMM sum is ~38 ms/token — the remaining optimization target.

## Where things live

| What | WSL workspace (`/home/jebc/vllm_mi210/`) | Box (`tai@192.168.12.17`) |
|---|---|---|
| **Harness kit (start here)** | `gemm-harness/` | `~/gemm-harness/` (identical) |
| Kernel + tests + sweeps | `gemm-harness/*.py` | `~/gemm-harness/*.py` |
| One-command runner | `gemm-harness/run-test.sh <script.py> [--stop-server]` | same |
| Kernel gate | `gemm-harness/verify.sh` (correctness + bench; stops server) | same |
| E2E benchmark | `gemm-harness/bench-e2e.sh [model] [port]` | same |
| Launch script | `start-fp8-aiter.sh` | `~/start-fp8-aiter.sh` (authoritative) |
| Bring-up log | `BRINGUP_LOG.md` | — |
| Tuned configs (stock kernel) | — | `~/tuned/` + `~/tuned-amd/` (installed in container) |
| Patched fp8_utils (bf16 cast) | — | `~/fp8_utils_fork.py` |
| Model | — | `/home/tai/models/qwen38-27b-fp8` (29 GB) |

The harness is smoke-tested end-to-end (2026-09-27): `./verify.sh` runs the full gate in a throwaway container and reproduces the reference numbers.

**Box access** (sshpass is pty-blocked by the local sandbox — use this instead):
```bash
SSH_ASKPASS=/tmp/askpass.sh SSH_ASKPASS_REQUIRE=force DISPLAY=:0 ssh -T tai@192.168.12.17 '<cmd>'
# /tmp/askpass.sh = 2-line script echoing the password; recreate if missing
```

**Server**: `bash ~/start-fp8-aiter.sh` on the box (recreates container `fp8-aiter` from `local/vllm-mi210:mi210.6-aiter`, installs all patches, ~4–5 min to healthy). Port 8001. Health: `curl localhost:8001/health`. **One GPU — stop the server before any GPU test** (`docker stop fp8-aiter`; a leftover server silently corrupts benchmarks — this bit us once).

## The kernel

`w8a8-kernel/triton_fp8_w8a8_block.py` — a fresh Triton kernel (davetha's W8A16 gfx90a kernel as the design ancestor; NOT gfx942 code). Key facts, all measured:

- **Decode**: both operands raw uint8 → `x.to(tl.float8e4nv, bitcast=True).to(tl.bfloat16)` (native cast, ~4.5 ops, exact on all 254 finite codes incl. denormals). **The fp16 bit-trick LOSES to this** (davetha measured 26/27 rows; the kernel is weight-streaming-bound, not VALU-bound — don't chase conversion tricks).
- **Scales in-loop**: `acc += dot(a,b) * (a_s ⊗ b_s)` per K-tile (outer product once, single accumulator multiply).
- **HARD CONSTRAINT: BLOCK_K ≤ 128** (= group_k). BLOCK_K=256 spans two scale groups → **silent numerics corruption** (fluent wrong text). This bug shipped once; the guard now catches it.
- **Ladder (M≤16)**: N≥20000 → (16,128,64)@4w/3s; 12000≤N<20000 → (16,32,128)@2w/2s; 8192≤N<12000 → (16,128,64)@4w/3s; N<8192 → (16,64,128)@4w/2s + **split-K** (4 if K%512==0 else 2; fp32 partials + reduce kernel).
- **Stream-K** (persistent, atomic fp32, P=208) is implemented and tested but **reverted** — wins only on down/qkv (+6–32%), loses on big-N; net e2e small. Code retained in the file, unused.

**Per-shape state (M=1) vs the stock kernel (tuned):**

| shape (×layers) | ms | GB/s | vs stock |
|---|---|---|---|
| gate_up 34816×5120 (×64) | 0.275 | 645 | 1.24× |
| gdn 16384×5120 (×48) | 0.156 | 536 | 1.59× |
| gdn 14336×5120 (×48) | 0.154 | 476 | 1.60× |
| down 5120×17408 (×64) | 0.182 | 489 | 1.45× |
| o 5120×6144 (×16) | 0.075 | 420 | 1.37× |
| qkv 8192×5120 (×16) | ~0.110 | — | ~1.5× |

## The W8A16 path (M ≤ 16, shipped 2026-09-27 evening)

`_w8a16_block_gemm_kernel` in the same file. The kernel class sets
`apply_input_quant = False` (the fork's `BlockScaledMMLinearKernel` supports
exactly this — "subclasses that accept BF16 input directly"), so A arrives as
raw bf16 and `apply_block_scaled_mm` routes: bf16 + M≤16 → W8A16 kernel;
bf16 + M>16 → quantize via `self.quant_fp8(...)` (the registered `quant_fp8`
CustomOp — see gotchas) then the W8A8 ladder; fp8 → W8A8 ladder (unchanged,
still the prefill path).

- **Config**: (16,64,128)@4w/2s + split-K=4 for EVERY model shape (K%512==0
  holds for all: 5120, 17408=17×1024, 6144). Falls back sk2/sk1 if not.
  BLOCK_K=128 keeps the ≤128 scale-group constraint; BLOCK_N=64 keeps each
  tile inside one 128-wide b_s block (b_s is then a [BN] broadcast, no a_s
  outer product — the accumulator path is one broadcast multiply).
- **Why it wins**: no A decode (was ~23% of loop instructions — each warp
  decodes the full A tile), no a_s loads, no A LDS staging, and split-K=4
  fills the 104 CUs (e.g. qkv 128 tiles ×4 = 512 programs vs 62% occupancy
  before). Sweep evidence: `p3_w8a16_sweep.py`, `p3b_splitk.py`.
- **Numerics**: reference is `A_bf16 @ dequant(B)` — the W8A16 result is
  strictly MORE accurate than W8A8 (no activation quant error). Measured rel
  err 1.9–3.2e-3, same order as the W8A8 path's bf16 noise.
- **P0 facts behind the redesign** (all measured, see proposals doc): the
  fp8→bf16 decode is 5.5 ops/value (asm-counted; includes a per-value
  `v_mul_f32` by 2^120 — the OCP-fn bias fixup; the ROCm Triton backend has
  no native OCP-e4m3fn conversion); the kernel is latency-bound, not
  issue-bound (the decode ablation came out NEGATIVE — see gotchas); the
  [N,K] 128-B-row streaming pattern caps at 910 GB/s while 256/512-B rows
  hit the 1203 GB/s device ceiling (1D ceiling 1202, d2d copy 1248).

## The harness (how to test)

All scripts run in a throwaway container from the image (create → docker cp → start -a). Pattern:

```bash
# on the box:
docker stop fp8-aiter   # free the GPU first!
docker rm -f w8a8-test
docker create --name w8a8-test --entrypoint python3 \
  --device /dev/kfd --device /dev/dri \
  --group-add video --group-add "$(getent group render | cut -d: -f3)" \
  --ipc host local/vllm-mi210:mi210.6-aiter /test.py
docker cp ~/w8a8-kernel/triton_fp8_w8a8_block.py w8a8-test:/tmp/
docker cp ~/w8a8-kernel/test_w8a8_block_gfx90a.py w8a8-test:/test.py
docker cp ~/fp8_utils_fork.py w8a8-test:$V/.../fp8_utils.py   # see start script for $V
for j in ~/tuned-amd/*.json; do docker cp "$j" w8a8-test:$CFG/; done
docker start -a w8a8-test
```

- **`test_w8a8_block_gfx90a.py`** — THE gate: correctness (18 W8A8 + 24 W8A16 checks vs fp32 dequant references, rel err < 1e-2) + benchmark vs the stock kernel. Never ship a kernel change without it passing.
- **`sweep_w8a8.py`** — config sweep (direct kernel launches, explicit BLOCK_M/N/K × warps × stages; 2-phase coarse+refine; winners verified). How the old W8A8 ladder was found.
- **`p3_w8a16_sweep.py` / `p3b_splitk.py`** — the W8A16 + split-K sweeps that produced the shipped config.
- **`p0_dump_asm.py` / `p0_stream_ctrl.py` / `p0_stream_ctrl2.py`** — P0 diagnostics: GCN asm dump + per-source-line instruction attribution; streaming control kernels (the BW ceiling per tile shape; 1D/copy references).
- **`p10_w8a16_wide.py`** — the wide-load scale-in-decode experiment (LOST — register blowup; kept as the design record for the inline-asm retry).
- **`test_streamk.py`** — the stream-K experiment harness (prod vs sk104 vs sk208).
- **`tune-fp8-fix.py`** — the STOCK kernel's config sweep + the dequant-reference guard pattern (the reference function is the reusable part).

**Deploy path**: edit the kernel → run the test → if it wins, `bash ~/start-fp8-aiter.sh` (it installs `~/w8a8-kernel/triton_fp8_w8a8_block.py` into the container) → warm the server (first requests JIT-compile; discard the first 2–3 runs) → benchmark.

## Remaining headroom (ranked ideas — updated 2026-09-27 evening)

1. **Wide-load B (BK_LOAD 256/512) — the biggest remaining lever.** The
   control kernel proved 256/512-B rows stream at 1203 GB/s vs 910 for the
   128-B rows the BLOCK_K≤128 constraint forces. The W8A16 kernel is at
   821 GB/s (gate_up) — the gap to 1203 is the prize. Two known ways:
   (a) scale-in-decode (fold b_s×2^120 into the decode's bias multiply, one
   dot over the wide tile, zero accumulator math) — the p10 attempt LOST
   (561 GB/s) because a hand-written Triton decode materializes ~6 full
   u32/f32 intermediate tiles → register blowup; it needs
   `tl.inline_asm_elementwise` (atomic per-element-group, no intermediates)
   to work; (b) interleaved k-repack of B (pairs (j, j+128) adjacent) so a
   [BN,256] load + reshape (BN,128,2) + `tl.split` yields the two scale
   groups — A needs the same interleave (a tiny per-call permute kernel or
   an in-kernel gather via `tl.interleave` index vectors).
2. **Packed-pair decode (fp8-Marlin)** — still open, now with the exact
   design: `Out1 = (q & 0x80008000) | ((q & 0x7F007F00) >> 4)` etc. = 9 ops
   per 4 values (2.25/value) vs the current 5.5, bias foldable into b_s
   (exact power of two), NO weight repack needed (lane extraction in k-order
   works on the plain [N,K] buffer viewed as uint32). Requires multi-output
   `tl.inline_asm_elementwise` (untested on the fork's Triton — toy-kernel
   it first). Expected: the decode is 5.5 ops/value ≈ 57% of memory time at
   the current 910-ceiling; halving it is worth ~10-20% IF the kernel is
   decode-latency-exposed (unknown — the ablation was broken, see gotchas).
3. **Prefill rungs** (M>64: (128,128,32) inherited, unmeasured above M=64;
   M=2048 currently ~1.10–1.17× vs stock). The M>16 path still quantizes A
   (via `self.quant_fp8`) and runs the W8A8 ladder — a W8A16 prefill sweep
   (BM 32–128) may win the same way the decode path did.
4. **M 17–64 rungs** are inherited from the W8A16 ladder, never re-swept
   for the block-scale kernel (now they'd run W8A8+quant — sweep W8A16
   BM=32/64 there too).

## Spec-decode backend headroom (second workstream — items 1+2 DONE 2026-09-27)

**DONE: draft-vocab truncation** (+~20% e2e): HyperQwen's patch ported 4/4 to the fork's `qwen3_5_mtp.py` (`~/qwen3_5_mtp_patched.py`, installed by the launch script); the drafter scores 40,960 rows (419 MB) instead of 248,320 (2.54 GB). Artifacts in the model dir (`draft-head.safetensors`, `mtp_draft_vocab_ids.pt`, index entry; `.bak-draft` backup; delete the three to revert). `MTP_DRAFT_VOCAB=0` disables. Kit: `mtp-opt/`.
**DONE: probabilistic draft sampling** (+10-15%): zero port — the fork has `draft_sample_method: "probabilistic"` as a speculative-config option (default "greedy"); now set in the launch script.
**Combined measured (with n=5)**: 44.6–65.5 tok/s (avg ~57.5 vs ~47 at n=3 without) = **+22%**. Per-draft acceptance 47.7%, tokens/step ~3.4.

Remaining items:
3. ~~Drafter requant~~ **DEAD** — Qwen already ships the MTP linears FP8 in the official checkpoint (only `mtp.fc` 105 MB + norms bf16). Remaining opportunity ~0.2 ms/step.
4. ~~Sort-free sampler~~ **DONE 2026-09-27** — HyperQwen's patch applied cleanly (7 files, 0 rejects): sort-free top-k/top-p (≤64), multi-block row softmax (140→10 µs), draft top-k/top-p truncation. Two env vars added to `envs.py`. Measured: code 68.8, photosynthesis 66.3 (avg ~58.8).
5. **Verify-attention check** (unknown): confirm ROCM_AITER_FA actually serves the query_len=n+1 verify path at hd-256 (not falling back to Triton).
6. ~~FP8 KV cache via W8A16 decode~~ **DONE 2026-09-27**: 3-layer fix — (a) AITER assertion patched to accept `fn` alongside `fnuz`; (b) decode routed through Triton `unified_attention` (handles any dtype via `.to()`) instead of HIP `paged_attention_v1` (can't compile `fn` on gfx90a); (c) `rocm_aiter_fa.py` condition patched. Result: **601K KV tokens (2.29× concurrency at 262K)** at 41–61 tok/s (~10% slower than bf16 KV, 2× capacity). The speed gap is from Triton decode vs HIP decode — closing it requires porting the HIP kernel for `fn` on gfx90a (a C++ project).

Already in the fork: verify-ctx attention partitioning (davetha's biggest spec win), GDN spec bounds fix, `mamba_ssm_cache_dtype` (fp16 GDN state, ~1%).

## Gotchas (each cost us a cycle)

- **BLOCK_K > 128 = silent corruption** (scale-group violation). The test guard catches it — always run it.
- **GPU contention**: one server OR one test container, never both. Check `docker ps` first.
- **JIT compile storms**: first 2–3 requests after a fresh container compile Triton kernels (1–20 tok/s readings). Warm up before measuring.
- **The fork's device name is `AMD_Instinct_MI210`** (stock 0.30 uses `Instinct_MI210`) — config JSONs must use the fork's name (see `~/tuned-amd/`).
- **Compile-cache crash on kernel flip**: the torch_compile cache key doesn't include the kernel selection — the launch script always creates a FRESH container (never `docker restart` after swapping kernel files).
- **Short prompts (1 token: "Hi"/"Hello"/"Hey") produce garbage** — pre-existing, persists in eager mode (not CUDA-graphs), long prompts unaffected, accepted by the user for agentic use. The TRITON_ATTN bisection was never completed — open if it ever matters.
- **AITER linear must stay off for FP8** (`VLLM_ROCM_USE_AITER_LINEAR=0`): AITER's blockscale GEMM is gfx1250-Gluon (a port, not a tune). AITER attention (ROCM_AITER_FA) stays ON.
- **torch.compile + lazy imports = dynamo `find_spec` crash**: any `from x import y` INSIDE a method that dynamo traces raises `Unsupported: find_spec`. The W8A16 dispatch learned this the hard way — keep imports at module level.
- **torch.compile specializes shape-dependent Python branches at TRACE time**: the model is compiled once for the range (1, max_num_batched_tokens), traced at M=4096 — an `if M <= 16` in traced code bakes in the M>16 path for every M. Runtime shape dispatch must live inside an opaque custom op. (Cost a full deploy cycle to find; see the update banner.)
- **`print` in traced code = dynamo crash** (`Failed to trace builtin operator print`); prints inside custom-op impls are fine (they run eagerly) — that's the dispatch-logging pattern.
- **Quantize inside the compiled region only via `self.quant_fp8(...)`** (the registered `quant_fp8` CustomOp) or inside your own opaque op: a direct `per_token_group_quant_fp8()` call in traced Python hits `find_spec` (deep-gemm probe).
- **`test_decode_ablation.py` is BROKEN as a "1-op cast" probe** (measured -4 to -7% = ablated SLOWER): `uint8→bf16` on this Triton lowers to a ~13-op path (worse than the 5.5-op fp8 decode it replaces). A real ablation needs `tl.inline_asm_elementwise` with a single `v_mov_b32`. The negative result did prove the kernel is latency-bound, not instruction-issue-bound.
- **KV-cache dtype on this stack**: `auto` (bf16) is the only value that works end-to-end today. `fp8_e4m3` needs the pa_v1 HIP JIT build that fails in-container (clang rejects `-amdgpu-coerce-illegal-types=1`); explicit `float16`/`bfloat16` are rejected by AITER's `pa_v1` outright (only `auto` and fp8 forms are handled). The fp8-KV work is tracked in the spec-decode section item 6.
- **Long `sleep N` in a local tool call gets aborted** — run waits as background jobs (or remote loops) and poll with short commands.
- **lm_head FP8 was abandoned** (neutral speed; the code is kept: `quant-lmhead.py`, `patch-lmhead-fp8.sh` — needs the `get_quant_method` + embedding-loader patches + `weight_scale_inv` naming; +1.27 GB VRAM was the only reliable win).
- rsync/scp from WSL: use the SSH_ASKPASS env (sshpass is pty-blocked locally).

## Server config (for comparability)

`/models/qwen38-27b-fp8`, TP1, max-model-len 262144, max-seqs 8, batched-tokens 4096, gpu-util 0.93, `--kv-cache-dtype auto` (bf16 KV — see gotchas; the fp8-KV experiments are in spec-decode item 6), `--attention-backend ROCM_AITER_FA`, `--reasoning-parser qwen3`, MTP n=**5** probabilistic (`{"method":"mtp","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}`), `--enforce-eager` OFF. Port 8001 (maps to container 8000). Benchmark: `~/gemm-harness/bench-e2e.sh` (warmup x3 + the 6-prompt set + acceptance; the earlier `~/bench-mi210.sh` hardcodes model name `qwen38-27b`). **Only compare runs with the same KV dtype** — fp8 vs auto KV is worth ~10% by itself.
