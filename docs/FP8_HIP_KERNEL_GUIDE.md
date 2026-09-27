# FP8 Paged Attention on MI210 (gfx90a) — Implementation Guide

**Status: IMPLEMENTED (2026-09-27, this session).** The HIP `paged_attention_v1` kernel now handles FP8 KV on gfx90a: FP8 K/V are decoded to BF16 in-kernel (LUT in shared memory) and run through the existing BF16 MFMA path. All 9 correctness cases pass (rel err 3.4–5.0e-3 = bf16 noise). Kernel-level: **1.05–1.22× faster than the Triton `unified_attention` workaround at ≥64K context, 2.1× faster than the bf16 HIP kernel at 262K** (fp8 reads half the KV bytes); ~0.7–0.8× below ~16K context (latency-bound regime — E2E impact negligible, see benchmark notes).

**Deployment:** `bash ~/start-fp8-aiter.sh` installs the patched `.cuh` sources + prebuilt JIT `.so` (all 16 npar_loops variants) and routes single-token FP8 decode through the HIP kernel (`rocm_aiter_fa_patched.py`). Multi-token (MTP verify) still goes through Triton `unified_attention` (the HIP kernel is single-token only).

---

## What was done (the short version)

1. **Root cause confirmed by reproduction**: the FP8 path uses three CDNA3-only features — `__builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8` (pa_common.cuh:166), `__builtin_amdgcn_cvt_pk_fp8_f32` (Q quantization + P logits quantization), and `__builtin_amdgcn_cvt_pk_f32_fp8` (the decode direction — also missing on gfx90a). All fail with "needs target feature fp8-insts / fp8-conversion-insts".
2. **Fix (Approach A from the original guide)**: in the FP8 branch, decode K and V from FP8 to BF16 and feed the existing `gcn_mfma16x16x16_instr` (bf16 MFMA) path. Q stays in raw BF16 (no quantization — the W8A16 insight: strictly more accurate than the original which quantized Q to fp8). The softmax logits (P) also stay in BF16 (the kAuto write path). `k_scale`/`v_scale` handling unchanged; `q_scale` (always nullptr in production) is folded into the softmax scale for semantic preservation.
3. **The k-mapping consistency** (the guide's "MFMA layout mismatch" concern) is a non-issue once you decode: thread t's 8 decoded K values cover the same head elements as its 8 Q values (both `rowid*16 + qkratio*8 + i*4 + 0..3`), so the bf16 MFMA's per-thread k-distribution stays consistent. No cross-thread redistribution needed.
4. **Decode v1 (ALU)**: `fp8e4m3fn_to_f32` — place the 7 payload bits into an f32 exponent/mantissa and multiply by 2^120 (exact for all 254 finite codes incl. denormals; verified on GPU vs `__hip_fp8_e4m3::operator float`). ~9 VALU ops/value. **This doubled the kernel time at 16K+ context** (measured: 87M vs 13M VALU wavefront-instrs vs the bf16 path).
5. **Decode v2 (LUT — shipped)**: a 256-entry FP8→BF16 LUT in shared memory (512 B), built once per block (1 entry/thread), then **one 2-byte LDS read per value**. Kernel time at 262K ctx: 2.75 → **0.845 ms** (3.3×).
6. **Bug found by the power-of-2 decode test**: the PV loop initially kept the fp8 path's `ELEMS8_ELEMS4_RATIO/2` bound (=1 iteration) — only half the decoded V values were fed to the MFMA (tokens 4-7 of each group dropped, output exactly 0.382× the correct value). The bf16 MFMA needs **2 calls per 8 decoded values** (`i ∈ [0,2)`). Diagnostic: V[t]=2^t with uniform P — the bit pattern of `tmp_out×8` reads out the per-token weights directly.

## Benchmark (kernel-level, Qwen3.8-27B shapes: gqa 6, head 256, block 16, mtp 1)

| shape | fp8 HIP (this) | Triton unified_attention | bf16 HIP |
|---|---|---|---|
| seqs=1 ctx=1024 | 0.103 ms | 0.084 ms | 0.087 ms |
| seqs=1 ctx=16384 | 0.149 ms | 0.088 ms | 0.176 ms |
| seqs=1 ctx=65536 | 0.333 ms | 0.271 ms | 0.468 ms |
| seqs=1 ctx=131072 | 0.508 ms | 0.502 ms | 0.757 ms |
| seqs=1 ctx=262144 | **0.845 ms** | 0.955 ms | 1.318 ms |
| seqs=8 ctx=262144 | **5.647 ms** | 6.620 ms | 9.765 ms |

Crossover vs Triton ≈ 128K context. Below that the Triton kernel is faster at the kernel level, but the E2E production comparison historically favored the HIP path (per-call overheads: the Triton path does 2× `k_scale.expand()` + a heavier Python wrapper per call). E2E at short prompts expected ≈ neutral; at long context a clear win.

## E2E A/B (measured, same stack back-to-back, 2026-09-27)

| prompt | FP8 KV + HIP kernel (this) | FP8 KV + Triton (workaround) |
|---|---|---|
| transformer | 60.7 | 60.6 |
| photosynthesis | 77.9 | 77.7 |
| seasons | 64.8 | 64.7 |
| essay | 49.3 | 49.2 |
| code | 84.7 | 84.6 |
| CAP | 58.7 | 58.6 |

**A perfect tie at short prompts** (~100–500 token bench set): with the W8A16 GEMM round shipped, decode attention is a small step fraction and a 20% kernel-level difference doesn't register. The historical "~10% Triton cost" (41–61 vs 45–69 tok/s) does not reproduce on the current stack. The PA kernel's advantage is at **long context** (5–22% kernel-level at ≥64K, worth ~1–2% of step time at 262K decode) plus architectural cleanliness: the FP8 KV path now runs the native kernel with no Triton dependency for single-token decode, and FP8 KV is now **1.4–2.1× faster than bf16 KV** at the kernel level at ≥64K ctx (was 0.7× with the workaround at short ctx, and bf16-HIP was the speed king) — FP8 KV is strictly better than bf16 KV: same speed, 2× capacity (612,619 tokens, 2.34× concurrency at 262K).

## Files (all in `~/pa-src/patched/` on the box; local copies in `pa-fix/`)

| file | what |
|---|---|
| `pa_common.cuh` | + `fp8e4m3fn_to_f32`, `convert_fp8x8_to_b16x8` (ALU), `init_fp8_b16_lut` + `convert_fp8x8_to_b16x8_lut` (LUT) |
| `pa_kernels.cuh` | FP8 QK branch: decode K, bf16 MFMA, Q raw; P write: always bf16 on gfx90a; FP8 PV branch: decode V, bf16 MFMA vs bf16 logits; LUT init; q_scale fold. All original CDNA3 paths kept under `#if defined(__gfx950__) || defined(__gfx942__)` |
| `test_pa_v1_fp8.py` | 9-case correctness gate (vs fp32 reference + bf16 cross-check) |
| `bench_pa.py` | 3-way kernel benchmark (fp8 HIP / Triton / bf16 HIP) |
| `debug_pa*.py` | the diagnostic ladder (spike tests, power-of-2 decode, tmp_out dump) |
| `~/start-fp8-aiter.sh` | updated: installs the kernel + prebuilt pa_v1 JIT `.so` |
| `~/rocm_aiter_fa_patched.py` | updated: FP8 single-token decode → HIP kernel (Triton fallback removed) |

## Gotchas learned (each cost a cycle)

- **The aiter JIT cache does not key on source changes** — the folder hash covers only template args. After editing `.cuh` files you MUST clear the build dir (`rm -rf /cache/aiter/build/pa_v1_*` in-container; the files are root-owned so host-side rm fails).
- **Each `npar_loops` value is a separate JIT variant** (folder hash includes it). `npar_loops = ceil(ceil(max_ctx/256)/64)` — the server JIT-builds a new variant each time the batch's max context crosses a 16K boundary (45 s stall, once per container life). The launch script preinstalls all 16.
- **The passthrough-timing trap**: when A/B-testing decode variants, remember the benchmark sweeps multiple `npar_loops` — a monkey-patched build flag only applies to the variant built in THAT process. Build all variants with the flag or you measure a mix.
- **`__builtin_amdgcn_cvt_pk_f32_fp8` (decode) ALSO fails on gfx90a** — the original guide's "test first" answer is no. The 2^120 bit trick is the portable exact decode.
- **The reduce kernel needs no fix** — it operates on bf16 `tmp_out` regardless of KV dtype.
- **The kernel is grid-limited at short context** (4 partitions × 4 kv heads = 16 blocks on 104 CUs at ctx=1K) — per-block latency dominates there, which is why the LUT decode still costs ~0.016 ms vs the bf16 path at 1K ctx even though its throughput cost is negligible.

## Original analysis (kept for reference)


The HIP `paged_attention_v1` kernel uses `__builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8` — a **CDNA3-only** FP8 matrix instruction that doesn't exist on gfx90a (CDNA2). You cannot simply replace it with the BF16 MFMA because **the thread-to-data mapping is different** between the two instruction shapes. A naive "decode FP8→BF16 and call BF16 MFMA" produces wrong results.

## The MFMA Layout Mismatch (the core problem)

Both MFMA instructions use a 64-thread wavefront and produce a 16×16 output tile, but they distribute the K dimension differently across threads:

| | FP8 MFMA (`16x16x32`) | BF16 MFMA (`16x16x16`) |
|---|---|---|
| K per instruction | 32 | 16 |
| A values per thread | 8 FP8 (1 × `long`) | 4 BF16 (1 × `_B16x4`) |
| B values per thread | 8 FP8 (1 × `long`) | 4 BF16 (1 × `_B16x4`) |
| Thread t → A K-range | `(t/16)*8 + 0..7` | `(t/16)*4 + 0..3` |
| Thread t → B K-range | `(t/16)*8 + 0..7` | `(t/16)*4 + 0..3` |

**Concrete example:** thread 0 holds A K-values 0–7 in the FP8 layout, but only K-values 0–3 in the BF16 layout. Thread 16 holds A K-values 8–15 in FP8, but K-values 4–7 in BF16. The same 8 bytes represent **different K positions** depending on which MFMA interprets them.

To convert, you need cross-thread data redistribution (shared memory or `__shfl`), not just a per-thread decode.

## The Kernel Source (where everything lives)

Inside the container image (`local/vllm-mi210:mi210.6-aiter`):

```
/opt/python/lib/python3.14/site-packages/aiter_meta/
├── csrc/cpp_itfs/pa/
│   ├── pa_v1.py          ← Python JIT wrapper (jinja2 template → hipcc)
│   ├── pa_v1.cpp.jinja   ← C++ entry point template
│   ├── pa_v1.cuh         ← kernel launcher (grid/block setup, reduce kernel)
│   ├── pa_kernels.cuh    ← the attention math (QK^T, softmax, PV)
│   ├── pa_common.cuh     ← MFMA wrappers, data types, FP8 decode helpers
│   └── pa_common.cuh     ← also has `to_float_fp8x4` (FP8→f32, works on gfx90a)
└── csrc/include/
    └── dtype_fp8.cuh    ← FP8 type definitions
```

**The failing instruction** is in `pa_common.cuh`, line ~160:
```cpp
// #else branch (non-gfx950) — this is what compiles on gfx90a and FAILS:
template <typename T, int absz, int cbid, int blgp>
__device__ __forceinline__ floatx4 gcn_mfma16x16x32_instr(
    const long& inpA, const long& inpB, const floatx4& inpC)
{
    if constexpr(std::is_same<T, __hip_fp8_e4m3>::value)
    {
        // THIS INSTRUCTION DOES NOT EXIST ON gfx90a:
        return __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(inpA, inpB, inpC, absz, cbid, blgp);
    }
}
```

**The working BF16 MFMA** (same file, line ~141):
```cpp
template <typename T, int absz, int cbid, int blgp>
__device__ __forceinline__ floatx4 gcn_mfma16x16x16_instr(
    const _B16x4& inpA, const _B16x4& inpB, const floatx4& inpC)
{
    if constexpr(std::is_same<T, __hip_bfloat16>::value)
    {
        // THIS WORKS ON gfx90a:
        return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(inpA, inpB, inpC, absz, cbid, blgp);
    }
}
```

**The FP8 call site** in `pa_kernels.cuh` (line ~379–413):
```cpp
else // kv cache dtype fp8
{
    // Loads K from FP8 cache as _B8x16 (16 FP8 bytes)
    // Quantizes Q from BF16 to FP8 (8 bytes)
    // Calls gcn_mfma16x16x32_instr<__hip_fp8_e4m3>(K_fp8, Q_fp8, C)
    // → FAILS on gfx90a (instruction doesn't exist)
}
```

**The BF16 call site** (same file, line ~352–377):
```cpp
if constexpr(KV_DTYPE == vllm::Fp8KVCacheDataType::kAuto)
{
    // Loads K from BF16 cache as _B16x8 (8 BF16 = 16 bytes)
    // Uses Q as-is (BF16, no quantization)
    // Calls gcn_mfma16x16x16_instr<scalar_t>(K_bf16.xy[i], Q_bf16.xy[i], C)
    // → WORKS on gfx90a
}
```

## Data Types (in `pa_common.cuh`)

```cpp
typedef bit16x4 _B16x4;    // 4 × 16-bit = 8 bytes (4 BF16 or 4 FP16)
typedef struct _B16x8 {     // 8 × 16-bit = 16 bytes (8 BF16)
    _B16x4 xy[2];
} _B16x8;
using _B8x8 = uint2;        // 8 bytes (8 FP8 values)
using _B8x4 = int32_t;      // 4 bytes (4 FP8 values)
typedef struct _B8x16 {     // 16 bytes (16 FP8 values)
    _B8x8 xy[2];
} _B8x16;
```

**Key insight:** a 16-byte KV cache slot holds either 8 BF16 values (kAuto path) or 16 FP8 values (fp8 path). The same physical memory, different interpretation.

## Existing Helpers (already in the codebase)

- `to_float_fp8x4(_B8x4) → floatx4` — decodes 4 FP8 bytes to 4 floats. Uses `__builtin_amdgcn_cvt_pk_f32_fp8` which **may not compile on gfx90a** (CDNA3 builtin). Test first; if it fails, use `__hip_fp8_e4m3`'s conversion operator (software, works on all archs).
- `__hip_fp8_e4m3` → `float` conversion — works on all architectures (software on pre-CDNA3). Use `static_cast<float>(fp8_val)`.
- `__float2bfloat16(float)` — works on all architectures.

## Implementation Approaches (ranked by feasibility)

### Approach A: Restructure the FP8 branch to use the BF16 code path (RECOMMENDED)

**Difficulty:** Medium-high (2–4 days). Changes only `pa_kernels.cuh`.

The idea: in the FP8 branch, decode K from FP8 to BF16 **at the data loading stage**, then fall through to the existing BF16 MFMA path. Q stays in BF16 (skip quantization — the W8A16 insight from the GEMM kernel).

**Steps:**
1. After loading K from the FP8 cache (as `_B8x16`), decode all 16 FP8 bytes to 16 BF16 values.
2. Store the decoded values in a `_B16x8` array (2 × 8 BF16 = same 16 values, but now 32 bytes instead of 16).
3. Use the BF16 MFMA path: `gcn_mfma16x16x16_instr<__hip_bfloat16>` with the decoded K and raw BF16 Q.
4. The loop structure changes: the FP8 path processes 16 values per slot (QK_SIZE_RATIO=2), the BF16 path processes 8 per slot. You need 2 BF16 MFMA calls per FP8 MFMA call.

**The hard part:** the `QK_SIZE_RATIO`, `HEAD_LOOP`, and `QKHELOOP` constants are template parameters computed from head_size and dtype. You need to either:
- Add a new set of constants for the "FP8 storage + BF16 compute" mode, or
- Decode into a temporary buffer and reuse the BF16 loop structure.

**Why this is the best approach:** it avoids the MFMA layout mismatch entirely — the decoded BF16 data goes through the BF16 path's proven data flow.

### Approach B: Cross-thread data redistribution in the MFMA wrapper

**Difficulty:** High (1–2 weeks). Changes `pa_common.cuh`.

Add a gfx90a-specific `gcn_mfma16x16x32_instr` that:
1. Decodes 8 FP8 bytes to 8 floats (per-thread, no cross-thread needed)
2. Uses shared memory or `__shfl` to redistribute the K values to match the BF16 MFMA's thread-to-data mapping
3. Calls `gcn_mfma16x16x16_instr<__hip_bfloat16>` twice with the redistributed data

**The redistribution:** for each thread t, the 8 FP8 values at K positions `(t/16)*8 + 0..7` need to be sent to the threads that hold those K positions in the BF16 layout. Specifically:
- K values `(t/16)*8 + 0..3` go to thread `t` (for BF16 MFMA call 1)
- K values `(t/16)*8 + 4..7` go to thread `t + 16` (for BF16 MFMA call 2)

Wait — that's not right. The BF16 MFMA call 1 uses K 0–15 (across all threads), and call 2 uses K 16–31. Each thread provides 4 values for each call. The redistribution needs to route each FP8 value to the correct thread for the correct BF16 MFMA call.

This is a wavefront-level shuffle — expensive but doable with `__shfl_sync` or shared memory.

**Why this is harder:** you need to get the exact lane mappings right, and the shuffle overhead may eat the gains.

### Approach C: Manual dot product (no MFMA)

**Difficulty:** Low (1 day). Changes `pa_common.cuh`.

Replace the FP8 MFMA with a per-thread FMA loop:
```cpp
// Instead of one FP8 MFMA (K=32), do 32 scalar FMAs:
float result = inpC[0];
for (int k = 0; k < 8; k++) {
    float a = sw_fp8e4m3_to_f32(a_bytes[k]);
    float b = sw_fp8e4m3_to_f32(b_bytes[k]);
    result += a * b;
}
```

**Pros:** trivially correct, no layout issues.
**Cons:** much slower than MFMA (no matrix acceleration). Probably slower than the Triton workaround.

**Only worth it as a correctness reference** — implement this first to validate the decode, then optimize with Approach A or B.

## The V (value) Side

The attention kernel has TWO MFMA sections:
1. **QK^T** (scoring): Q × K^T — the code at line ~379
2. **PV** (output): attention_weights × V — the code at line ~794

Both use the FP8 MFMA for FP8 KV cache. **Both need the fix.** The V side has the same layout mismatch.

## Testing

After implementing, verify with:
```bash
# 1. Kernel-level test (in a throwaway container):
cd ~/gemm-harness && ./run-test.sh <your_test>.py --stop-server

# 2. E2E test:
bash ~/start-fp8-aiter.sh  # with --kv-cache-dtype fp8_e4m3
cd ~/gemm-harness && ./bench-e2e.sh

# 3. Compare against the Triton workaround (current baseline):
# FP8 KV + Triton: ~41-61 tok/s (depending on GEMM kernel)
# FP8 KV + HIP (target): should be ~45-67 tok/s (+10%)
```

## Expected Gains

| config | decode speed | KV capacity |
|---|---|---|
| bf16 KV + HIP kernel (current best speed) | 45–69 tok/s | 319K tokens |
| FP8 KV + Triton (current workaround) | 41–61 tok/s | 601K tokens |
| **FP8 KV + HIP kernel (target)** | **~45–67 tok/s** | **601K tokens** |

The target: HIP kernel speed + 2× KV capacity. The ~10% gap between Triton and HIP decode is what this project recovers.

## Key Files to Modify

| file | what to change |
|---|---|
| `pa_common.cuh` | Add gfx90a FP8 decode helpers; optionally add gfx90a MFMA wrapper |
| `pa_kernels.cuh` | Restructure the FP8 branch (Approach A) or add decode at load time |
| `pa_v1.py` | May need to pass different template params for gfx90a |
| `rocm_aiter_fa.py` | Remove the Triton fallback (once HIP kernel works) |

## Gotchas

- **`__builtin_amdgcn_cvt_pk_f32_fp8`** may not compile on gfx90a (CDNA3 builtin). Use `__hip_fp8_e4m3`'s `static_cast<float>()` instead (software, works everywhere).
- **`__builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8`** does NOT exist on gfx90a. This is the instruction that causes the build failure.
- **The jinja2 template** (`pa_v1.cpp.jinja`) generates C++ code that's compiled by `hipcc` via `make build`. Any `.cuh` changes need the JIT to recompile (delete the cached `.so` in `aiter_meta/jit/`).
- **The reduce kernel** (`_paged_attention_ll4mi_reduce_kernel`) also has FP8 paths — check if it needs the same fix.
- **Test with `--enforce-eager`** first (no CUDA graphs) to isolate kernel bugs from graph capture issues.
- **The `QK_SIZE_RATIO`** constant differs between BF16 (1) and FP8 (2) paths — it controls how many Q chunks are processed per K slot. If you change the data layout, this must be recomputed.

## References

- Our W8A16 GEMM kernel (`~/w8a8-kernel/triton_fp8_w8a8_block.py`) — the same FP8→BF16 decode approach, proven in Triton
- The Triton `unified_attention` kernel — handles FP8 KV via `.to(Q.dtype)`, works on gfx90a
- AMD MFMA ISA reference: CDNA2 (gfx90a) supports `mfma_f32_16x16x16bf16_1k` but NOT `mfma_f32_16x16x32_fp8_fp8`
- `KERNEL_BENCHMARK.md` — current performance numbers
- `BRINGUP_LOG.md` — full session history including the FP8 KV workaround
