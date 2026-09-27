# Kernel Benchmark — W8A16 Decode Path (final, 2026-09-27 evening)

**Server:** fp8-aiter, 262K ctx, `--kv-cache-dtype auto` (bf16 KV), MTP n=5 +
probabilistic + draft-vocab + sort-free sampler. Same config for ALL runs below.

## E2E Results — same-config A/B (the only valid comparison)

| prompt | baseline (503-line W8A8) | W8A16 v1 (broken dispatch) | **W8A16 v2 (fixed)** |
|---|---|---|---|
| transformer | 45.1 | 46.8 | **60.5** |
| photosynthesis | 62.4 | 63.0 | **77.3** |
| seasons | 53.9 | 53.5 | **71.6** |
| essay | 46.3 | 41.4 | **48.7** |
| code | 63.6 | 65.0 | **83.4** |
| CAP | 46.7 | 47.1 | **56.7** |
| **average** | **53.0** | **52.8** | **66.4 (+25.3%)** |

Acceptance: 45.7–46.8% in all three arms (the gain is pure speed, not
acceptance). All outputs coherent.

**IMPORTANT — how NOT to benchmark this:**
- The earlier "~58.8 baseline" number in the first version of this file was a
  HISTORICAL measurement from a different server config (32K ctx / 0.90 util,
  pre-fp8-KV era). It is not reproducible on the current config and must not
  be used as a comparison arm. Always re-run both kernels back-to-back.
- The "W8A16 v1" column looked like a tie with the baseline — that was a real
  bug (below), not noise.

## The bug the tie exposed (fixed 2026-09-27 evening)

The W8A16 kernel was **never executing** in the server. Dispatch logging
(one-shot prints inside the custom op) showed every call arriving as
`dtype=torch.float8_e4m3fn` from inductor-generated code. Root cause: the model
is compiled ONCE for the range (1, 4096) and traced at the profile shape
(M=4096) — the Python-level `if M <= 16` branch in `apply_block_scaled_mm`
was evaluated at TRACE time (taking the quant+W8A8 fallback) and specialized
away. The compiled graph baked in quant→W8A8 for EVERY M; the W8A16 path was
dead code at all sizes.

**Fix:** move the M-branch INSIDE the opaque custom op
(`_w8a16_block_gemm_dispatch`): the graph calls one op with bf16 A; the op
branches at runtime per call — including during CUDA-graph capture, where each
capture size launches the right kernels (verified: capture logs show
`M=12 → W8A16`, `M=4096 → quant+W8A8`).

## Kernel-level numbers (verify.sh, 42/42 checks pass)

| shape | baseline ms | W8A16 ms | delta |
|---|---|---|---|
| gate_up 34816×5120 | 0.274 | 0.217 | +21% |
| gdn 16384×5120 | 0.156 | 0.111 | +29% |
| gdn 14336×5120 | 0.155 | 0.101 | +35% |
| down 5120×17408 | 0.182 | 0.142 | +22% |
| o 5120×6144 | 0.075 | 0.061 | +19% |
| qkv 8192×5120 | 0.119 | 0.064 | +46% |
| mtp_fc 20480×5120 | 0.185 | 0.137 | +26% |

Holds at M ∈ {1, 6, 8} (ab_m_sweep.py). Config: (16,64,128)@4w/2s + split-K=4
everywhere; correctness rel err 1.9–3.2e-3 vs fp32 dequant references.

## How to reproduce

```bash
# W8A16 (current, active):
bash ~/start-fp8-aiter.sh && until curl -s localhost:8001/health; do sleep 5; done
cd ~/gemm-harness && ./bench-e2e.sh

# Baseline arm:
cp ~/w8a8-kernel/triton_fp8_w8a8_block.py ~/w8a8-kernel/triton_fp8_w8a8_block_NEW.py
cp ~/w8a8-kernel/triton_fp8_w8a8_block_BASELINE.py ~/w8a8-kernel/triton_fp8_w8a8_block.py
bash ~/start-fp8-aiter.sh && until curl -s localhost:8001/health; do sleep 5; done
cd ~/gemm-harness && ./bench-e2e.sh
# ...then restore: cp ~/w8a8-kernel/triton_fp8_w8a8_block_NEW.py ~/w8a8-kernel/triton_fp8_w8a8_block.py && bash ~/start-fp8-aiter.sh
```

## Files (on the box)

| file | md5 | what |
|---|---|---|
| `~/w8a8-kernel/triton_fp8_w8a8_block.py` | `e90115f5` | **W8A16 v2 (fixed) — active** |
| `~/w8a8-kernel/triton_fp8_w8a8_block_BASELINE.py` | `90a50017` | baseline (503-line W8A8) |
| `~/w8a8-kernel/triton_fp8_w8a8_block_NEW.py` | (older rev) | stale copy — safe to delete |
