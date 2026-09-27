#!/usr/bin/env python3
"""Stream-K experiment for the W8A8 block kernel.

Compares: production path (split-K wrapper) vs stream-K (persistent, atomic
fp32 accumulation) on the model's real shapes at decode Ms, with each shape's
sweep-best tile config. Correctness vs the fp32 dequant reference.
"""
import sys

import torch
import triton

sys.path.insert(0, "/tmp")

from triton_fp8_w8a8_block import (  # noqa: E402
    _streamk_gemm,
    _w8a8_block_gemm,
)

torch.manual_seed(0)
DEV = "cuda"
G_N, G_K = 128, 128

# (N, K, tile cfg, warps, stages) -- sweep winners per shape
SHAPES = [
    (34816, 5120, (16, 128, 64), 4, 3),   # gate_up
    (16384, 5120, (16, 32, 128), 2, 2),   # gdn
    (14336, 5120, (16, 32, 128), 2, 2),   # gdn
    (5120, 17408, (16, 64, 128), 4, 2),   # down
    (5120, 6144, (16, 64, 128), 4, 2),    # o
    (8192, 5120, (16, 128, 64), 4, 3),    # qkv
]
M_GRID = [1, 4, 16]
NUM_CUS = 104


def make(M, N, K):
    A = torch.randn(M, K, device=DEV, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    B = torch.randn(N, K, device=DEV, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    As = torch.rand(M, K // G_K, device=DEV, dtype=torch.float32) + 0.5
    Bs = torch.rand(N // G_N, K // G_K, device=DEV, dtype=torch.float32) + 0.5
    return A, B, As, Bs


def reference(A, B, As, Bs, M, N, K):
    Af = A.to(torch.float32).reshape(M, K // G_K, G_K) * As[:, :, None]
    Af = Af.reshape(M, K)
    Bf = B.to(torch.float32).reshape(N // G_N, G_N, K // G_K, G_K) * Bs[:, None, :, None]
    Bf = Bf.reshape(N, K)
    return Af @ Bf.t()


def main():
    print("=== correctness (stream-K, P=104) ===")
    for (N, K, cfg, w, s) in SHAPES:
        for M in [1, 4]:
            A, B, As, Bs = make(M, N, K)
            ref = reference(A, B, As, Bs, M, N, K)
            out = _streamk_gemm(
                A, B, As, Bs, G_N, G_K, torch.bfloat16,
                *cfg, NUM_CUS, w, s,
            )
            rel = (out.float() - ref).abs().max().item() / ref.abs().max().item()
            status = "OK" if rel < 1e-2 else "FAIL"
            print(f"N={N} K={K} M={M}: rel err {rel:.2e} {status}")
            assert rel < 1e-2, f"STREAM-K NUMERICS FAIL N={N} K={K} M={M}"

    print("\n=== benchmark: production (split-K) vs stream-K ===")
    print(f"{'shape':>16} {'M':>3} {'prod ms':>8} {'sk104':>7} {'sk208':>7} {'best':>7}")
    for (N, K, cfg, w, s) in SHAPES:
        for M in M_GRID:
            A, B, As, Bs = make(M, N, K)
            prod_ms = triton.testing.do_bench(
                lambda: _w8a8_block_gemm(A, B, As, Bs, G_N, G_K, torch.bfloat16),
                warmup=10, rep=20,
            )
            sk1 = triton.testing.do_bench(
                lambda: _streamk_gemm(
                    A, B, As, Bs, G_N, G_K, torch.bfloat16, *cfg, NUM_CUS, w, s
                ),
                warmup=10, rep=20,
            )
            sk2 = triton.testing.do_bench(
                lambda: _streamk_gemm(
                    A, B, As, Bs, G_N, G_K, torch.bfloat16, *cfg, 2 * NUM_CUS, w, s
                ),
                warmup=10, rep=20,
            )
            best = min(prod_ms, sk1, sk2)
            tag = "prod" if best == prod_ms else ("sk104" if best == sk1 else "sk208")
            print(
                f"{'N=' + str(N):>16} {M:>3} {prod_ms:>8.3f} {sk1:>7.3f} "
                f"{sk2:>7.3f} {tag:>7}"
            )
    print("DONE")


if __name__ == "__main__":
    main()
