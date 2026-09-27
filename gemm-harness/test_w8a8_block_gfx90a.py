#!/usr/bin/env python3
"""Verify + benchmark the W8A8/W8A16 block-scaled gfx90a kernels.

1. Correctness: every (shape, M) against a float32 dequant reference.
   - W8A8 path (A fp8 + As): the M > 16 (prefill) production path.
   - W8A16 path (A bf16, no As): the M <= 16 (decode) production path --
     reference is A_bf16 @ dequant(B).
2. Benchmark: vs the stock w8a8_triton_block_scaled_mm (bf16-cast-patched,
   tuned configs) -- the previous production path.

Run inside the v0.30 ROCm image with the patched fp8_utils.py and the new
kernel on the import path.
"""
import sys

import torch
import triton

sys.path.insert(0, "/tmp")

from triton_fp8_w8a8_block import (  # noqa: E402
    _w8a16_block_gemm,
    _w8a8_block_gemm,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (  # noqa: E402
    w8a8_triton_block_scaled_mm,
)

torch.manual_seed(0)
DEV = "cuda"
G_N, G_K = 128, 128

SHAPES = [
    (34816, 5120),   # mlp gate_up
    (16384, 5120),   # gdn in_proj
    (14336, 5120),   # gdn in_proj
    (5120, 17408),   # mlp down
    (5120, 6144),    # attn o
    (8192, 5120),    # attn qkv
]
M_GRID = [1, 4, 16, 2048]


def make(M, N, K):
    A = torch.randn(M, K, device=DEV, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    B = torch.randn(N, K, device=DEV, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    As = torch.rand(M, K // G_K, device=DEV, dtype=torch.float32) + 0.5
    Bs = torch.rand(N // G_N, K // G_K, device=DEV, dtype=torch.float32) + 0.5
    return A, B, As, Bs


def make_w8a16(M, N, K):
    A = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
    B = torch.randn(N, K, device=DEV, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    Bs = torch.rand(N // G_N, K // G_K, device=DEV, dtype=torch.float32) + 0.5
    return A, B, Bs


def reference(A, B, As, Bs, M, N, K):
    Af = A.to(torch.float32).reshape(M, K // G_K, G_K) * As[:, :, None]
    Af = Af.reshape(M, K)
    Bf = B.to(torch.float32).reshape(N // G_N, G_N, K // G_K, G_K) * Bs[:, None, :, None]
    Bf = Bf.reshape(N, K)
    return Af @ Bf.t()


def reference_w8a16(A, B, Bs, M, N, K):
    Bf = B.to(torch.float32).reshape(N // G_N, G_N, K // G_K, G_K) * Bs[:, None, :, None]
    Bf = Bf.reshape(N, K)
    return A.to(torch.float32) @ Bf.t()


def main():
    print("=== correctness: W8A8 path (prefill production path) ===")
    for (N, K) in SHAPES:
        for M in [1, 4, 16]:
            A, B, As, Bs = make(M, N, K)
            ref = reference(A, B, As, Bs, M, N, K)
            out = _w8a8_block_gemm(A, B, As, Bs, G_N, G_K, torch.bfloat16)
            rel = (out.float() - ref).abs().max().item() / ref.abs().max().item()
            status = "OK" if rel < 1e-2 else "FAIL"
            print(f"N={N} K={K} M={M}: rel err {rel:.2e} {status}")
            assert rel < 1e-2, f"NUMERICS FAIL at N={N} K={K} M={M}"

    print("\n=== correctness: W8A16 path (decode production path) ===")
    for (N, K) in SHAPES:
        for M in [1, 4, 6, 16]:
            A, B, Bs = make_w8a16(M, N, K)
            ref = reference_w8a16(A, B, Bs, M, N, K)
            out = _w8a16_block_gemm(A, B, Bs, G_N, G_K, torch.bfloat16)
            rel = (out.float() - ref).abs().max().item() / ref.abs().max().item()
            status = "OK" if rel < 1e-2 else "FAIL"
            print(f"N={N} K={K} M={M}: rel err {rel:.2e} {status}")
            assert rel < 1e-2, f"W8A16 NUMERICS FAIL at N={N} K={K} M={M}"

    print("\n=== benchmark: W8A16 (M<=16) / W8A8 (M>16) vs stock (tuned) ===")
    print(f"{'shape':>18} {'M':>5} {'new ms':>8} {'stock ms':>9} {'speedup':>8}")
    for (N, K) in SHAPES:
        for M in M_GRID:
            if M <= 16:
                A, B, Bs = make_w8a16(M, N, K)
                new_ms = triton.testing.do_bench(
                    lambda: _w8a16_block_gemm(
                        A, B, Bs, G_N, G_K, torch.bfloat16),
                    warmup=10, rep=20,
                )
                # stock comparison arm: quantize A the way production did
                Aq = A.to(torch.float8_e4m3fn)
                As = torch.rand(M, K // G_K, device=DEV, dtype=torch.float32) + 0.5
            else:
                A, B, As, Bs = make(M, N, K)
                new_ms = triton.testing.do_bench(
                    lambda: _w8a8_block_gemm(A, B, As, Bs, G_N, G_K, torch.bfloat16),
                    warmup=10, rep=20,
                )
            stock_ms = triton.testing.do_bench(
                lambda: w8a8_triton_block_scaled_mm(
                    Aq if M <= 16 else A, B, As if M <= 16 else As, Bs,
                    [G_N, G_K], torch.bfloat16,
                ),
                warmup=10, rep=20,
            )
            print(
                f"{'N=' + str(N) + ' K=' + str(K):>18} {M:>5} "
                f"{new_ms:>8.3f} {stock_ms:>9.3f} {stock_ms / new_ms:>7.2f}x"
            )
    print("DONE")


if __name__ == "__main__":
    main()
