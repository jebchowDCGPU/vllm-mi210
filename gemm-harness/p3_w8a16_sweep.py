#!/usr/bin/env python3
"""P3: W8A16 variant of the CURRENT kernel structure + config sweep.

Keeps everything that works in the current kernel (black-box fp8->bf16 decode
at 5.5 ops/value, [N,K] tiles + tl.trans, per-tile accumulator scale) and
changes:
  - A arrives as bf16 (W8A16): no A decode (~23% of loop instructions), no
    a_s loads, no A LDS staging, and the accumulator path drops to a single
    broadcast multiply (acc += dot * b_s[None, :]).
  - Sweeps BLOCK_K=128 configs (the control kernel showed 128-B rows give
    883 GB/s vs 714 for the 64-B rows of the current gate_up rung -- the
    original 2-shape coarse sweep may have missed this).

Benchmarks the current W8A8 kernel (from /tmp) at its ladder configs as the
baseline in the same run.
"""
import sys

import torch
import triton
import triton.language as tl

sys.path.insert(0, "/tmp")

torch.manual_seed(0)
DEV = "cuda"
G_N, G_K = 128, 128


@triton.jit
def _w8a16_block_kernel(
    a_ptr,  # [M, K] bf16 (raw activations)
    b_ptr,  # [N, K] uint8
    bs_ptr,  # [N // G_N, K // G_K] fp32
    c_ptr,  # [M, N] bf16
    M,
    N,
    K,
    group_n,
    group_k,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    stride_bs_n,
    stride_bs_k,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_n = offs_n < N

    offs_bsn = offs_n // group_n
    bs_ptrs = bs_ptr + offs_bsn * stride_bs_n

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk

    for k_start in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)  # bf16
        b_u8 = tl.load(b_ptrs, mask=mask_n[:, None], other=0)
        b_dot = b_u8.to(tl.float8e4nv, bitcast=True).to(tl.bfloat16)

        offs_ks = k_start // group_k
        b_s = tl.load(bs_ptrs + offs_ks * stride_bs_k, mask=mask_n, other=0.0)
        accumulator += (
            tl.dot(a, tl.trans(b_dot), out_dtype=tl.float32) * b_s[None, :]
        )
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = accumulator.to(c_ptr.type.element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_c = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, c, mask=mask_c)


def w8a16_gemm(a, b_u8, bs, BM, BN, BK, warps, stages):
    M, K = a.shape
    N = b_u8.shape[0]
    c = torch.empty((M, N), dtype=torch.bfloat16, device=a.device)
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    _w8a16_block_kernel[grid](
        a, b_u8, bs, c, M, N, K, G_N, G_K,
        a.stride(0), a.stride(1), b_u8.stride(0), b_u8.stride(1),
        c.stride(0), c.stride(1), bs.stride(0), bs.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
        num_warps=warps, num_stages=stages,
    )
    return c


SHAPES = [
    ("gate_up", 34816, 5120),
    ("gdn_16k", 16384, 5120),
    ("gdn_14k", 14336, 5120),
    ("down", 5120, 17408),
    ("o", 5120, 6144),
    ("qkv", 8192, 5120),
]

# current-kernel ladder rungs (M<=16) for the baseline
CURRENT = {
    "gate_up": (16, 128, 64, 4, 3),
    "gdn_16k": (16, 32, 128, 2, 2),
    "gdn_14k": (16, 32, 128, 2, 2),
    "down": (16, 64, 128, 4, 2),  # + split4 in production
    "o": (16, 64, 128, 4, 2),  # + split2 in production
    "qkv": (16, 128, 64, 4, 3),
}


def make_w8a16(M, N, K):
    A = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
    B = torch.randn(N, K, device=DEV, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    Bs = torch.rand(N // G_N, K // G_K, device=DEV, dtype=torch.float32) + 0.5
    return A, B, Bs


def make_w8a8(M, N, K):
    A = torch.randn(M, K, device=DEV, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    B = torch.randn(N, K, device=DEV, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    As = torch.rand(M, K // G_K, device=DEV, dtype=torch.float32) + 0.5
    Bs = torch.rand(N // G_N, K // G_K, device=DEV, dtype=torch.float32) + 0.5
    return A, B, As, Bs


def reference(A, B, Bs, M, N, K):
    Bf = B.to(torch.float32).reshape(N // G_N, G_N, K // G_K, G_K) * Bs[:, None, :, None]
    Bf = Bf.reshape(N, K)
    return A.to(torch.float32) @ Bf.t()


def main():
    from triton_fp8_w8a8_block import _w8a8_block_gemm  # noqa: PLC0415

    # ---- correctness of the W8A16 variant ----
    print("=== W8A16 correctness (rel err vs fp32 dequant reference) ===")
    ok = True
    for name, N, K in SHAPES:
        for M in (1, 4):
            A, B, Bs = make_w8a16(M, N, K)
            ref = reference(A, B, Bs, M, N, K)
            C = w8a16_gemm(A, B.view(torch.uint8), Bs, 16, 64, 128, 4, 2)
            rel = (C.float() - ref).abs().max().item() / ref.abs().max().item()
            if rel >= 1e-2:
                ok = False
                print(f"  {name:>8} M={M}: rel={rel:.2e} FAIL")
    print("  all OK" if ok else "  FAILED")

    # ---- baseline: current W8A8 kernel at its ladder configs ----
    print("\n=== baseline: current W8A8 kernel (ladder cfgs) ===")
    base = {}
    for name, N, K in SHAPES:
        A8, B8, As8, Bs8 = make_w8a8(1, N, K)
        ms = triton.testing.do_bench(
            lambda: _w8a8_block_gemm(A8, B8, As8, Bs8, G_N, G_K, torch.bfloat16),
            warmup=10, rep=20,
        )
        base[name] = ms
        print(f"  {name:>8}: {ms:.3f} ms  {N * K / ms / 1e6:.0f} GB/s")

    # ---- W8A16 sweep ----
    print("\n=== W8A16 sweep (top 4 per shape) ===")
    cfgs = []
    for BN in (32, 64, 128, 256):
        for BK in (64, 128):
            for w in (4, 8):
                for s in (2, 3):
                    cfgs.append((16, BN, BK, w, s))

    best = {}
    for name, N, K in SHAPES:
        A, B, Bs = make_w8a16(1, N, K)
        rows = []
        for BM, BN, BK, w, s in cfgs:
            if N % BN != 0:
                continue
            try:
                C = w8a16_gemm(A, B.view(torch.uint8), Bs, BM, BN, BK, w, s)
                torch.cuda.synchronize()
                ms = triton.testing.do_bench(
                    lambda: w8a16_gemm(A, B.view(torch.uint8), Bs, BM, BN, BK, w, s),
                    warmup=10, rep=20,
                )
            except Exception:  # noqa: BLE001
                continue
            rows.append((ms, BM, BN, BK, w, s))
        rows.sort()
        for ms, BM, BN, BK, w, s in rows[:4]:
            gbs = N * K / ms / 1e6
            d = (base[name] - ms) / base[name] * 100
            print(f"  {name:>8} ({BM},{BN},{BK})@{w}w/{s}s: {ms:.3f} ms  "
                  f"{gbs:.0f} GB/s  {d:+.0f}% vs current")
        if rows:
            best[name] = rows[0]

    # ---- M=4 with best configs ----
    print("\n=== M=4 with best configs ===")
    for name, N, K in SHAPES:
        if name not in best:
            continue
        _, BM, BN, BK, w, s = best[name]
        A, B, Bs = make_w8a16(4, N, K)
        ms = triton.testing.do_bench(
            lambda: w8a16_gemm(A, B.view(torch.uint8), Bs, BM, BN, BK, w, s),
            warmup=10, rep=20,
        )
        print(f"  {name:>8} M=4 ({BM},{BN},{BK})@{w}w/{s}s: {ms:.3f} ms")
    print("DONE")


if __name__ == "__main__":
    main()
