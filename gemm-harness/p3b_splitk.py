#!/usr/bin/env python3
"""P3b: W8A16 kernel + split-K sweep (P5 merged).

The W8A16 variant won +13-27% direct (p3_w8a16_sweep.py). Now add the
split-K machinery (fp32 partials + reduce, same as the W8A8 kernel) and
sweep split_k for the small-N shapes where wave quantization dominates:

  down  5120x17408: 80 tiles at BN=64 = 0.77 waves  -> split 4/8
  o     5120x6144:  96 tiles at BN=64 ... wait, N=5120 -> 80 tiles
  qkv   8192x5120:  128 tiles at BN=64 = 1.23 waves -> split 4/8
  gdn_14k 14336:    448 tiles at BN=32 = 4.3 waves  -> split 2

K % (split_k * 128) == 0 for all: 17408=17*1024, 6144=6*1024, 5120=5*1024.
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
    a_ptr,  # [M, K] bf16
    b_ptr,  # [N, K] uint8
    bs_ptr,  # [N // G_N, K // G_K] fp32
    c_ptr,  # [M, N] bf16 (SPLIT_K=1) or fp32 partials [SPLIT_K, M, N]
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
    SPLIT_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2) if SPLIT_K > 1 else 0

    if SPLIT_K > 1:
        k_base = pid_k * (K // SPLIT_K)
        k_end = k_base + (K // SPLIT_K)
    else:
        k_base = 0
        k_end = K

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_n = offs_n < N

    offs_bsn = offs_n // group_n
    bs_ptrs = bs_ptr + offs_bsn * stride_bs_n

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + (k_base + offs_k)[None, :] * stride_ak
    b_ptrs = b_ptr + offs_n[:, None] * stride_bn + (k_base + offs_k)[None, :] * stride_bk

    for k_start in range(k_base, k_end, BLOCK_K):
        a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
        b_u8 = tl.load(b_ptrs, mask=mask_n[:, None], other=0)
        b_dot = b_u8.to(tl.float8e4nv, bitcast=True).to(tl.bfloat16)

        offs_ks = k_start // group_k
        b_s = tl.load(bs_ptrs + offs_ks * stride_bs_k, mask=mask_n, other=0.0)
        accumulator += (
            tl.dot(a, tl.trans(b_dot), out_dtype=tl.float32) * b_s[None, :]
        )
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if SPLIT_K > 1:
        c = accumulator
        c_ptrs = c_ptr + pid_k.to(tl.int64) * M * N \
            + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    else:
        c = accumulator.to(c_ptr.type.element_ty)
        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_c = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, c, mask=mask_c)


@triton.jit
def _splitk_reduce(
    partials_ptr, c_ptr, M, N,
    SPLIT_K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for s in range(SPLIT_K):
        p = partials_ptr + s.to(tl.int64) * M * N \
            + offs_m[:, None] * N + offs_n[None, :]
        acc += tl.load(p, mask=mask, other=0.0)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :],
             acc.to(c_ptr.type.element_ty), mask=mask)


def w8a16_gemm(a, b_u8, bs, BM, BN, BK, warps, stages, split_k=1):
    M, K = a.shape
    N = b_u8.shape[0]
    if split_k > 1:
        assert K % (split_k * G_K) == 0
    c = torch.empty((M, N), dtype=torch.bfloat16, device=a.device)
    if split_k == 1:
        grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
        _w8a16_block_kernel[grid](
            a, b_u8, bs, c, M, N, K, G_N, G_K,
            a.stride(0), a.stride(1), b_u8.stride(0), b_u8.stride(1),
            c.stride(0), c.stride(1), bs.stride(0), bs.stride(1),
            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, SPLIT_K=1,
            num_warps=warps, num_stages=stages,
        )
        return c
    partials = torch.empty((split_k, M, N), dtype=torch.float32, device=a.device)
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN), split_k)
    _w8a16_block_kernel[grid](
        a, b_u8, bs, partials, M, N, K, G_N, G_K,
        a.stride(0), a.stride(1), b_u8.stride(0), b_u8.stride(1),
        partials.stride(1), partials.stride(2),
        bs.stride(0), bs.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, SPLIT_K=split_k,
        num_warps=warps, num_stages=stages,
    )
    _splitk_reduce[(triton.cdiv(M, 16), triton.cdiv(N, 128))](
        partials, c, M, N, SPLIT_K=split_k, BLOCK_M=16, BLOCK_N=128,
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

# (BM, BN, BK, w, s) candidates per shape -- direct winners + neighbors
CANDS = {
    "gate_up": [(16, 128, 128, 8, 2), (16, 64, 128, 4, 2), (16, 128, 128, 4, 2)],
    "gdn_16k": [(16, 64, 64, 4, 2), (16, 64, 128, 4, 2), (16, 32, 128, 2, 2)],
    "gdn_14k": [(16, 64, 64, 4, 2), (16, 64, 128, 4, 2), (16, 32, 128, 2, 2)],
    "down": [(16, 64, 128, 4, 2), (16, 32, 128, 4, 2), (16, 64, 128, 4, 3),
             (16, 128, 128, 4, 2)],
    "o": [(16, 64, 128, 4, 2), (16, 32, 128, 4, 2), (16, 128, 128, 4, 2)],
    "qkv": [(16, 64, 128, 4, 3), (16, 64, 128, 4, 2), (16, 32, 128, 4, 2),
            (16, 128, 128, 4, 3)],
}

BASE = {  # from p3 run: current W8A8 kernel (production ladder, ms)
    "gate_up": 0.274, "gdn_16k": 0.156, "gdn_14k": 0.155,
    "down": 0.182, "o": 0.074, "qkv": 0.119,
}


def make(M, N, K):
    A = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
    B = torch.randn(N, K, device=DEV, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    Bs = torch.rand(N // G_N, K // G_K, device=DEV, dtype=torch.float32) + 0.5
    return A, B, Bs


def reference(A, B, Bs, M, N, K):
    Bf = B.to(torch.float32).reshape(N // G_N, G_N, K // G_K, G_K) * Bs[:, None, :, None]
    Bf = Bf.reshape(N, K)
    return A.to(torch.float32) @ Bf.t()


def main():
    # correctness with split-K on the shapes that will use it
    print("=== split-K correctness ===")
    for name, N, K in SHAPES:
        for sk in (2, 4, 8):
            if K % (sk * G_K) != 0:
                continue
            A, B, Bs = make(1, N, K)
            ref = reference(A, B, Bs, 1, N, K)
            C = w8a16_gemm(A, B.view(torch.uint8), Bs, 16, 64, 128, 4, 2, split_k=sk)
            rel = (C.float() - ref).abs().max().item() / ref.abs().max().item()
            print(f"  {name:>8} split{sk}: rel={rel:.2e} {'OK' if rel < 1e-2 else 'FAIL'}")

    print("\n=== split-K sweep (top 4 per shape) ===")
    for name, N, K in SHAPES:
        A, B, Bs = make(1, N, K)
        rows = []
        for BM, BN, BK, w, s in CANDS[name]:
            for sk in (1, 2, 4, 8):
                if K % (sk * G_K) != 0 or N % BN != 0:
                    continue
                try:
                    C = w8a16_gemm(A, B.view(torch.uint8), Bs, BM, BN, BK, w, s, sk)
                    torch.cuda.synchronize()
                    ms = triton.testing.do_bench(
                        lambda: w8a16_gemm(
                            A, B.view(torch.uint8), Bs, BM, BN, BK, w, s, sk),
                        warmup=10, rep=20,
                    )
                except Exception:  # noqa: BLE001
                    continue
                rows.append((ms, BM, BN, BK, w, s, sk))
        rows.sort()
        for ms, BM, BN, BK, w, s, sk in rows[:4]:
            gbs = N * K / ms / 1e6
            d = (BASE[name] - ms) / BASE[name] * 100
            print(f"  {name:>8} ({BM},{BN},{BK})@{w}w/{s}s sk{sk}: {ms:.3f} ms  "
                  f"{gbs:.0f} GB/s  {d:+.0f}% vs current")
    print("DONE")


if __name__ == "__main__":
    main()
