#!/usr/bin/env python3
"""P10: wide-load W8A16 block-scale kernel (experiment).

Design (from the P0 findings):
  - B loads are [BLOCK_N, BLOCK_K] uint8 tiles with BLOCK_K in {256, 512}:
    256/512-B contiguous rows reach the ~1200 GB/s device ceiling, vs 910
    for the 128-B rows the BLOCK_K<=128 constraint forces on the old kernel.
  - A arrives as bf16 (W8A16): no A decode, no a_s (MI210 has no fp8 MFMA;
    the activation quant round-trip buys nothing on this hardware).
  - The b_s scale is folded into the decode's bias multiply: the fp8->f32
    bit placement yields value*2^-120, so multiplying by (b_s*2^120) gives
    the scaled value in f32; one RTZ pack to bf16 and the dot accumulates
    directly (tl.dot(a, b, acc)) -- zero per-tile accumulator math.
  - Decode is exact on all 254 finite e4m3fn codes incl. denormals (the
    bit pattern lands in f32 denormals exactly when fp8 e=0); the only new
    rounding is the RTZ f32->bf16 pack of the SCALED value (~2^-9 one-sided).

BLOCK_K is a multiple of group_k (128): each iteration spans BLOCK_K//128
scale groups; the per-element group index is (k_start + offs_k)//group_k.
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
def _w8a16_wide_kernel(
    a_ptr,  # [M, K] bf16 (raw activations)
    b_ptr,  # [N, K] uint8 (fp8 e4m3fn bytes)
    bs_ptr,  # [N // G_N, K // G_K] fp32, PRE-MULTIPLIED by 2**120
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
    BLOCK_K: tl.constexpr,  # load width; multiple of group_k
    EVEN_K: tl.constexpr,  # K % BLOCK_K == 0 -> skip k masks
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_n = offs_n < N

    # all rows of this tile share one n-scale-block (BLOCK_N <= group_n)
    bs_row = bs_ptr + (pid_n * BLOCK_N // group_n) * stride_bs_n

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk

    for k_start in range(0, K, BLOCK_K):
        if EVEN_K:
            a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
            b_u8 = tl.load(b_ptrs, mask=mask_n[:, None], other=0)
        else:
            k_rem = K - k_start
            a = tl.load(
                a_ptrs, mask=mask_m[:, None] & (offs_k[None, :] < k_rem),
                other=0.0,
            )
            b_u8 = tl.load(
                b_ptrs, mask=mask_n[:, None] & (offs_k[None, :] < k_rem),
                other=0,
            )

        # --- decode fp8 -> scaled bf16 (scale folded into the bias mul) ---
        u = b_u8.to(tl.uint32)
        fbits = ((u & 0x7F) << 20) | ((u & 0x80) << 24)
        f = fbits.to(tl.float32, bitcast=True)  # = value * 2^-120 (exact)
        # per-k-group scale vector (BLOCK_K//group_k distinct values)
        g = (k_start + offs_k) // group_k
        b_s = tl.load(bs_row + g * stride_bs_k)  # [BLOCK_K] fp32 (pre*2^120)
        f = f * b_s[None, :]  # = value * b_s
        b_dot = f.to(tl.bfloat16, fp_downcast_rounding="rtz")

        acc = tl.dot(a, tl.trans(b_dot), acc, out_dtype=tl.float32)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(c_ptr.type.element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_c = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, c, mask=mask_c)


def w8a16_wide_gemm(a, b_u8, bs, BM, BN, BK, warps, stages):
    """a: [M, K] bf16; b_u8: [N, K] uint8; bs: [N//128, K//128] fp32 (raw)."""
    M, K = a.shape
    N = b_u8.shape[0]
    assert BN <= G_N and G_N % BN == 0, "tile must sit inside one scale block"
    assert BK % G_K == 0
    bs_pre = bs * (2.0 ** 120)  # exact power-of-two fold
    c = torch.empty((M, N), dtype=torch.bfloat16, device=a.device)
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    _w8a16_wide_kernel[grid](
        a, b_u8, bs_pre, c, M, N, K, G_N, G_K,
        a.stride(0), a.stride(1), b_u8.stride(0), b_u8.stride(1),
        c.stride(0), c.stride(1), bs_pre.stride(0), bs_pre.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, EVEN_K=(K % BK == 0),
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
    # ---- correctness on all shapes, M in {1, 4} ----
    print("=== correctness (rel err vs fp32 dequant reference) ===")
    ok = True
    for name, N, K in SHAPES:
        for M in (1, 4):
            A, B, Bs = make(M, N, K)
            ref = reference(A, B, Bs, M, N, K)
            C = w8a16_wide_gemm(A, B.view(torch.uint8), Bs, 16, 32, 256, 8, 2)
            rel = (C.float() - ref).abs().max().item() / ref.abs().max().item()
            status = "OK " if rel < 1e-2 else "FAIL"
            if rel >= 1e-2:
                ok = False
            print(f"  {name:>8} M={M}: rel={rel:.2e} {status}")
    if not ok:
        print("CORRECTNESS FAILED -- aborting benchmark")
        return

    # ---- config sweep on the three heaviest shapes ----
    print("\n=== sweep (ms; best per shape) ===")
    cfgs = []
    for BN in (16, 32, 64):
        for BK in (256, 512):
            for w in (4, 8):
                for s in (2, 3):
                    cfgs.append((16, BN, BK, w, s))

    best = {}
    for name, N, K in SHAPES:
        A, B, Bs = make(1, N, K)
        rows = []
        for BM, BN, BK, w, s in cfgs:
            try:
                C = w8a16_wide_gemm(A, B.view(torch.uint8), Bs, BM, BN, BK, w, s)
                torch.cuda.synchronize()
                ms = triton.testing.do_bench(
                    lambda: w8a16_wide_gemm(
                        A, B.view(torch.uint8), Bs, BM, BN, BK, w, s),
                    warmup=10, rep=20,
                )
            except Exception:  # noqa: BLE001
                continue
            rows.append((ms, BM, BN, BK, w, s))
        rows.sort()
        for ms, BM, BN, BK, w, s in rows[:3]:
            gbs = N * K / ms / 1e6
            print(f"  {name:>8} ({BM},{BN},{BK})@{w}w/{s}s: {ms:.3f} ms  {gbs:.0f} GB/s")
        if rows:
            best[name] = rows[0]

    # ---- M=4 check with the best config ----
    print("\n=== M=4 with best configs ===")
    for name, N, K in SHAPES:
        if name not in best:
            continue
        _, BM, BN, BK, w, s = best[name]
        A, B, Bs = make(4, N, K)
        ms = triton.testing.do_bench(
            lambda: w8a16_wide_gemm(A, B.view(torch.uint8), Bs, BM, BN, BK, w, s),
            warmup=10, rep=20,
        )
        print(f"  {name:>8} M=4 ({BM},{BN},{BK})@{w}w/{s}s: {ms:.3f} ms")
    print("DONE")


if __name__ == "__main__":
    main()
