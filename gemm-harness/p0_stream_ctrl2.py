#!/usr/bin/env python3
"""P0 part 2: why does the tiled [N,K] streaming pattern cap at ~910 GB/s when
the d2d copy does 1248? Test three variants on the gate_up shape:

  1D   -- each program reads one contiguous chunk (the copy-like ceiling)
  NK   -- current orientation: B[N, K] K-contiguous, [BN, BK] tiles
          (rows of BK bytes, stride K between rows) -- the 910 GB/s cap
  KN   -- transposed storage: B.t() [K, N] N-contiguous, [BK, BN] tiles
          (loads contiguous along N -- what the stock kernel does)

Also probes wide-BK loads (256/512 B rows) in NK orientation -- the control
has no scale groups, so it can test load widths the real kernel can't use
directly (a real kernel could still issue wide loads and split them into
128-wide scale tiles).
"""
import sys

import torch
import triton
import triton.language as tl

sys.path.insert(0, "/tmp")

torch.manual_seed(0)
DEV = "cuda"


@triton.jit
def _ctrl_1d(b_ptr, out_ptr, total, CHUNK: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    base = pid.to(tl.int64) * CHUNK
    acc = tl.zeros((BLOCK,), dtype=tl.int32)
    for off in range(0, CHUNK, BLOCK):
        idx = base + off + tl.arange(0, BLOCK)
        v = tl.load(b_ptr + idx, mask=idx < total, other=0)
        acc += v.to(tl.int32)
    tl.store(out_ptr + pid, tl.sum(acc, axis=0))


@triton.jit
def _ctrl_nk(b_ptr, out_ptr, N, K, stride_bn, stride_bk,
             BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_n = offs_n < N
    acc = tl.zeros((BLOCK_N,), dtype=tl.int32)
    b_ptrs = b_ptr + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk
    for _k in range(0, K, BLOCK_K):
        b = tl.load(b_ptrs, mask=mask_n[:, None], other=0)
        acc += tl.sum(b.to(tl.int32), axis=1)
        b_ptrs += BLOCK_K * stride_bk
    tl.store(out_ptr + offs_n, acc, mask=mask_n)


@triton.jit
def _ctrl_kn(b_ptr, out_ptr, N, K, stride_bk, stride_bn,
             BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr):
    # b is [K, N] with N contiguous (stride_bn == 1)
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_n = offs_n < N
    acc = tl.zeros((BLOCK_N,), dtype=tl.int32)
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    for _k in range(0, K, BLOCK_K):
        b = tl.load(b_ptrs, mask=mask_n[None, :], other=0)
        acc += tl.sum(b.to(tl.int32), axis=0)
        b_ptrs += BLOCK_K * stride_bk
    tl.store(out_ptr + offs_n, acc, mask=mask_n)


def bench(fn):
    fn()
    torch.cuda.synchronize()
    return triton.testing.do_bench(fn, warmup=10, rep=20)


def main():
    N, K = 34816, 5120
    B = torch.randint(0, 255, (N, K), device=DEV, dtype=torch.uint8)
    Bt = B.t().contiguous()  # [K, N], N-contiguous
    nbytes = N * K

    # ---- 1D contiguous ceiling ----
    best1d = None
    for chunk_kb in (256, 512, 1024, 2048):
        CHUNK = chunk_kb * 1024
        nprog = triton.cdiv(nbytes, CHUNK)
        out = torch.empty(nprog, device=DEV, dtype=torch.int32)
        for BLOCK in (1024, 2048, 4096):
            for w in (4, 8):
                try:
                    ms = bench(lambda: _ctrl_1d[(nprog,)](
                        B, out, nbytes, CHUNK=CHUNK, BLOCK=BLOCK,
                        num_warps=w, num_stages=2))
                except Exception:  # noqa: BLE001
                    continue
                gbs = nbytes / ms / 1e6
                if best1d is None or gbs > best1d[0]:
                    best1d = (gbs, chunk_kb, BLOCK, w, ms)
    print(f"1D contiguous: best {best1d[0]:.0f} GB/s "
          f"(chunk={best1d[1]}KB BLOCK={best1d[2]} {best1d[3]}w {best1d[4]:.3f}ms)")

    # ---- NK orientation (current) incl. wide BK ----
    print("\nNK orientation (current [N,K], K-contig rows):")
    rows = []
    for BN in (32, 64, 128):
        for BK in (64, 128, 256, 512):
            for w in (4, 8):
                for s in (2, 3):
                    if K % BK != 0:
                        continue
                    out = torch.empty(N, device=DEV, dtype=torch.int32)
                    try:
                        ms = bench(lambda: _ctrl_nk[(triton.cdiv(N, BN),)](
                            B, out, N, K, B.stride(0), B.stride(1),
                            BLOCK_N=BN, BLOCK_K=BK, num_warps=w, num_stages=s))
                    except Exception:  # noqa: BLE001
                        continue
                    rows.append((N * K / ms / 1e6, BN, BK, w, s, ms))
    rows.sort(reverse=True)
    for gbs, BN, BK, w, s, ms in rows[:6]:
        print(f"  {gbs:7.0f} GB/s  ({BN},{BK})@{w}w/{s}s  {ms:.3f} ms")

    # ---- KN orientation (transposed storage) ----
    print("\nKN orientation (transposed [K,N], N-contig loads):")
    rows = []
    for BN in (32, 64, 128, 256):
        for BK in (32, 64, 128):
            for w in (4, 8):
                for s in (2, 3):
                    if K % BK != 0:
                        continue
                    out = torch.empty(N, device=DEV, dtype=torch.int32)
                    try:
                        ms = bench(lambda: _ctrl_kn[(triton.cdiv(N, BN),)](
                            Bt, out, N, K, Bt.stride(0), Bt.stride(1),
                            BLOCK_K=BK, BLOCK_N=BN, num_warps=w, num_stages=s))
                    except Exception:  # noqa: BLE001
                        continue
                    rows.append((N * K / ms / 1e6, BN, BK, w, s, ms))
    rows.sort(reverse=True)
    for gbs, BN, BK, w, s, ms in rows[:6]:
        print(f"  {gbs:7.0f} GB/s  ({BN},{BK})@{w}w/{s}s  {ms:.3f} ms")

    print("\nDONE")


if __name__ == "__main__":
    main()
