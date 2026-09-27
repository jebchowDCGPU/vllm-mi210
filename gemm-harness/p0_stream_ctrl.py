#!/usr/bin/env python3
"""P0: streaming control kernel -- the achievable-BW ceiling for the W8A8
kernel's B access pattern (K-contiguous [N,K] uint8, [BLOCK_N, BLOCK_K]
tiles, same grid shape, software-pipelined K loop), with a trivial int
consume instead of decode+dot.

Interpretation:
  - If the best control config hits ~1.3 TB/s, the real kernel's gap to peak
    is compute/pipeline (decode, trans, accumulator) -> P1/P2/P4 are the
    levers.
  - If the control also does ~650 GB/s, the access pattern itself is the
    limit -> only tiling/MLP changes (P4) can help.

Also prints two device-level references: torch uint8 sum (pure read) and
d2d copy (read+write).
"""
import sys

import torch
import triton
import triton.language as tl

sys.path.insert(0, "/tmp")

torch.manual_seed(0)
DEV = "cuda"


@triton.jit
def _stream_ctrl(
    b_ptr,  # [N, K] uint8
    out_ptr,  # [N] int32
    N,
    K,
    stride_bn,
    stride_bk,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
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


SHAPES = [
    ("gate_up", 34816, 5120),
    ("gdn_16k", 16384, 5120),
    ("gdn_14k", 14336, 5120),
    ("down", 5120, 17408),
    ("o", 5120, 6144),
    ("qkv", 8192, 5120),
]

# current ladder rungs, for direct comparison
CURRENT = {
    "gate_up": (128, 64, 4, 3),
    "gdn_16k": (32, 128, 2, 2),
    "gdn_14k": (32, 128, 2, 2),
    "down": (64, 128, 4, 2),
    "o": (64, 128, 4, 2),
    "qkv": (128, 64, 4, 3),
}


def bench_ctrl(B, out, N, K, BN, BK, w, s):
    grid = (triton.cdiv(N, BN),)

    def launch():
        _stream_ctrl[grid](
            B, out, N, K, B.stride(0), B.stride(1),
            BLOCK_N=BN, BLOCK_K=BK, num_warps=w, num_stages=s,
        )

    launch()
    torch.cuda.synchronize()
    return triton.testing.do_bench(launch, warmup=10, rep=20)


def main():
    print(f"{'shape':>10} {'cfg (BN,BK,w,s)':>18} {'ms':>8} {'GB/s':>7}")
    results = {}
    for name, N, K in SHAPES:
        B = torch.randint(0, 255, (N, K), device=DEV, dtype=torch.uint8)
        out = torch.empty(N, device=DEV, dtype=torch.int32)
        best = None
        rows = []
        # current rung first (marked), then the sweep
        cfgs = [CURRENT[name] + (True,)]
        for BN in (32, 64, 128, 256):
            for BK in (64, 128):
                for w in (2, 4, 8):
                    for s in (2, 3, 4, 5):
                        cfgs.append((BN, BK, w, s, False))
        for BN, BK, w, s, is_cur in cfgs:
            if N % BN != 0 or K % BK != 0:
                continue
            try:
                ms = bench_ctrl(B, out, N, K, BN, BK, w, s)
            except Exception:  # noqa: BLE001
                continue
            gbs = N * K / ms / 1e6
            tag = " *CUR*" if is_cur else ""
            rows.append((ms, gbs, BN, BK, w, s, tag))
            if best is None or ms < best[0]:
                best = (ms, gbs, BN, BK, w, s)
        rows.sort()
        for ms, gbs, BN, BK, w, s, tag in rows[:6]:
            print(f"{name:>10} ({BN},{BK},{w}w,{s}s){tag:>7} {ms:>8.3f} {gbs:>7.0f}")
        results[name] = {"best": best, "current_rung": rows[-1] if rows else None}
        # find the current rung's row
        for r in rows:
            if r[6]:
                print(f"{name:>10} CURRENT RUNG: {r[0]:.3f} ms  {r[1]:.0f} GB/s")

    # device-level references on the gate_up-sized buffer
    B = torch.randint(0, 255, (34816, 5120), device=DEV, dtype=torch.uint8)
    B2 = torch.empty_like(B)
    torch.cuda.synchronize()
    ms = triton.testing.do_bench(lambda: B.sum(dtype=torch.int64), warmup=10, rep=20)
    print(f"\nreference torch uint8 sum (pure read): {ms:.3f} ms  "
          f"{B.numel() / ms / 1e6:.0f} GB/s")
    ms = triton.testing.do_bench(lambda: B2.copy_(B), warmup=10, rep=20)
    print(f"reference d2d copy (read+write): {ms:.3f} ms  "
          f"{2 * B.numel() / ms / 1e6:.0f} GB/s")
    print("DONE")


if __name__ == "__main__":
    main()
