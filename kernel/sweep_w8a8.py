#!/usr/bin/env python3
"""Config sweep for the W8A8 block kernel (direct kernel launches, explicit
configs). Two-phase: coarse on the two problem shapes (gate_up: 41% of GEMM
time at 590 GB/s; down: 363 GB/s), refine top-8 across all shapes.
Winners verified against the fp32 dequant reference.
"""
import json
import os
import sys

import torch
import triton

sys.path.insert(0, "/tmp")

from triton_fp8_w8a8_block import _w8a8_block_gemm_kernel  # noqa: E402

torch.manual_seed(0)
OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/sweep-out"
os.makedirs(OUT, exist_ok=True)
DEV = "cuda"
G_N, G_K = 128, 128

SHAPES = [
    (34816, 5120),   # gate_up  -- 590 GB/s currently, biggest lever
    (16384, 5120),   # gdn
    (14336, 5120),   # gdn
    (5120, 17408),   # down     -- 363 GB/s, K-heavy
    (5120, 6144),    # o
    (8192, 5120),    # qkv (probably)
]
M_GRID = [1, 4, 16, 2048]


def make(M, N, K):
    A = torch.randn(M, K, device=DEV, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    B = torch.randn(N, K, device=DEV, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    As = torch.rand(M, K // G_K, device=DEV, dtype=torch.float32) + 0.5
    Bs = torch.rand(N // G_N, K // G_K, device=DEV, dtype=torch.float32) + 0.5
    C = torch.empty(M, N, device=DEV, dtype=torch.bfloat16)
    return A, B, As, Bs, C


def launch(A, B, As, Bs, C, M, N, K, cfg):
    grid = (triton.cdiv(M, cfg["BM"]), triton.cdiv(N, cfg["BN"]))
    _w8a8_block_gemm_kernel[grid](
        A.view(torch.uint8), B.view(torch.uint8), As, Bs, C,
        M, N, K, G_N, G_K,
        A.stride(0), A.stride(1), B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        As.stride(0), As.stride(1), Bs.stride(0), Bs.stride(1),
        BLOCK_M=cfg["BM"], BLOCK_N=cfg["BN"], BLOCK_K=cfg["BK"],
        num_warps=cfg["w"], num_stages=cfg["s"],
    )


def reference(A, B, As, Bs, M, N, K):
    Af = A.to(torch.float32).reshape(M, K // G_K, G_K) * As[:, :, None]
    Af = Af.reshape(M, K)
    Bf = B.to(torch.float32).reshape(N // G_N, G_N, K // G_K, G_K) * Bs[:, None, :, None]
    Bf = Bf.reshape(N, K)
    return Af @ Bf.t()


def bench(A, B, As, Bs, C, M, N, K, cfg):
    try:
        launch(A, B, As, Bs, C, M, N, K, cfg)
        torch.cuda.synchronize()
        return triton.testing.do_bench(
            lambda: launch(A, B, As, Bs, C, M, N, K, cfg), warmup=10, rep=20
        )
    except Exception:
        return float("inf")


def grid():
    for bm in (16, 32):
        for bn in (32, 64, 128):
            for bk in (32, 64, 128):
                for w in (2, 4, 8):
                    for s in (2, 3, 4):
                        yield {"BM": bm, "BN": bn, "BK": bk, "w": w, "s": s}


def main():
    # ---- phase 1: coarse on gate_up + down at M=1 ----
    scores = {}
    for (N, K) in [(34816, 5120), (5120, 17408)]:
        A, B, As, Bs, C = make(1, N, K)
        for cfg in grid():
            key = tuple(sorted(cfg.items()))
            ms = bench(A, B, As, Bs, C, 1, N, K, cfg)
            scores[key] = scores.get(key, 0.0) + ms
        print(f"coarse done for N={N}", flush=True)
    ranked = sorted(scores.items(), key=lambda kv: kv[1])
    top = [dict(k) for k, _ in ranked[:10]]
    print("=== top coarse (sum ms over 2 shapes) ===")
    for k, v in ranked[:10]:
        print(f"{v:.4f}  {dict(k)}")

    # ---- phase 2: refine across all shapes and Ms, verify winners ----
    results = {}
    for (N, K) in SHAPES:
        per_m = {}
        for M in M_GRID:
            A, B, As, Bs, C = make(M, N, K)
            best, best_ms = None, float("inf")
            for cfg in top:
                ms = bench(A, B, As, Bs, C, M, N, K, cfg)
                if ms < best_ms:
                    best, best_ms = cfg, ms
            if M <= 16:  # verify winner
                ref = reference(A, B, As, Bs, M, N, K)
                launch(A, B, As, Bs, C, M, N, K, best)
                rel = (C.float() - ref).abs().max().item() / ref.abs().max().item()
                assert rel < 1e-2, f"winner wrong: {best} rel={rel}"
            gbs = (N * K) / best_ms / 1e6
            per_m[M] = {"cfg": best, "ms": best_ms, "GBps": round(gbs)}
            print(f"N={N} K={K} M={M}: {best_ms:.3f} ms  {gbs:.0f} GB/s  {best}", flush=True)
        results[f"{N}x{K}"] = per_m
    with open(os.path.join(OUT, "w8a8-sweep.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("DONE")


if __name__ == "__main__":
    main()
