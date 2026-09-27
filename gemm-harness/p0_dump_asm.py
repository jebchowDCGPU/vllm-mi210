#!/usr/bin/env python3
"""P0 diagnostics: dump the generated GCN asm for the W8A8 kernel's ladder
configs + instruction histogram + register usage. Answers:

  1. What does the fp8->bf16 decode actually lower to on this Triton build?
     (op count per element -- the 4.5-op vs ~17-op question)
  2. Does tl.trans(b_dot) cost an LDS round-trip (ds_write/ds_read around
     the dot)?
  3. Load widths (dwordx4?) and how many stages are actually prefetched.
  4. Register pressure (n_regs / n_spills -- can 2 programs co-reside/CU?)

Prints everything to stdout (the container is removed after the run, so
stdout is the only way out). Full asm is printed between markers; the
histogram + excerpts give the quick read.
"""
import re
import sys
from collections import Counter

import torch
import triton

sys.path.insert(0, "/tmp")

from triton_fp8_w8a8_block import _w8a8_block_gemm_kernel  # noqa: E402

torch.manual_seed(0)
DEV = "cuda"
G_N, G_K = 128, 128

# (name, N, K, BM, BN, BK, warps, stages) -- the current ladder rungs
CONFIGS = [
    ("gate_up", 34816, 5120, 16, 128, 64, 4, 3),
    ("gdn_16k", 16384, 5120, 16, 32, 128, 2, 2),
    ("gdn_14k", 14336, 5120, 16, 32, 128, 2, 2),
    ("down", 5120, 17408, 16, 64, 128, 4, 2),
    ("o", 5120, 6144, 16, 64, 128, 4, 2),
    ("qkv", 8192, 5120, 16, 128, 64, 4, 3),
]


def make(M, N, K):
    A = torch.randn(M, K, device=DEV, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    B = torch.randn(N, K, device=DEV, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    As = torch.rand(M, K // G_K, device=DEV, dtype=torch.float32) + 0.5
    Bs = torch.rand(N // G_N, K // G_K, device=DEV, dtype=torch.float32) + 0.5
    return A, B, As, Bs


def opcode_hist(asm_text):
    hist = Counter()
    for line in asm_text.splitlines():
        line = line.strip()
        if not line or line.startswith("//") or line.startswith(";"):
            continue
        m = re.match(r"^[A-Za-z0-9_$.]+:\s*(.*)$", line)
        body = m.group(1) if m else line
        body = body.strip()
        if not body:
            continue
        op = body.split(None, 1)[0]
        if op.split(".")[0] in (
            "v", "s", "ds", "global", "buffer", "image", "flat", "exp", "ds_gws",
        ) or op.startswith(("v_", "s_", "ds_", "global_", "buffer_", "flat_")):
            hist[op] += 1
    return hist


def main():
    for name, N, K, BM, BN, BK, w, s in CONFIGS:
        A, B, As, Bs = make(1, N, K)
        C = torch.empty(1, N, device=DEV, dtype=torch.bfloat16)
        grid = (triton.cdiv(1, BM), triton.cdiv(N, BN))
        h = _w8a8_block_gemm_kernel[grid](
            A.view(torch.uint8), B.view(torch.uint8), As, Bs, C,
            1, N, K, G_N, G_K,
            A.stride(0), A.stride(1), B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
            As.stride(0), As.stride(1), Bs.stride(0), Bs.stride(1),
            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, SPLIT_K=1,
            num_warps=w, num_stages=s,
        )
        torch.cuda.synchronize()

        print(f"\n{'=' * 70}\n### {name}: N={N} K={K} cfg=({BM},{BN},{BK})@{w}w/{s}s")
        try:
            md = dict(h.metadata) if hasattr(h, "metadata") else {}
            print(f"metadata: {md}")
        except Exception as e:  # noqa: BLE001
            print(f"metadata: ERR {e}")
        for attr in ("n_regs", "n_spills"):
            print(f"{attr}: {getattr(h, attr, 'N/A')}")

        asm = None
        if hasattr(h, "asm"):
            keys = list(h.asm.keys())
            print(f"asm keys: {keys}")
            for k in ("amdgcn", "ptx", "llir"):
                if k in h.asm and h.asm[k]:
                    asm = h.asm[k]
                    print(f"using asm['{k}'] ({len(asm.splitlines())} lines)")
                    break
        if asm is None:
            print("!! no asm available")
            continue

        # vgpr count from the LLVM metadata comments
        for pat in (r"NumVgprs:\s*(\d+)", r"; VGPRs:\s*(\d+)", r"vgprs:\s*(\d+)"):
            m = re.search(pat, asm)
            if m:
                print(f"vgpr count (regex {pat!r}): {m.group(1)}")
                break

        hist = opcode_hist(asm)
        total = sum(hist.values())
        print(f"\n-- instruction histogram ({total} instrs) --")
        for op, c in hist.most_common(40):
            print(f"  {c:6d}  {op}")

        # decode-relevant ops (fp8->bf16 path)
        decode_ops = {
            op: c for op, c in hist.items()
            if op.startswith(("v_cvt", "v_and", "v_or", "v_lshl", "v_lshr",
                              "v_pack", "v_perm", "v_mul_f32", "v_fma"))
        }
        print(f"\n-- decode-suspect ops --\n  {decode_ops}")

        # trans check: LDS traffic
        lds = {op: c for op, c in hist.items() if op.startswith("ds_")}
        print(f"-- LDS ops (tl.trans round-trip?) --\n  {lds}")

        # load widths
        loads = {op: c for op, c in hist.items()
                 if op.startswith(("global_load", "buffer_load", "flat_load"))}
        print(f"-- global loads --\n  {loads}")

        # mfma
        mfma = {op: c for op, c in hist.items() if op.startswith("v_mfma")}
        print(f"-- MFMA --\n  {mfma}")

        # print the full asm between markers for offline analysis
        print(f"\n-----BEGIN ASM {name}-----")
        print(asm)
        print(f"-----END ASM {name}-----")

    print("\nDONE")


if __name__ == "__main__":
    main()
