#!/usr/bin/env python3
"""Decode-cost ablation: how much of the W8A8 kernel's time is the fp8->bf16
decode actually costing? Run BEFORE building the packed-pair decode.

Method: launch the kernel with DECODE_ABLATION=1, which replaces the 4.5-op
fp8e4nv->bf16 native cast with a 1-op uint8->bf16 cast (WRONG VALUES, valid
timing). The timing delta between the two modes = the exposed decode cost.
If the ablated kernel is <5% faster, the packed-pair decode (a 50% decode
reduction) is dead on arrival. If >=15%, build it.

Usage: python3 test_decode_ablation.py   (in the harness container)
"""
import sys

import torch
import triton
import triton.language as tl

sys.path.insert(0, "/tmp")

# Import the kernel module, then re-declare the kernel with the ablation flag
# by text-patching: simplest is a local copy of the kernel with the flag.
# To avoid drift, we patch the source at import time.
import importlib.util

src = open("/tmp/triton_fp8_w8a8_block.py").read()
# add the ablation constexpr to the kernel signature and the decode branch.
# Anchor on the END of the signature (SPLIT_K line + closing paren) -- robust
# against comment lines between the constexpr declarations.
assert src.count("    SPLIT_K: tl.constexpr,\n):") == 1, "signature anchor moved"
src = src.replace(
    "    SPLIT_K: tl.constexpr,\n):",
    "    SPLIT_K: tl.constexpr,\n    DECODE_ABLATION: tl.constexpr,\n):",
)
src = src.replace(
    "        a_dot = a_u8.to(tl.float8e4nv, bitcast=True).to(tl.bfloat16)\n"
    "        b_dot = b_u8.to(tl.float8e4nv, bitcast=True).to(tl.bfloat16)",
    "        if DECODE_ABLATION:\n"
    "            # WRONG VALUES, valid timing: 1-op cast instead of the\n"
    "            # 4.5-op fp8->bf16 decode. Measures the exposed decode cost.\n"
    "            a_dot = a_u8.to(tl.bfloat16)\n"
    "            b_dot = b_u8.to(tl.bfloat16)\n"
    "        else:\n"
    "            a_dot = a_u8.to(tl.float8e4nv, bitcast=True).to(tl.bfloat16)\n"
    "            b_dot = b_u8.to(tl.float8e4nv, bitcast=True).to(tl.bfloat16)",
)
# pass the flag through the wrapper's launches
src = src.replace(
    "            SPLIT_K=1,\n            **launch_opts,",
    "            SPLIT_K=1,\n            DECODE_ABLATION=ABLATION,\n"
    "            **launch_opts,",
)
src = src.replace(
    "        SPLIT_K=split_k,\n        **launch_opts,",
    "        SPLIT_K=split_k,\n        DECODE_ABLATION=ABLATION,\n"
    "        **launch_opts,",
)
src = src.replace(
    "import torch\n\nfrom vllm.platforms import current_platform",
    "import torch\n\nABLATION = False\n\n"
    "from vllm.platforms import current_platform",
)
# all patches must have applied (signature + branch + 2 launches)
assert src.count("DECODE_ABLATION") >= 4, (
    f"patch drift: only {src.count('DECODE_ABLATION')} DECODE_ABLATION sites"
)
open("/tmp/triton_fp8_w8a8_block_abl.py", "w").write(src)

spec = importlib.util.spec_from_file_location(
    "kernel_abl", "/tmp/triton_fp8_w8a8_block_abl.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
_w8a8_block_gemm = mod._w8a8_block_gemm

torch.manual_seed(0)
DEV = "cuda"
G_N, G_K = 128, 128

SHAPES = [
    (34816, 5120),   # gate_up  -- the packed-pair target (44% of GEMM time)
    (16384, 5120),   # gdn
    (5120, 17408),   # down
    (5120, 6144),    # o
]


def make(M, N, K):
    A = torch.randn(M, K, device=DEV, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    B = torch.randn(N, K, device=DEV, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    As = torch.rand(M, K // G_K, device=DEV, dtype=torch.float32) + 0.5
    Bs = torch.rand(N // G_N, K // G_K, device=DEV, dtype=torch.float32) + 0.5
    return A, B, As, Bs


def main():
    print(f"{'shape':>16} {'M':>3} {'real ms':>8} {'ablated ms':>10} "
          f"{'decode %':>8}  verdict")
    for (N, K) in SHAPES:
        for M in [1, 4]:
            A, B, As, Bs = make(M, N, K)

            mod.ABLATION = False
            real_ms = triton.testing.do_bench(
                lambda: _w8a8_block_gemm(A, B, As, Bs, G_N, G_K, torch.bfloat16),
                warmup=10, rep=20,
            )
            mod.ABLATION = True
            abl_ms = triton.testing.do_bench(
                lambda: _w8a8_block_gemm(A, B, As, Bs, G_N, G_K, torch.bfloat16),
                warmup=10, rep=20,
            )
            exposed = (real_ms - abl_ms) / real_ms * 100
            verdict = (
                "packed-pair DEAD (<5% exposed)"
                if exposed < 5
                else "MAYBE (5-15%)"
                if exposed < 15
                else "BUILD IT (>=15% exposed)"
            )
            print(
                f"{'N=' + str(N):>16} {M:>3} {real_ms:>8.3f} {abl_ms:>10.3f} "
                f"{exposed:>7.1f}%  {verdict}"
            )
    print("DONE")


if __name__ == "__main__":
    main()
