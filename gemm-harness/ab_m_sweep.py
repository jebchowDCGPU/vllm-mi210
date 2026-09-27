#!/usr/bin/env python3
"""A/B the W8A16 kernel vs the BASELINE W8A8 kernel at the REAL serving M
values (draft M=1, verify M=6, graph-padded M=8), on all model shapes PLUS
the MTP drafter fc shape (20480x5120) which was never swept.

Explains the e2e neutrality: the microbench wins were at M=1/4 only.
"""
import importlib.util
import sys

import torch
import triton

sys.path.insert(0, "/tmp")

torch.manual_seed(0)
DEV = "cuda"
G_N, G_K = 128, 128


def load_mod(name, path, kill_op_reg=False):
    src = open(path).read()
    if kill_op_reg:
        # avoid duplicate vllm::w8a8_block_gemm_gfx90a registration: the
        # primary module already registered it
        src = src.replace(
            "from vllm.utils.torch_utils import direct_register_custom_op",
            "direct_register_custom_op = None",
        )
    import tempfile
    tmp = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
    tmp.write(src)
    tmp.close()
    spec = importlib.util.spec_from_file_location(name, tmp.name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


new = load_mod("kern_new", "/tmp/triton_fp8_w8a8_block.py")
base = load_mod(
    "kern_base", "/tmp/triton_fp8_w8a8_block_baseline.py", kill_op_reg=True
)

SHAPES = [
    ("gate_up", 34816, 5120),
    ("gdn_16k", 16384, 5120),
    ("gdn_14k", 14336, 5120),
    ("down", 5120, 17408),
    ("o", 5120, 6144),
    ("qkv", 8192, 5120),
    ("mtp_fc", 20480, 5120),   # the drafter's fc -- never swept before
]
M_GRID = [1, 6, 8]


def make(M, N, K):
    A16 = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
    A8 = A16.to(torch.float8_e4m3fn)
    B = torch.randn(N, K, device=DEV, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    As = torch.rand(M, K // G_K, device=DEV, dtype=torch.float32) + 0.5
    Bs = torch.rand(N // G_N, K // G_K, device=DEV, dtype=torch.float32) + 0.5
    return A16, A8, B, As, Bs


def main():
    print(f"{'shape':>9} {'M':>2} {'base ms':>8} {'new ms':>8} {'delta':>7}")
    for name, N, K in SHAPES:
        for M in M_GRID:
            A16, A8, B, As, Bs = make(M, N, K)
            try:
                base_ms = triton.testing.do_bench(
                    lambda: base._w8a8_block_gemm(
                        A8, B, As, Bs, G_N, G_K, torch.bfloat16),
                    warmup=10, rep=20,
                )
            except Exception as e:  # noqa: BLE001
                print(f"{name:>9} {M:>2} base ERR {e}")
                continue
            try:
                new_ms = triton.testing.do_bench(
                    lambda: new._w8a16_block_gemm(
                        A16, B, Bs, G_N, G_K, torch.bfloat16),
                    warmup=10, rep=20,
                )
            except Exception as e:  # noqa: BLE001
                print(f"{name:>9} {M:>2} new ERR {e}")
                continue
            d = (base_ms - new_ms) / base_ms * 100
            print(f"{name:>9} {M:>2} {base_ms:>8.3f} {new_ms:>8.3f} {d:>+6.1f}%")
    print("DONE")


if __name__ == "__main__":
    main()
