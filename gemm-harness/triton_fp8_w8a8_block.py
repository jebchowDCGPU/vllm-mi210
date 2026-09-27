# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Triton W8A8 block-scaled FP8 GEMM for ROCm gfx90a (MI210).

Extends davetha's TritonW8A16Fp8LinearKernel (vllm fork
rocm-wfp8a16-gfx90a-v0.28.0rc2, triton_fp8_w8a16.py) from per-channel W8A16 to
the official Qwen3.8-27B-FP8 scheme: fp8 e4m3fn weights with 128x128 block
scales AND fp8 e4m3fn activations with per-token-group (128) scales.

What is inherited from the W8A16 kernel (measured facts, see its 750-line
header):
  - bf16-native decode as the default (fp8e4nv -> bf16 cast, ~4.5 ops, exact on
    all 254 finite codes incl. denormals; the fp16 bit-trick never won because
    the kernel is weight-streaming-bound, not VALU-bound)
  - the (M, N)-keyed tile ladder tuned on this exact 104-CU card
  - the 2D (m, n) grid; weights read through explicit strides, K-contiguous
    view left alone (coalesces at BLOCK_K > BLOCK_N rungs)

What changes for W8A8:
  - A arrives fp8-quantized (per-token-group dynamic scheme, quantized by
    Fp8LinearMethod before apply_block_scaled_mm) -> decoded bf16-native
    in-loop like B
  - scales move from the epilogue into the K loop: a_s is per (row, k-group),
    b_s per (128-col, k-group) block. BLOCK_K <= 128 (one scale group per
    tile) is a CORRECTNESS constraint of this scheme -- the ladder's 32/64
    values satisfy it.
  - the accumulator is scaled per K-tile: acc += dot(a,b) * a_s[:,None] *
    b_s[None,:], the same pattern as the stock _w8a8_triton_block_scaled_mm.

Numerics guard: this file ships with a standalone verification harness
(test_w8a8_block_gfx90a.py) that checks every winner against a float32
dequant reference on the model's real shapes.
"""

from collections.abc import Sequence

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
try:  # import location differs across vLLM versions
    from vllm.utils.torch_utils import direct_register_custom_op
except ImportError:  # pragma: no cover
    try:
        from vllm.utils import direct_register_custom_op
    except ImportError:
        direct_register_custom_op = None

try:  # importable both inside the vllm package and standalone (for testing)
    from .BlockScaledMMLinearKernel import Fp8BlockScaledMMLinearKernel

    _IN_VLLM_PACKAGE = True
except ImportError:
    _IN_VLLM_PACKAGE = False

# module-level import (used INSIDE the custom op only -- never traced):
# the M-branch must live inside the opaque custom op, because the model is
# compiled ONCE for the range (1, max_num_batched_tokens) and traced at the
# profile shape (M=4096) -- a Python-level `if M <= 16` in
# apply_block_scaled_mm gets specialized away at trace time and the compiled
# graph bakes in the M>16 (quant + W8A8) path for EVERY M.
try:
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8,
    )
except Exception:  # noqa: BLE001  # standalone (test) import
    per_token_group_quant_fp8 = None


@triton.jit
def _w8a8_block_gemm_kernel(
    # Pointers
    a_ptr,  # [M, K]  uint8 (raw float8_e4m3fn bytes, quantized activations)
    b_ptr,  # [N, K]  uint8 (raw float8_e4m3fn bytes, weights)
    as_ptr,  # [M, K // G_K]  fp32 per-token-group activation scales
    bs_ptr,  # [N // G_N, K // G_K]  fp32 per-block weight scales
    c_ptr,  # [M, N]  bf16/fp16 output (SPLIT_K=1) or fp32 partials
    # Dimensions
    M,
    N,
    K,
    # Quant group sizes (G_N x G_K blocks for B; G_K groups for A)
    group_n,
    group_k,
    # Strides
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    stride_as_m,
    stride_as_k,
    stride_bs_n,
    stride_bs_k,
    # Tile sizes -- gfx90a ladder from the W8A16 kernel
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    # Split-K factor: each program handles 1/SPLIT_K of the K range.
    # SPLIT_K=1 stores bf16 directly; SPLIT_K>1 stores fp32 partials
    # (layout [SPLIT_K, M, N], c strides are per-partial) for the reducer.
    SPLIT_K: tl.constexpr,
):
    """
    C[M, N] = (dequant(A)[M, K] * a_s) @ (dequant(B)[N, K] * b_s)^T

    Both operands are raw e4m3fn bytes decoded to bf16 in-loop (native cast,
    exact on all 254 finite codes). Scales are applied per K-tile in the fp32
    accumulator; BLOCK_K <= group_k keeps one scale group per tile.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2) if SPLIT_K > 1 else 0

    # K range for this split. Each split's base is 128-aligned (the wrapper
    # guarantees K % (SPLIT_K * group_k) == 0), so scale-group indexing stays
    # exact.
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

    as_ptrs = as_ptr + offs_m * stride_as_m
    offs_bsn = offs_n // group_n
    bs_ptrs = bs_ptr + offs_bsn * stride_bs_n

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + (k_base + offs_k)[None, :] * stride_ak
    b_ptrs = b_ptr + offs_n[:, None] * stride_bn + (k_base + offs_k)[None, :] * stride_bk

    for k_start in range(k_base, k_end, BLOCK_K):
        k_rem = k_end - k_start

        mask_a = mask_m[:, None] & (offs_k[None, :] < k_rem)
        a_u8 = tl.load(a_ptrs, mask=mask_a, other=0)
        mask_b = mask_n[:, None] & (offs_k[None, :] < k_rem)
        b_u8 = tl.load(b_ptrs, mask=mask_b, other=0)

        a_dot = a_u8.to(tl.float8e4nv, bitcast=True).to(tl.bfloat16)
        b_dot = b_u8.to(tl.float8e4nv, bitcast=True).to(tl.bfloat16)

        offs_ks = k_start // group_k
        a_s = tl.load(as_ptrs + offs_ks * stride_as_k, mask=mask_m, other=0.0)
        b_s = tl.load(bs_ptrs + offs_ks * stride_bs_k, mask=mask_n, other=0.0)
        # Outer product of the tile scales once, then a single accumulator
        # multiply per element (halves the accumulator-path work vs two).
        s_tile = a_s[:, None] * b_s[None, :]

        accumulator += tl.dot(a_dot, tl.trans(b_dot), out_dtype=tl.float32) * s_tile

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if SPLIT_K > 1:
        # fp32 partials at [pid_k, M, N]
        c = accumulator
        c_ptrs = c_ptr + pid_k.to(tl.int64) * M * N \
            + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    else:
        c = accumulator.to(c_ptr.type.element_ty)
        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_c = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, c, mask=mask_c)


@triton.jit
def _w8a8_block_splitk_reduce(
    partials_ptr,  # [SPLIT_K, M, N] fp32
    c_ptr,  # [M, N] bf16/fp16
    M,
    N,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
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


# ---------------------------------------------------------------------------
# W8A16 path (M <= 16): A arrives as bf16 (apply_input_quant=False on the
# kernel class -- the MI210 has no fp8 MFMA, so the activation quant
# round-trip buys nothing on this hardware). Measured 2026-09-27
# (p3_w8a16_sweep.py + p3b_splitk.py): the single config (16,64,128)@4w/2s
# with split-K=4 beats the W8A8 ladder on every model shape:
#   gate_up +21%, gdn_16k +28%, gdn_14k +35%, down +22%, o +18%, qkv +46%
# (split-K=4 fills the 104 CUs; e.g. qkv 128 tiles x4 = 512 programs).
# The accumulator path is a single broadcast multiply (no a_s outer
# product); the A decode (~23% of the old loop's instructions) is gone.
# ---------------------------------------------------------------------------


@triton.jit
def _w8a16_block_gemm_kernel(
    a_ptr,  # [M, K] bf16 (raw activations)
    b_ptr,  # [N, K] uint8 (raw float8_e4m3fn bytes)
    bs_ptr,  # [N // G_N, K // G_K] fp32 per-block weight scales
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
    a_ptrs = a_ptr + offs_m[:, None] * stride_am \
        + (k_base + offs_k)[None, :] * stride_ak
    b_ptrs = b_ptr + offs_n[:, None] * stride_bn \
        + (k_base + offs_k)[None, :] * stride_bk

    for k_start in range(k_base, k_end, BLOCK_K):
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

    if SPLIT_K > 1:
        c = accumulator
        c_ptrs = c_ptr + pid_k.to(tl.int64) * M * N \
            + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    else:
        c = accumulator.to(c_ptr.type.element_ty)
        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_c = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, c, mask=mask_c)


def _w8a16_block_gemm(
    a: torch.Tensor,  # [M, K] bf16 (raw activations)
    b: torch.Tensor,  # [N, K] float8_e4m3fn
    b_scales: torch.Tensor,  # [N // group_n, K // group_k] fp32
    group_n: int,
    group_k: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """W8A16 block-scaled GEMM for M <= 16 (decode/spec-decode path).

    Sweep-derived single config: (16, 64, 128)@4w/2s + split-K=4 (falls back
    to 2/1 when K is not divisible). BLOCK_N=64 keeps every tile inside one
    128-wide weight-scale block.
    """
    M, K = a.shape
    N = b.shape[0]
    assert group_k >= 1 and group_n >= 1

    b_u8 = b.view(torch.uint8)

    BLOCK_M, BLOCK_N, BLOCK_K = 16, 64, 128
    num_warps, num_stages = 4, 2
    split_k = 4 if K % (4 * group_k) == 0 else (
        2 if K % (2 * group_k) == 0 else 1
    )
    if N % BLOCK_N != 0:
        # odd N: fall back to the caller (W8A8 path handles any N)
        raise ValueError(f"W8A16 path needs N % {BLOCK_N} == 0, got N={N}")

    c = torch.empty((M, N), dtype=out_dtype, device=a.device)

    if split_k == 1:
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        _w8a16_block_gemm_kernel[grid](
            a, b_u8, b_scales, c,
            M, N, K, group_n, group_k,
            a.stride(0), a.stride(1), b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            b_scales.stride(0), b_scales.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            SPLIT_K=1,
            num_warps=num_warps, num_stages=num_stages,
        )
        return c

    partials = torch.empty(
        (split_k, M, N), dtype=torch.float32, device=a.device
    )
    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(N, BLOCK_N),
        split_k,
    )
    _w8a16_block_gemm_kernel[grid](
        a, b_u8, b_scales, partials,
        M, N, K, group_n, group_k,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        partials.stride(1), partials.stride(2),  # per-partial [M, N]
        b_scales.stride(0), b_scales.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        SPLIT_K=split_k,
        num_warps=num_warps, num_stages=num_stages,
    )
    _w8a8_block_splitk_reduce[(triton.cdiv(M, 16), triton.cdiv(N, 128))](
        partials, c, M, N, SPLIT_K=split_k, BLOCK_M=16, BLOCK_N=128,
    )
    return c


def _w8a16_block_gemm_fake(a, b, b_scales, group_n, group_k, out_dtype):
    M, K = a.shape
    N = b.shape[0]
    return torch.empty((M, N), dtype=out_dtype, device=a.device)


def _w8a16_block_gemm_dispatch(
    a: torch.Tensor,  # [M, K] bf16 (raw activations)
    b: torch.Tensor,  # [N, K] float8_e4m3fn
    b_scales: torch.Tensor,  # [N // group_n, K // group_k] fp32
    group_n: int,
    group_k: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Custom-op impl: runtime M dispatch (opaque to torch.compile).

    The M-branch MUST live inside the custom op: the model is compiled once
    for the range (1, max_num_batched_tokens) and traced at the profile
    shape (M=4096), so a Python-level `if M <= 16` in apply_block_scaled_mm
    is specialized away at trace time and the compiled graph bakes in the
    M>16 (quant + W8A8) path for EVERY M -- the W8A16 kernel then never
    runs (this exact bug shipped and cost the whole e2e gain; found via
    dispatch logging 2026-09-27). Inside the op the branch runs eagerly
    per call, including during CUDA-graph capture (each capture size
    launches the right kernels).
    """
    M, K = a.shape
    N = b.shape[0]
    if (
        M <= 16
        and K % group_k == 0
        and N % 64 == 0
        and per_token_group_quant_fp8 is not None
    ):
        return _w8a16_block_gemm(a, b, b_scales, group_n, group_k, out_dtype)
    # M > 16 (prefill) or odd shape: quantize here, run the W8A8 ladder
    a_q, a_scales = per_token_group_quant_fp8(a, group_k)
    return _w8a8_block_gemm(
        a_q, b, a_scales, b_scales, group_n, group_k, out_dtype
    )


def _w8a8_block_gemm(
    a: torch.Tensor,  # [M, K] float8_e4m3fn (quantized activations)
    b: torch.Tensor,  # [N, K] float8_e4m3fn (weights)
    a_scales: torch.Tensor,  # [M, K // group_k] fp32
    b_scales: torch.Tensor,  # [N // group_n, K // group_k] fp32
    group_n: int,
    group_k: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Host wrapper: gfx90a (M, N)-keyed tile ladder, sweep-derived, with
    split-K for the small-N K-heavy shapes."""
    assert group_k >= 1 and group_n >= 1
    M, K = a.shape
    N = b.shape[0]

    a_u8 = a.view(torch.uint8)
    b_u8 = b.view(torch.uint8)

    # (M, N)-keyed ladder. Direct-path rungs from the dedicated sweep
    # (sweep_w8a8.py); split-K for the small-N K-heavy shapes. Stream-K was
    # measured (test_streamk.py, 2026-09-27): wins on down (0.182->0.172) and
    # qkv (0.124->0.084) but loses on big-N; net e2e gain small -- REVERTED to
    # split-K per user decision (stream-K code retained above, unused).
    num_stages = None
    split_k = 1
    if M <= 16:
        if N >= 20000:
            BLOCK_M, BLOCK_N, BLOCK_K = 16, 128, 64
            num_warps, num_stages = 4, 3
        elif N >= 12000:
            BLOCK_M, BLOCK_N, BLOCK_K = 16, 32, 128
            num_warps, num_stages = 2, 2
        elif N >= 8192:
            # qkv band: the wide tile (sweep-measured 0.110 vs 0.124)
            BLOCK_M, BLOCK_N, BLOCK_K = 16, 128, 64
            num_warps, num_stages = 4, 3
        else:
            # down / o band: split-K fills the 104 CUs (80 programs at BN=64)
            BLOCK_M, BLOCK_N, BLOCK_K = 16, 64, 128
            num_warps, num_stages = 4, 2
            if K % (4 * group_k) == 0:
                split_k = 4
            elif K % (2 * group_k) == 0:
                split_k = 2
    elif M <= 32:
        if N >= 20000:
            BLOCK_M, BLOCK_N, BLOCK_K = 16, 128, 64
            num_warps, num_stages = 4, 3
        elif N >= 8192:
            BLOCK_M, BLOCK_N, BLOCK_K = 32, 32, 64
            num_warps = 2
        else:
            BLOCK_M, BLOCK_N, BLOCK_K = 16, 64, 128
            num_warps, num_stages = 4, 2
    elif M <= 64:
        if N >= 8192:
            BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
            num_warps = None
        else:
            BLOCK_M, BLOCK_N, BLOCK_K = 16, 64, 128
            num_warps, num_stages = 4, 2
    else:
        BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32
        num_warps = None

    c = torch.empty((M, N), dtype=out_dtype, device=a.device)
    launch_opts = {}
    if num_warps is not None:
        launch_opts["num_warps"] = num_warps
    if num_stages is not None:
        launch_opts["num_stages"] = num_stages

    if split_k == 1:
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        _w8a8_block_gemm_kernel[grid](
            a_u8, b_u8, a_scales, b_scales, c,
            M, N, K, group_n, group_k,
            a.stride(0), a.stride(1), b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            a_scales.stride(0), a_scales.stride(1),
            b_scales.stride(0), b_scales.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            SPLIT_K=1,
            **launch_opts,
        )
        return c

    partials = torch.empty(
        (split_k, M, N), dtype=torch.float32, device=a.device
    )
    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(N, BLOCK_N),
        split_k,
    )
    _w8a8_block_gemm_kernel[grid](
        a_u8, b_u8, a_scales, b_scales, partials,
        M, N, K, group_n, group_k,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        partials.stride(1), partials.stride(2),  # per-partial [M, N]
        a_scales.stride(0), a_scales.stride(1),
        b_scales.stride(0), b_scales.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        SPLIT_K=split_k,
        **launch_opts,
    )
    _w8a8_block_splitk_reduce[(triton.cdiv(M, 16), triton.cdiv(N, 128))](
        partials, c, M, N, SPLIT_K=split_k, BLOCK_M=16, BLOCK_N=128,
    )
    return c


def _w8a8_block_gemm_fake(
    a, b, a_scales, b_scales, group_n, group_k, out_dtype
):
    M, K = a.shape
    N = b.shape[0]
    return torch.empty((M, N), dtype=out_dtype, device=a.device)


if direct_register_custom_op is not None:
    try:
        # fork/0.28 signature: (op_name, op_func, mutates_args, fake_impl, ...)
        direct_register_custom_op(
            "w8a8_block_gemm_gfx90a",
            _w8a8_block_gemm,
            mutates_args=[],
            fake_impl=_w8a8_block_gemm_fake,
        )
    except TypeError:
        # 0.30 signature: (op_name, op_func, fake_func, ..., mutates_args=...)
        direct_register_custom_op(
            "w8a8_block_gemm_gfx90a",
            _w8a8_block_gemm,
            _w8a8_block_gemm_fake,
            mutates_args=[],
        )
    try:
        direct_register_custom_op(
            "w8a16_block_gemm_gfx90a",
            _w8a16_block_gemm_dispatch,
            mutates_args=[],
            fake_impl=_w8a16_block_gemm_fake,
        )
    except TypeError:
        direct_register_custom_op(
            "w8a16_block_gemm_gfx90a",
            _w8a16_block_gemm_dispatch,
            _w8a16_block_gemm_fake,
            mutates_args=[],
        )


# ---------------------------------------------------------------------------
# Stream-K variant (experiment): persistent grid of exactly NUM_PROGRAMS
# programs; the linearized (m,n,k) tile space is split into contiguous ranges;
# each program walks its range, accumulating per (m,n) run, and atomically adds
# into a pre-zeroed fp32 workspace. A cast epilogue produces the output.
# Eliminates wave quantization vs split-K (e.g. down: 320 programs = 3.08 waves
# on 104 CUs). Atomic fp32 adds make results run-to-run nondeterministic at the
# ulp level -- fine at our 1e-2 rel-err tolerance.
# ---------------------------------------------------------------------------


@triton.jit
def _w8a8_block_gemm_streamk_kernel(
    a_ptr,  # [M, K] uint8
    b_ptr,  # [N, K] uint8
    as_ptr,  # [M, K // G_K] fp32
    bs_ptr,  # [N // G_N, K // G_K] fp32
    c_ptr,  # [M, N] fp32, PRE-ZEROED
    M,
    N,
    K,
    group_n,
    group_k,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_as_m,
    stride_as_k,
    stride_bs_n,
    stride_bs_k,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
):
    pid = tl.program_id(0)
    m_tiles = tl.cdiv(M, BLOCK_M)
    n_tiles = tl.cdiv(N, BLOCK_N)
    k_tiles = tl.cdiv(K, BLOCK_K)
    total = m_tiles * n_tiles * k_tiles

    units = (total + NUM_PROGRAMS - 1) // NUM_PROGRAMS
    u_start = pid * units
    u_end = tl.minimum(u_start + units, total)

    offs_k = tl.arange(0, BLOCK_K)

    u = u_start
    while u < u_end:
        # decode unit -> (mt, nt, kt); linear order m-major, then n, then k
        mt = u // (n_tiles * k_tiles)
        rem = u % (n_tiles * k_tiles)
        nt = rem // k_tiles
        kt = rem % k_tiles
        # run: consecutive units sharing (mt, nt), clipped to this program
        run_limit = (mt * n_tiles + nt + 1) * k_tiles
        run_end = tl.minimum(u_end, run_limit)
        # absolute K range covered by this run
        k_lo = kt * BLOCK_K
        k_hi = tl.minimum(k_lo + (run_end - u) * BLOCK_K, K)

        offs_m_abs = mt * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n_abs = nt * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = offs_m_abs < M
        mask_n = offs_n_abs < N

        as_ptrs = as_ptr + offs_m_abs * stride_as_m
        offs_bsn = offs_n_abs // group_n
        bs_ptrs = bs_ptr + offs_bsn * stride_bs_n

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        a_ptrs = a_ptr + offs_m_abs[:, None] * stride_am \
            + (k_lo + offs_k)[None, :] * stride_ak
        b_ptrs = b_ptr + offs_n_abs[:, None] * stride_bn \
            + (k_lo + offs_k)[None, :] * stride_bk

        for k_abs in range(k_lo, k_hi, BLOCK_K):
            k_rem = k_hi - k_abs
            mask_a = mask_m[:, None] & (offs_k[None, :] < k_rem)
            a_u8 = tl.load(a_ptrs, mask=mask_a, other=0)
            mask_b = mask_n[:, None] & (offs_k[None, :] < k_rem)
            b_u8 = tl.load(b_ptrs, mask=mask_b, other=0)

            a_dot = a_u8.to(tl.float8e4nv, bitcast=True).to(tl.bfloat16)
            b_dot = b_u8.to(tl.float8e4nv, bitcast=True).to(tl.bfloat16)

            offs_ks = k_abs // group_k
            a_s = tl.load(as_ptrs + offs_ks * stride_as_k, mask=mask_m, other=0.0)
            b_s = tl.load(bs_ptrs + offs_ks * stride_bs_k, mask=mask_n, other=0.0)
            s_tile = a_s[:, None] * b_s[None, :]

            accumulator += tl.dot(a_dot, tl.trans(b_dot), out_dtype=tl.float32) \
                * s_tile

            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        c_ptrs = c_ptr + offs_m_abs[:, None] * N + offs_n_abs[None, :]
        mask_c = mask_m[:, None] & mask_n[None, :]
        tl.atomic_add(c_ptrs, accumulator, mask=mask_c)

        u = run_end


@triton.jit
def _cast_f32_out(c32_ptr, out_ptr, total, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    v = tl.load(c32_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, v.to(out_ptr.type.element_ty), mask=mask)


def _streamk_gemm(
    a, b, a_scales, b_scales, group_n, group_k, out_dtype,
    BLOCK_M, BLOCK_N, BLOCK_K, num_programs, num_warps, num_stages,
):
    M, K = a.shape
    N = b.shape[0]
    c32 = torch.zeros((M, N), dtype=torch.float32, device=a.device)
    _w8a8_block_gemm_streamk_kernel[(num_programs,)](
        a.view(torch.uint8), b.view(torch.uint8), a_scales, b_scales, c32,
        M, N, K, group_n, group_k,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        a_scales.stride(0), a_scales.stride(1),
        b_scales.stride(0), b_scales.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        NUM_PROGRAMS=num_programs,
        num_warps=num_warps, num_stages=num_stages,
    )
    out = torch.empty((M, N), dtype=out_dtype, device=a.device)
    total = M * N
    _cast_f32_out[(triton.cdiv(total, 1024),)](c32, out, total, BLOCK=1024)
    return out


if _IN_VLLM_PACKAGE:

    class TritonW8A8Fp8BlockScaledLinearKernel(Fp8BlockScaledMMLinearKernel):
        """
        W8A8 block-scaled FP8 GEMM for ROCm gfx90a: davetha's W8A16 kernel
        structure extended to the official 128x128-block scheme (fp8 weights +
        fp8 activations, in-loop scales). Registered ahead of
        TritonFp8BlockScaledMMKernel on ROCm.

        M <= 16 (decode/spec-decode) runs the W8A16 path: the class sets
        apply_input_quant=False so A arrives as bf16 (the MI210 has no fp8
        MFMA -- the activation quant round-trip is pure overhead here), and
        apply_block_scaled_mm dispatches to the sweep-tuned
        (16,64,128)@4w/2s+split4 W8A16 kernel. M > 16 (prefill) quantizes A
        in the wrapper and runs the original W8A8 ladder unchanged.
        """

        # accept BF16 input directly: skip the base-class input quant
        apply_input_quant = False

        @classmethod
        def is_supported(cls, compute_capability=None):
            if not current_platform.is_rocm():
                return False, "TritonW8A8Fp8BlockScaledLinearKernel requires ROCm"
            from vllm.platforms.rocm import on_gfx90a

            if not on_gfx90a():
                return (
                    False,
                    "TritonW8A8Fp8BlockScaledLinearKernel is only "
                    "tuned/verified on gfx90a",
                )
            return True, None

        def apply_block_scaled_mm(
            self,
            A: torch.Tensor,
            B: torch.Tensor,
            As: torch.Tensor,
            Bs: torch.Tensor,
        ) -> torch.Tensor:
            group_n = list(self.weight_group_shape)[0]
            group_k = list(self.weight_group_shape)[1]
            out_dtype = self.config.out_dtype
            if A.dtype == torch.bfloat16:
                # NOTE: no Python-level M-branch here! The model is compiled
                # once for the range (1, max_num_batched_tokens) and traced
                # at the profile shape (M=4096) -- an `if M <= 16` here gets
                # specialized away at trace time and the compiled graph bakes
                # in the quant+W8A8 path for EVERY M (the W8A16 kernel then
                # never runs; found via dispatch logging 2026-09-27). The
                # runtime M dispatch lives INSIDE the opaque custom op.
                return torch.ops.vllm.w8a16_block_gemm_gfx90a(
                    A, B, Bs, group_n, group_k, out_dtype
                )
            return torch.ops.vllm.w8a8_block_gemm_gfx90a(
                A, B, As, Bs, group_n, group_k, out_dtype
            )
