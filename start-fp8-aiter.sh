#!/bin/bash
# FP8 W8A8 full stack: davetha mi210.6-aiter image + official checkpoint +
# MTP n=3 + ROCM_AITER_FA attention + bf16-cast patch + tuned configs +
# the W8A8 block kernel (davetha-ladder extension) registered first on ROCm.
set -e
V=/opt/python/lib/python3.14/site-packages/vllm
FPU=$V/model_executor/layers/quantization/utils/fp8_utils.py
CFG=$V/model_executor/layers/quantization/utils/configs
SM=$V/model_executor/kernels/linear/scaled_mm
LIN=$V/model_executor/kernels/linear/__init__.py
AITER=/opt/python/lib/python3.14/site-packages/aiter

docker stop fp8-aiter 2>/dev/null || true
docker rm -f fp8-aiter 2>/dev/null || true

# ---- 1) patch the fork's fp8_utils.py (bf16-native cast) ----
docker run --rm --entrypoint cat local/vllm-mi210:mi210.6-aiter $FPU > /home/tai/fp8_utils_fork.py
python3 - <<'EOF'
f = "/home/tai/fp8_utils_fork.py"
src = open(f).read()
old_a = "        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)"
old_b = "        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)"
new_a = old_a + ".to(tl.bfloat16)"
new_b = old_b + ".to(tl.bfloat16)"
na, nb = src.count(old_a), src.count(old_b)
assert na == 1 and nb == 1, f"load-line counts: a={na} b={nb}"
open(f, "w").write(src.replace(old_a, new_a).replace(old_b, new_b))
print("fp8_utils patched")
EOF

# ---- 2) patch the fork's linear __init__.py (register W8A8 kernel first) ----
docker run --rm --entrypoint cat local/vllm-mi210:mi210.6-aiter $LIN > /home/tai/linear_init_fork.py
python3 - <<'EOF'
f = "/home/tai/linear_init_fork.py"
src = open(f).read()
anchor = "from vllm.model_executor.kernels.linear.scaled_mm.triton_fp8_w8a16 import ("
assert src.count(anchor) == 1, "w8a16 import anchor not found"
idx = src.index(anchor)
end = src.index(")", idx) + 1
src = (
    src[:end]
    + "\nfrom vllm.model_executor.kernels.linear.scaled_mm.triton_fp8_w8a8_block import (\n    TritonW8A8Fp8BlockScaledLinearKernel,\n)"
    + src[end:]
)
old_list = """    PlatformEnum.ROCM: [
        AiterFp8BlockScaledMMKernel,
        TritonFp8BlockScaledMMKernel,
    ],"""
new_list = """    PlatformEnum.ROCM: [
        TritonW8A8Fp8BlockScaledLinearKernel,
        AiterFp8BlockScaledMMKernel,
        TritonFp8BlockScaledMMKernel,
    ],"""
assert src.count(old_list) == 1, f"ROCm list anchor count {src.count(old_list)}"
src = src.replace(old_list, new_list)
open(f, "w").write(src)
print("linear __init__ patched: W8A8 kernel registered first on ROCm")
EOF

# ---- 3) harvest AITER JIT .so from the old INT8 container ----
mkdir -p /home/tai/aiter-so && rm -f /home/tai/aiter-so/*.so
docker cp qwen38-mi210:$AITER/jit/. /tmp/aiter-jit-tmp 2>/dev/null || true
cp /tmp/aiter-jit-tmp/*.so /home/tai/aiter-so/ 2>/dev/null || true
rm -rf /tmp/aiter-jit-tmp

# ---- 4) create stopped, install everything ----
docker create --name fp8-aiter \
  --device /dev/kfd --device /dev/dri \
  --group-add video --group-add "$(getent group render | cut -d: -f3)" \
  --ipc host --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  -v /home/tai/models:/models \
  -p 8001:8000 \
  -e VLLM_ROCM_USE_AITER=1 \
  -e VLLM_ROCM_USE_AITER_LINEAR=0 \
  -e HIP_FORCE_DEV_KERNARG=1 \
  -e HSA_NO_SCRATCH_RECLAIM=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e GPU_PINNED_MIN_XFER_SIZE=67108864 \
  local/vllm-mi210:mi210.6-aiter \
  /models/qwen38-27b-fp8 \
    --served-model-name qwen38-27b-fp8 \
    --tensor-parallel-size 1 \
    --max-model-len 262144 \
    --max-num-seqs 8 \
    --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.93 \
    --kv-cache-dtype fp8 \
    --attention-backend ROCM_AITER_FA \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 5, "draft_sample_method": "probabilistic"}' > /dev/null

docker cp /home/tai/fp8_utils_fork.py fp8-aiter:$FPU
docker cp /home/tai/linear_init_fork.py fp8-aiter:$LIN
docker cp /home/tai/w8a8-kernel/triton_fp8_w8a8_block.py fp8-aiter:$SM/triton_fp8_w8a8_block.py
docker cp /home/tai/qwen3_5_mtp_patched.py fp8-aiter:$V/model_executor/models/qwen3_5_mtp.py
# sampler patch (sort-free small-k top-k/top-p + multi-block softmax + draft truncation)
for f in v1/sample/metadata.py v1/sample/ops/row_softmax.py v1/sample/ops/topk_topp_sampler.py \
         v1/sample/rejection_sampler.py v1/sample/sampler.py \
         v1/spec_decode/llm_base_proposer.py v1/worker/gpu_input_batch.py; do
  docker cp /home/tai/sampler-stage/$f fp8-aiter:$V/$f
done
docker cp /home/tai/envs_fork.py fp8-aiter:$V/envs.py
for j in /home/tai/tuned-amd/*.json; do docker cp "$j" fp8-aiter:$CFG/; done
for s in /home/tai/aiter-so/*.so; do docker cp "$s" fp8-aiter:$AITER/jit/; done
echo "installed: fp8 patch + W8A8 kernel + registration + $(ls /home/tai/tuned-amd/*.json | wc -l) configs + $(ls /home/tai/aiter-so/*.so | wc -l) aiter .so"

docker start fp8-aiter
echo "fp8-aiter started with the W8A8 kernel"
