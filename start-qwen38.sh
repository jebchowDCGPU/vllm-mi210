#!/bin/bash
# Recreate + start the qwen38-mi210 INT8 baseline container.
# Fix vs original: --device /dev/dri (whole dir) instead of a pinned renderD129,
# which broke after reboot renumbering. Profiler flags dropped for clean baseline.
set -e

# Backup the old container definition before removing it
docker inspect qwen38-mi210 > ~/qwen38-mi210-inspect-backup.json 2>/dev/null || true
docker rm -f qwen38-mi210 2>/dev/null || true

docker run -d --name qwen38-mi210 \
  --device /dev/kfd --device /dev/dri \
  --group-add video --group-add "$(getent group render | cut -d: -f3)" \
  --ipc host --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  -p 8000:8000 \
  -v /home/tai/models:/home/tai/models \
  -v /home/tai/mi210-cache:/cache \
  -v /home/tai/profiles:/profiles \
  -e HIP_FORCE_DEV_KERNARG=1 \
  -e HSA_COREDUMP_PATTERN=/dev/null \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_ROCM_USE_AITER=1 \
  -e HSA_NO_SCRATCH_RECLAIM=1 \
  local/vllm-mi210:mi210.6-aiter \
  /home/tai/models/qwen38-27b-int8 \
    --served-model-name qwen38-27b \
    --tensor-parallel-size 1 \
    --max-model-len 131072 \
    --max-num-seqs 64 \
    --max-num-batched-tokens 2048 \
    --gpu-memory-utilization 0.90 \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --speculative-config '{"method": "dflash", "model": "/home/tai/models/qwen38-dflash2", "num_speculative_tokens": 8}'

echo "container started; tailing logs..."
sleep 20
docker logs --tail 20 qwen38-mi210 2>&1
