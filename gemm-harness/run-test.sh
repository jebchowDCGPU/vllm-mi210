#!/bin/bash
# run-test.sh — run a python test script in a throwaway GPU container.
#
# Usage:  ./run-test.sh <script.py> [--stop-server]
#
# The script is copied to /test.py in the container; the W8A8 kernel is placed
# at /tmp/triton_fp8_w8a8_block.py (tests import it via sys.path /tmp). The
# patched fp8_utils.py and the tuned stock-kernel configs are installed so the
# "stock" comparison arm matches production.
#
# --stop-server stops the fp8-aiter server first (one GPU: a running server
# corrupts timings). Without the flag, a WARNING is printed if it is up.
set -euo pipefail
cd "$(dirname "$0")"

V=/opt/python/lib/python3.14/site-packages/vllm
CFG=$V/model_executor/layers/quantization/utils/configs
FPU=$V/model_executor/layers/quantization/utils/fp8_utils.py

SCRIPT="${1:-}"
[ -n "$SCRIPT" ] && [ -f "$SCRIPT" ] || { echo "usage: $0 <script.py> [--stop-server]"; exit 1; }
shift || true

if [ "${1:-}" = "--stop-server" ]; then
  docker stop fp8-aiter > /dev/null 2>&1 || true
  shift || true
fi
if docker ps --format '{{.Names}}' | grep -q '^fp8-aiter$'; then
  echo "WARNING: fp8-aiter server is UP — GPU contention will corrupt timings." >&2
  echo "         Re-run with --stop-server, or: docker stop fp8-aiter" >&2
fi

docker rm -f gemm-test > /dev/null 2>&1 || true
docker create --name gemm-test --entrypoint python3 \
  --device /dev/kfd --device /dev/dri \
  --group-add video --group-add "$(getent group render | cut -d: -f3)" \
  --ipc host \
  local/vllm-mi210:mi210.6-aiter /test.py > /dev/null

docker cp "$SCRIPT" gemm-test:/test.py
docker cp triton_fp8_w8a8_block.py gemm-test:/tmp/triton_fp8_w8a8_block.py
# baseline kernel (for A/B tests); harmless if absent
[ -f triton_fp8_w8a8_block_baseline.py ] && \
  docker cp triton_fp8_w8a8_block_baseline.py gemm-test:/tmp/triton_fp8_w8a8_block_baseline.py || true
docker cp "$HOME/fp8_utils_fork.py" gemm-test:"$FPU"
for j in "$HOME"/tuned-amd/*.json; do docker cp "$j" gemm-test:"$CFG/"; done

echo "=== running $SCRIPT ==="
docker start -a gemm-test
docker rm -f gemm-test > /dev/null 2>&1 || true
