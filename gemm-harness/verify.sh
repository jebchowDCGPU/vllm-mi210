#!/bin/bash
# verify.sh — the kernel gate. NEVER ship a kernel change without this passing.
#
# Runs test_w8a8_block_gfx90a.py: 15 correctness checks vs the fp32 dequant
# reference (rel err < 1e-2) + benchmark vs the stock kernel (tuned configs).
# Stops the server first (one GPU).
set -euo pipefail
cd "$(dirname "$0")"
./run-test.sh test_w8a8_block_gfx90a.py --stop-server
