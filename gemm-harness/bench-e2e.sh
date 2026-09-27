#!/bin/bash
# bench-e2e.sh — warm e2e benchmark against the running server.
#
# Usage: ./bench-e2e.sh [model-name] [port] [--skip-warmup]
#
# Warms up (first requests after a fresh container JIT-compile — discard),
# then runs the standard prompt set and prints decode tok/s per prompt plus
# the MTP acceptance rate. Compare runs only on the same prompt set.
set -euo pipefail
MODEL="${1:-qwen38-27b-fp8}"
PORT="${2:-8001}"
SKIP_WARMUP="${3:-}"

gen() {
  local PROMPT="$1" TOKS="$2"
  local T0=$(date +%s.%N)
  R=$(curl -s --max-time 240 "localhost:$PORT/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL\",\"prompt\":\"$PROMPT\",\"max_tokens\":$TOKS,\"temperature\":0}")
  local T1=$(date +%s.%N)
  echo "$R" | T0="$T0" T1="$T1" python3 -c "
import json, sys, os
r = json.load(sys.stdin)
ct = r['usage']['completion_tokens']
w = float(os.environ['T1']) - float(os.environ['T0'])
print(f'decode={ct/w:.1f} tok/s  | {r[\"choices\"][0][\"text\"][:55].replace(chr(10), \" \")}')
"
}

if [ "$SKIP_WARMUP" != "--skip-warmup" ]; then
  echo "=== warmup x3 (JIT) ==="
  gen "Hi" 30
  gen "Hello there, how are you today?" 30
  gen "The quick brown fox" 30
fi

echo "=== measured ==="
gen "Explain how a transformer attention mechanism works." 200
gen "Describe the process of photosynthesis in plants." 200
gen "What causes the seasons on Earth?" 200
gen "Write a detailed essay about the history of computing." 300
gen "Write a Python quicksort implementation with tests." 300
gen "Explain the CAP theorem and its implications for distributed systems." 300

echo "=== MTP acceptance ==="
curl -s "localhost:$PORT/metrics" \
  | grep '^vllm:spec_decode_num_\(accepted\|draft\)_tokens_total' | grep -v created
