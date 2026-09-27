#!/bin/bash
# Patch rocm_aiter_fa.py: route FP8 KV decode through the Triton
# unified_attention kernel instead of the HIP paged_attention_v1
# (which cannot compile for float8_e4m3fn on gfx90a).
set -e
V=/opt/python/lib/python3.14/site-packages
FA=$V/vllm/v1/attention/backends/rocm_aiter_fa.py

docker run --rm --entrypoint cat local/vllm-mi210:mi210.6-aiter $FA > /home/tai/rocm_aiter_fa_patched.py
python3 /home/tai/patch_fa.py

# update launch script
grep -q "rocm_aiter_fa_patched" ~/start-fp8-aiter.sh || \
  sed -i "\|docker cp /home/tai/unified_attention_patched.py|a docker cp /home/tai/rocm_aiter_fa_patched.py fp8-aiter:$FA" ~/start-fp8-aiter.sh
echo "patched and staged"
