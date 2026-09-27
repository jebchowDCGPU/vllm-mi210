import re

f = "/home/tai/rocm_aiter_fa_patched.py"
src = open(f).read()

# Route ALL decode through the Triton unified_attention when KV is FP8
old = (
    "                if (\n"
    "                    self.sliding_window[0] != -1\n"
    "                    or decode_max_query_len > 1\n"
    "                    or self.sinks is not None\n"
    "                ):"
)
new = (
    "                # gfx90a patch: also route FP8 KV through unified_attention\n"
    "                # (paged_attention_v1 HIP kernel cannot compile fn on CDNA2)\n"
    "                if (\n"
    "                    self.sliding_window[0] != -1\n"
    "                    or decode_max_query_len > 1\n"
    "                    or self.sinks is not None\n"
    "                    or self.kv_cache_dtype == torch.float8_e4m3fn\n"
    "                ):"
)
assert src.count(old) == 1, f"condition anchor count {src.count(old)}"
src = src.replace(old, new)
open(f, "w").write(src)
print("rocm_aiter_fa.py patched: FP8 KV decode -> Triton unified_attention")
