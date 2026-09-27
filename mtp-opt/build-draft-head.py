#!/usr/bin/env python3
"""Build the vocab-truncated draft head for MTP speculative decoding.

Adapted from HyperQwen's prepare/build_draft_vocab.py for OUR checkpoint
(bf16 lm_head, not int8): slices the top-N lm_head rows listed in
draft_vocab_ids.json (HyperQwen's shipped list: Danish web + English Wikipedia
+ Python + model outputs, 95% held-out coverage, same tokenizer) into a new
shard `draft-head.safetensors`, writes mtp_draft_vocab_ids.pt, and updates
model.safetensors.index.json (backed up first). Additive -- the original
tensors are untouched; delete the three files to revert.
"""
import json
import shutil

import torch
from safetensors.torch import load_file, save_file

MODEL = "/models/qwen38-27b-fp8"
IDS = "/prep/draft_vocab_ids.json"

ids = sorted(set(json.load(open(IDS))))
ids_t = torch.tensor(ids, dtype=torch.int64)
print(f"draft vocab: {len(ids)} ids")

tensors = load_file(f"{MODEL}/outside.safetensors")
lm = tensors["lm_head.weight"]  # [248320, 5120] bf16
assert lm.shape[0] > max(ids)
sub = lm.index_select(0, ids_t.to(lm.device)).contiguous()
print(f"draft head: {tuple(sub.shape)} {sub.dtype} = {sub.numel() * 2 / 1e6:.0f} MB")

save_file({"mtp.draft_lm_head.weight": sub}, f"{MODEL}/draft-head.safetensors")
torch.save(ids_t, f"{MODEL}/mtp_draft_vocab_ids.pt")

# index update (backup first); the index is the commit point
shutil.copy2(f"{MODEL}/model.safetensors.index.json",
             f"{MODEL}/model.safetensors.index.json.bak-draft")
idx = json.load(open(f"{MODEL}/model.safetensors.index.json"))
idx["weight_map"]["mtp.draft_lm_head.weight"] = "draft-head.safetensors"
with open(f"{MODEL}/model.safetensors.index.json", "w") as f:
    json.dump(idx, f, indent=2)
print("index updated; DONE")
