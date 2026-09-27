#!/bin/bash
# Port HyperQwen's qwen3_5-mtp-draft-vocab patch (4 hunks) onto the fork's
# qwen3_5_mtp.py, build the draft-head artifacts, and stage everything.
# Probabilistic draft sampling is a config flag (draft_sample_method) -- no port.
set -e
V=/opt/python/lib/python3.14/site-packages/vllm
MTP=$V/model_executor/models/qwen3_5_mtp.py

# ---- 1) port the patch (string replacement, anchors verified on the fork) ----
docker run --rm --entrypoint cat local/vllm-mi210:mi210.6-aiter $MTP > /home/tai/qwen3_5_mtp_patched.py
python3 - <<'EOF'
f = "/home/tai/qwen3_5_mtp_patched.py"
src = open(f).read()

# Hunk 1: draft head creation, after the embed_tokens block
a1 = """        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )

        # Workaround: mtp.fc is stored as BF16 in NVFP4 checkpoints but is"""
b1 = """        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )

        # syv patch: vocab-truncated draft head. If the model dir ships
        # mtp_draft_vocab_ids.pt, the drafter scores only those rows
        # (mtp.draft_lm_head.*) instead of the full 248k-row lm_head; logits
        # of all other ids are -inf. Speculative decoding stays exact, only
        # the acceptance rate can change. MTP_DRAFT_VOCAB=0 disables.
        import os as _os
        self.draft_lm_head = None
        self.draft_vocab_ids = None
        _ids_path = _os.path.join(model_config.model, "mtp_draft_vocab_ids.pt")
        if _os.path.exists(_ids_path) and _os.environ.get("MTP_DRAFT_VOCAB", "1") != "0":
            _ids = torch.load(_ids_path, map_location="cpu")
            self.draft_vocab_ids = _ids
            self.draft_lm_head = ParallelLMHead(
                int(_ids.numel()),
                config.hidden_size,
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "draft_lm_head"),
            )
            logger.info("MTP drafter uses a %d-token draft head", int(_ids.numel()))

        # Workaround: mtp.fc is stored as BF16 in NVFP4 checkpoints but is"""
assert src.count(a1) == 1, f"hunk1 anchor {src.count(a1)}"
src = src.replace(a1, b1)

# Hunk 2: draft logits processor, after the main one
a2 = "        self.logits_processor = LogitsProcessor(config.vocab_size)\n"
b2 = a2 + """        # syv patch: vocab-truncated draft head
        self.draft_logits_processor = (
            LogitsProcessor(int(self.model.draft_vocab_ids.numel()))
            if getattr(self.model, "draft_lm_head", None) is not None
            else None
        )
"""
assert src.count(a2) == 1, f"hunk2 anchor {src.count(a2)}"
src = src.replace(a2, b2)

# Hunk 3: compute_logits draft-head scoring path
a3 = """    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)"""
b3 = """    ) -> torch.Tensor | None:
        # syv patch: vocab-truncated draft head
        if self.draft_logits_processor is not None:
            sub = self.draft_logits_processor(self.model.draft_lm_head, hidden_states)
            if sub is None:
                return None
            ids = self.model.draft_vocab_ids
            if ids.device != sub.device:
                ids = ids.to(sub.device)
                self.model.draft_vocab_ids = ids
            full = sub.new_full((sub.shape[0], self.config.vocab_size), float("-inf"))
            full.index_copy_(1, ids, sub)
            return full
        return self.logits_processor(self.lm_head, hidden_states)"""
assert src.count(a3) == 1, f"hunk3 anchor {src.count(a3)}"
src = src.replace(a3, b3)

# Hunk 4: load_weights skip when the draft head is disabled
a4 = """        def remap_weight_names(weights):
            for name, weight in weights:
                if name.startswith("mtp."):"""
b4 = """        def remap_weight_names(weights):
            for name, weight in weights:
                # syv patch: skip the truncated draft head when it is disabled
                if "draft_lm_head" in name and self.model.draft_lm_head is None:
                    continue
                if name.startswith("mtp."):"""
assert src.count(a4) == 1, f"hunk4 anchor {src.count(a4)}"
src = src.replace(a4, b4)

open(f, "w").write(src)
print("qwen3_5_mtp.py patched: 4/4 hunks applied")
EOF

# ---- 2) build the draft-head artifacts (idempotent; additive to the model dir) ----
docker run --rm --entrypoint python3 \
  -v /home/tai/models:/models \
  -v /home/tai/mtp-opt:/prep:ro \
  local/vllm-mi210:mi210.6-aiter /prep/build-draft-head.py

echo "artifacts built. Install happens at next launch (start-fp8-aiter.sh)."
