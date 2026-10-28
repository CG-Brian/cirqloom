"""GPT-2 Small adapter on TransformerLens TransformerBridge (M1 Phase 2, P2-2).

The only module that imports torch / transformer_lens. Everything it returns is numpy.

Verified 2026-10-27 with transformer-lens 4.3.0, torch 2.14.1, transformers 5.21.0, loading
config only (no weights):
  - cfg: n_layers 12, n_heads 12, n_ctx 1024, d_vocab 50257, default_prepend_bos True
  - tokenizer.bos_token_id 50256 (<|endoftext|>)
  - attention hook per layer: "blocks.{layer}.attn.hook_pattern"
  - HF revision below is the openai-community/gpt2 main commit on that date

Attention patterns do not need enable_compatibility_mode() (it folds LayerNorm and centres
weights, which changes logit-attribution coordinates, not softmax attention). Do not call it
here; a later logit-attribution adapter can.
"""

import numpy as np
import torch
from transformer_lens.model_bridge import TransformerBridge

from cirqloom_analysis.models.base import AdapterInfo, BaseModelAdapter

MODEL_ID = "gpt2-small"
HF_ID = "openai-community/gpt2"
REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
PATTERN_HOOK = "blocks.{layer}.attn.hook_pattern"

# What the pinned model must look like. Checked after loading, so a wrong revision or a library
# change fails loudly instead of producing a bundle for a different architecture.
EXPECTED = {"n_layers": 12, "n_heads": 12, "n_ctx": 1024, "d_vocab": 50257}
EXPECTED_BOS_TOKEN_ID = 50256


class GPT2Adapter(BaseModelAdapter):
    """GPT-2 Small, pinned revision, eval mode, float32."""

    def __init__(self, device: str = "cpu", seed: int = 0) -> None:
        # Step 1: reproducibility. The eval forward pass is deterministic; the seed covers
        # anything random later.
        torch.manual_seed(seed)

        # Step 2: load the pinned weights.
        self._model = TransformerBridge.boot_transformers(
            HF_ID, revision=REVISION, device=device, dtype=torch.float32
        )

        # Step 3: eval mode turns dropout off.
        self._model.eval()

        # Step 4: check the loaded model is the architecture we pinned. A missing attribute
        # (renamed by a library update) is reported as a mismatch, not an AttributeError, so
        # every problem is listed at once.
        mismatches = []
        for key, expected_value in EXPECTED.items():
            actual_value = getattr(self._model.cfg, key, None)
            if actual_value != expected_value:
                mismatches.append(f"{key}: expected {expected_value}, got {actual_value}")

        bos_token_id = self._model.tokenizer.bos_token_id
        if bos_token_id != EXPECTED_BOS_TOKEN_ID:
            mismatches.append(
                f"bos_token_id: expected {EXPECTED_BOS_TOKEN_ID}, got {bos_token_id}"
            )

        if mismatches:
            raise ValueError(f"{HF_ID}@{REVISION} is not GPT-2 Small: " + "; ".join(mismatches))

        self._device = device
        self._info = AdapterInfo(
            model_id=MODEL_ID,
            hf_id=HF_ID,
            revision=REVISION,
            tokenizer_id=HF_ID,
            n_layers=self._model.cfg.n_layers,
            n_heads=self._model.cfg.n_heads,
            n_ctx=self._model.cfg.n_ctx,
            vocab_size=self._model.cfg.d_vocab,
            bos_token_id=bos_token_id,
        )

    @property
    def info(self) -> AdapterInfo:
        # Step 5: built once in __init__ from the checked cfg; AdapterInfo is immutable.
        return self._info

    def tokenize(self, text: str) -> np.ndarray:
        # Step 6: prepend_bos is passed explicitly: every mask depends on BOS at position 0,
        # so a library default must not decide it.
        tokens = self._model.to_tokens(text, prepend_bos=True)  # torch [1, seq]
        return tokens[0].detach().cpu().numpy().astype(np.int64)

    def token_strings(self, token_ids: np.ndarray) -> list[str]:
        # Decode one id at a time so each label is exactly that token. A byte-level fragment
        # of a multi-byte character decodes to U+FFFD on its own; that is the honest label.
        return [self._model.tokenizer.decode([int(i)]) for i in token_ids]

    def _run_attention(self, token_ids: np.ndarray) -> np.ndarray:
        # Inputs are already validated by BaseModelAdapter.run_attention.

        # Step 7: numpy → torch on the model's device.
        tokens = torch.from_numpy(token_ids.astype(np.int64)).to(self._device)

        # Step 8: forward pass, caching only the attention patterns (§7.1-3).
        names = [PATTERN_HOOK.format(layer=layer) for layer in range(self._info.n_layers)]
        with torch.inference_mode():
            _, cache = self._model.run_with_cache(tokens, names_filter=names)

        # Step 9: [batch, head, q, k] per layer → [batch, layer, head, q, k], back to numpy.
        patterns = torch.stack([cache[name] for name in names], dim=1)
        return patterns.detach().cpu().numpy()
