"""Model adapter contract (M1 Phase 2, AGENTS.md §7.1, §14).

An adapter is the only place that knows the model library (TransformerLens, torch, hook names,
devices). Everything outside it speaks numpy and this module's conventions:

    token_ids  numpy int [batch, seq]                position 0 is BOS in every row
    patterns   numpy float [batch, layer, head, q, k]  row = destination, column = source

Inputs go in as numpy and are moved to torch/GPU inside the adapter; outputs come back with
.detach().cpu().numpy(). A new model means a new subclass; detectors and the pipeline do not
change.

No padding: every row of a batch has the same length. Padding would shift BOS (left padding)
or make one seq-derived mask cover prompts of different lengths, and both give plausible but
wrong scores. The pipeline groups prompts by length and runs each group separately.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from cirqloom_analysis.detectors.attention import validate_attention_pattern
from cirqloom_analysis.schema.bundle import ModelInfo


@dataclass(frozen=True)
class AdapterInfo:
    """What the pipeline and exporter need to know about a loaded model."""

    model_id: str
    """Our id, e.g. "gpt2-small". Used in node ids."""

    hf_id: str
    """Hugging Face repository, e.g. "openai-community/gpt2"."""

    revision: str
    """Exact pinned revision (commit hash) of the weights."""

    tokenizer_id: str

    n_layers: int
    n_heads: int

    n_ctx: int
    """Maximum sequence length the model accepts (GPT-2: 1024 positions)."""

    vocab_size: int
    """Valid token ids are 0 .. vocab_size - 1."""

    bos_token_id: int
    """Token placed at position 0 of every prompt (GPT-2: <|endoftext|>, 50256)."""

    def to_bundle_model(self) -> ModelInfo:
        """The bundle's `model` record (AGENTS.md §8)."""
        return ModelInfo(
            id=self.model_id,
            hf_id=self.hf_id,
            revision=self.revision,
            layers=self.n_layers,
            heads_per_layer=self.n_heads,
            tokenizer_id=self.tokenizer_id,
        )


def validate_token_ids(token_ids: np.ndarray, info: AdapterInfo) -> None:
    """Reject inputs that would give wrong or meaningless attention patterns.

    Requirements, in this order:
      - token_ids is a numpy array with ndim == 2: [batch, seq]
      - integer dtype (bool is not an integer here)
      - batch >= 1 and seq >= 1
      - seq <= info.n_ctx
      - every id in 0 .. info.vocab_size - 1
      - position 0 is info.bos_token_id in every row
      - no info.bos_token_id at positions 1 .. seq - 1

    Equal row lengths need no check: a 2-D array is rectangular by construction.

    Raises TypeError for a wrong type or dtype, ValueError for everything else.
    """
    # Check 1: a 2-D numpy array. A nested list would pass np.ndim but has no .dtype.
    if not isinstance(token_ids, np.ndarray):
        raise TypeError(f"token_ids must be a numpy array, got {type(token_ids).__name__}")
    if token_ids.ndim != 2:
        raise ValueError(f"token_ids must be [batch, seq], got shape {token_ids.shape}")

    # Check 2: any integer dtype (int32, int64, ...). numpy bool is not an integer subtype,
    # so this also rejects bool arrays.
    if not np.issubdtype(token_ids.dtype, np.integer):
        raise TypeError(f"token_ids must have an integer dtype, got {token_ids.dtype}")

    # Check 3: size. Checked before the value checks so .min()/.max() below never see an
    # empty array.
    batch, seq = token_ids.shape
    if batch < 1:
        raise ValueError("token_ids has no prompts (batch = 0)")
    if seq < 1:
        raise ValueError("token_ids has no positions (seq = 0)")
    if seq > info.n_ctx:
        raise ValueError(f"seq = {seq} exceeds the model's context length n_ctx = {info.n_ctx}")

    # Check 4: every id is in the vocabulary.
    low, high = int(token_ids.min()), int(token_ids.max())
    if low < 0 or high >= info.vocab_size:
        raise ValueError(
            f"token ids must be in 0..{info.vocab_size - 1}, got range {low}..{high}"
        )

    # Check 5: BOS at position 0 in every row; every mask assumes it.
    first = token_ids[:, 0]
    if not (first == info.bos_token_id).all():
        bad = np.flatnonzero(first != info.bos_token_id).tolist()
        raise ValueError(f"position 0 must be BOS ({info.bos_token_id}) in every row; rows {bad}")

    # Check 6: no BOS after position 0; it would add a second attention sink mid-sequence.
    # With seq == 1 the slice is empty and .any() is False, so a lone BOS passes.
    rows, positions = np.nonzero(token_ids[:, 1:] == info.bos_token_id)
    if rows.size:
        cells = list(zip(rows.tolist(), (positions + 1).tolist(), strict=True))
        raise ValueError(f"BOS ({info.bos_token_id}) found after position 0 at (row, pos) {cells}")


class BaseModelAdapter(ABC):
    """Loads one model and returns its attention patterns as numpy arrays."""

    @property
    @abstractmethod
    def info(self) -> AdapterInfo:
        """Architecture and provenance of the loaded model."""

    @abstractmethod
    def tokenize(self, text: str) -> np.ndarray:
        """Token ids [seq] for one text, with BOS prepended at position 0."""

    @abstractmethod
    def token_strings(self, token_ids: np.ndarray) -> list[str]:
        """Display string for each id in a 1-D array, leading spaces kept (" Mary", not
        "Mary"). For labels only: never re-tokenize these strings."""

    @abstractmethod
    def _run_attention(self, token_ids: np.ndarray) -> np.ndarray:
        """Library-specific forward pass. Receives already validated token ids and returns
        numpy [batch, layer, head, q, k]. Subclasses implement this, not run_attention."""

    def run_attention(self, token_ids: np.ndarray) -> np.ndarray:
        """Attention patterns [batch, layer, head, q, k] for a batch of equal-length prompts.

        Validates both sides of the model call, so every adapter gets the same checks:
        bad input never reaches the model, and a bad pattern never reaches a detector.
        """
        validate_token_ids(token_ids, self.info)
        patterns = self._run_attention(token_ids)

        batch, seq = token_ids.shape
        expected = (batch, self.info.n_layers, self.info.n_heads, seq, seq)
        if not isinstance(patterns, np.ndarray) or patterns.shape != expected:
            got = patterns.shape if isinstance(patterns, np.ndarray) else type(patterns).__name__
            raise ValueError(f"adapter returned attention {got}; expected {expected}")

        # Fold layers and heads into one axis (a view for contiguous arrays) to reuse the 4-D check.
        validate_attention_pattern(patterns.reshape(batch, -1, seq, seq))
        return patterns
