"""Masked attention scoring shared by every attention detector (checkpoint 2).

A detector is defined by *which (q, k) cells it looks at*. The cells are given as a boolean
mask; this module turns a pattern + mask into one score per (example, head).
"""

from dataclasses import dataclass

import numpy as np

from cirqloom_analysis.detectors.attention import validate_attention_pattern


@dataclass(frozen=True)
class MaskedScore:
    """Result of scoring one batch of attention patterns with one mask."""

    per_example: np.ndarray
    """Shape [batch, head]: mean attention weight over the masked cells, per example and head."""

    support: int
    """How many (q, k) cells the mask selects. Reported with every score (AGENTS.md §7.4)."""


def masked_head_mean(pattern: np.ndarray, mask: np.ndarray) -> MaskedScore:
    """Mean attention weight over the cells selected by `mask`, per example and head.

    Inputs:
      pattern: [batch, head, seq, seq], must pass validate_attention_pattern
      mask:    [seq, seq] bool, True on the (q, k) cells the detector looks at

    Requirements:
      - validate `pattern` first (reuse checkpoint 1)
      - mask must be 2-D bool with shape (seq, seq) matching the pattern → ValueError otherwise
      - mask must select at least one cell → ValueError (an empty mask would give NaN)
      - mask must not select any future cell (k > q) → ValueError (no future source indices)
      - no Python loops over batch or head
    """
    # Step 1: validate the pattern (checkpoint 1). It raises AttentionPatternError on any
    # problem, so nothing below runs on a malformed tensor.
    validate_attention_pattern(pattern)

    seq = pattern.shape[-1]

    # Step 2a: the mask must be boolean. An int 0/1 array would trigger numpy's integer
    # (fancy) indexing and silently select whole rows instead of cells.
    if mask.dtype != bool:
        raise ValueError(f"mask must be bool, got {mask.dtype}")

    # Step 2b: the mask must cover exactly the pattern's (dest, src) grid.
    if mask.shape != (seq, seq):
        raise ValueError(f"mask shape {mask.shape} does not match pattern seq {seq}")

    # Step 2c: at least one cell, otherwise the mean is 0/0 = NaN.
    if not mask.any():
        raise ValueError("mask selects no cells; the mean would be NaN")

    # Step 2d: no future cells (k > q). They are always 0 in a causal model, so including
    # them would silently drag the score down; it signals a bug (e.g. off-by-one) in the mask.
    future = np.triu(np.ones((seq, seq), dtype=bool), k=1)
    if (mask & future).any():
        raise ValueError("mask selects future cells (k > q)")

    # Steps 3–4: pick the masked cells for every batch and head at once
    # → [batch, head, n_cells], then average over the cells → [batch, head].
    per_example = pattern[..., mask].mean(axis=-1)

    # Step 5: return the scores together with how many cells they are based on.
    return MaskedScore(per_example=per_example, support=int(mask.sum()))
