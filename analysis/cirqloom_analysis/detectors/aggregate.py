"""Aggregate per-example detector scores into per-head summaries (checkpoint 6, AGENTS.md §7.1-7).

Unit of analysis = one example (prompt). The cells inside one example (its `support`) are
not independent: the same head reads the same sequence, so an easy sequence lifts all of
them together. Counting cells as samples would inflate n (pseudo-replication) and make the
evidence look stronger than it is. So:
  - n    = number of examples,
  - std  = spread between example scores, not between cells,
  - batches with different support (e.g. different repeat lengths) are pooled with equal
    weight per example; each example's support is kept alongside.

Only pool scores from the same detector and prompt suite: scores from different detectors or
suites are not on a comparable scale (AGENTS.md §5). That is the caller's (and the detector
spec's) responsibility; MaskedScore does not carry a detector id.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from cirqloom_analysis.detectors.scoring import MaskedScore


@dataclass(frozen=True)
class HeadSummary:
    """Per-head summary of one detector over one or more batches of examples."""

    mean: np.ndarray
    """Shape [head]: mean score over examples."""

    std: np.ndarray | None
    """Shape [head]: sample standard deviation (ddof=1) over examples.
    None when n < 2: the spread of a single example is undefined, and NaN must never reach
    a bundle (AGENTS.md §12). 0.0 would wrongly claim the score is perfectly consistent."""

    n: int
    """Number of examples (prompts). Not the number of mask cells."""

    per_example: np.ndarray
    """Shape [n, head]: every example's score, kept for inspection (AGENTS.md §7.1-7)."""

    support: np.ndarray
    """Shape [n]: how many mask cells each example's score averages over."""


def summarize(scores: Sequence[MaskedScore]) -> HeadSummary:
    """Pool one or more MaskedScore batches (same heads) into a per-head summary."""
    # Check 1: something to summarise. A head with no measurement is reported as
    # status "unavailable" with score None further down the pipeline, never as a mean of nothing.
    if len(scores) == 0:
        raise ValueError("no scores to summarize")

    # Check 2: every batch must score the same heads, i.e. the same trailing shape.
    head_shape = scores[0].per_example.shape[1:]
    for i, s in enumerate(scores):
        if s.per_example.ndim < 2 or s.per_example.shape[1:] != head_shape:
            raise ValueError(
                f"batch {i} has per_example shape {s.per_example.shape}; "
                f"expected [batch, {', '.join(map(str, head_shape))}]"
            )

    # Pool examples with equal weight: stack along the example axis.
    per_example = np.concatenate([s.per_example for s in scores], axis=0)
    support = np.concatenate([np.full(s.per_example.shape[0], s.support) for s in scores])
    n = per_example.shape[0]

    # Check 3: at least one example, and no NaN/inf (validated patterns and non-empty masks
    # should guarantee this; checked here because this is the last step before export).
    if n == 0:
        raise ValueError("batches contain no examples")
    if not np.isfinite(per_example).all():
        raise ValueError("per-example scores contain NaN or inf")

    mean = per_example.mean(axis=0)
    # ddof=1: sample standard deviation, the usual estimate of spread between examples.
    std = per_example.std(axis=0, ddof=1) if n >= 2 else None

    return HeadSummary(mean=mean, std=std, n=n, per_example=per_example, support=support)
