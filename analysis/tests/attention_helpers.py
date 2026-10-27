"""Shared helpers for attention tests (not a test module)."""

import numpy as np


def make_valid_pattern(batch: int, heads: int, seq: int, seed: int = 0) -> np.ndarray:
    """Random causal attention: lower-triangular, each row normalised to sum to 1."""
    rng = np.random.default_rng(seed)
    raw = rng.random((batch, heads, seq, seq)) + 0.01  # strictly positive: no zero rows
    lower = np.tril(raw)  # keep k <= q only
    return lower / lower.sum(axis=-1, keepdims=True)  # keepdims: divide each row by its sum


def selected(mask: np.ndarray) -> list[tuple[int, int]]:
    """The (q, k) cells a mask selects, in row order."""
    return [(int(q), int(k)) for q, k in np.argwhere(mask)]


def one_hot_head(seq: int, source_of: dict[int, int]) -> np.ndarray:
    """A [1, 1, seq, seq] pattern where row q puts all attention on source_of[q] (default: q)."""
    pattern = np.zeros((1, 1, seq, seq))
    for q in range(seq):
        pattern[0, 0, q, source_of.get(q, q)] = 1.0
    return pattern


def previous_token_head(seq: int) -> np.ndarray:
    """Ideal previous-token head: row q attends to q - 1 (row 0 can only attend to itself)."""
    return one_hot_head(seq, {q: q - 1 for q in range(1, seq)})


def sink_head(seq: int) -> np.ndarray:
    """Ideal attention-sink head: every row attends to position 0."""
    return one_hot_head(seq, dict.fromkeys(range(seq), 0))


def self_head(seq: int) -> np.ndarray:
    """Every row attends to itself."""
    return one_hot_head(seq, {})


def repeated_head(repeat_len: int, offset: int) -> np.ndarray:
    """Head on (BOS, X, X), |X| = repeat_len: 2nd-copy rows attend to q - repeat_len + offset.

    Rows 0..repeat_len (BOS and the first copy) sink to position 0, a common real behaviour.
    offset=1 → induction head; offset=0 → duplicate-token head.
    """
    seq = 1 + 2 * repeat_len
    source = {q: 0 for q in range(repeat_len + 1)}
    source |= {q: q - repeat_len + offset for q in range(repeat_len + 1, seq)}
    return one_hot_head(seq, source)


def uniform_head(seq: int) -> np.ndarray:
    """No preference: row q spreads attention evenly over k <= q."""
    lower = np.tril(np.ones((seq, seq)))
    return (lower / lower.sum(axis=-1, keepdims=True))[None, None]


def mix(patterns: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """Weighted sum of patterns, e.g. 70% induction + 30% sink. Weights must sum to 1 so rows
    still sum to 1, like a real head that does several things at once."""
    assert len(patterns) == len(weights) and abs(sum(weights) - 1) < 1e-12
    return sum(w * p for w, p in zip(weights, patterns, strict=True))
