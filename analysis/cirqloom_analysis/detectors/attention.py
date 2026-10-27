"""Attention tensor contract shared by every detector (AGENTS.md §7.1 steps 4–5).

Convention: pattern[batch, head, destination (q), source (k)].
Row q is the token that reads; column k is the token it reads from.
"""

import numpy as np


class AttentionPatternError(ValueError):
    """Raised when an attention tensor breaks the contract. Lists every problem at once."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        lines = "\n".join(f" - {p}" for p in problems)
        super().__init__(f"Invalid attention pattern ({len(problems)} problems):\n{lines}")


def validate_attention_pattern(pattern: np.ndarray, atol: float = 1e-4) -> None:
    """Check that `pattern` is a valid causal attention tensor.

    Requirements:
      - ndim == 4: [batch, n_heads, seq_dest, seq_src]      → stop immediately if not
      - seq_dest == seq_src                                  → stop immediately if not
      - all values finite (no NaN / inf)
      - no negative values (below -atol)
      - future cells (k > q) are 0 within atol
      - every row sums to 1 within atol

    Collect all problems from the last four checks, then raise AttentionPatternError once.
    No Python loops over batch or head: use vectorised numpy and broadcasting.
    """
    # Check 1 (stop): must be 4-D [batch, head, dest, src]. Every later check indexes the
    # last two axes, so a wrong rank would make them meaningless or crash.
    if pattern.ndim != 4:
        raise AttentionPatternError(
            [f"expected 4 dims [batch, head, dest, src], got shape {pattern.shape}"]
        )

    # Check 2 (stop): destination and source axes must have the same length (square).
    seq_dest, seq_src = pattern.shape[-2:]
    if seq_dest != seq_src:
        raise AttentionPatternError(
            [f"destination and source lengths differ: {seq_dest} vs {seq_src}"]
        )

    problems: list[str] = []

    # Check 3: no NaN or inf anywhere.
    finite = np.isfinite(pattern)
    if not finite.all():
        problems.append(f"{int((~finite).sum())} non-finite values (NaN or inf)")

    # NaN compares False with everything and poisons sums, so the remaining checks run on a
    # copy with non-finite cells zeroed. They are already reported by check 3.
    safe = np.where(finite, pattern, 0.0)

    # Check 4: softmax outputs are never negative (small float error is allowed).
    if (safe < -atol).any():
        problems.append(f"negative attention weights (min {safe.min():.3g})")

    # Check 5: causal. Cells above the diagonal (source k later than destination q) must be 0.
    # This also catches a transposed [k, q] pattern, which would be upper-triangular.
    # The 2-D mask applies to every batch and head at once via `...` indexing.
    future = np.triu(np.ones((seq_dest, seq_src), dtype=bool), k=1)
    leak = np.abs(safe[..., future]).max(initial=0.0)
    if leak > atol:
        problems.append(f"attention to future positions (k > q), max {leak:.3g}")

    # Check 6: each destination row is a probability distribution over sources, so it sums
    # to 1. Sum over axis -1 (source); column sums (axis -2) need not be 1.
    row_error = np.abs(safe.sum(axis=-1) - 1.0).max()
    if row_error > atol:
        problems.append(f"rows do not sum to 1 (max deviation {row_error:.3g})")

    # Report checks 3–6 together.
    if problems:
        raise AttentionPatternError(problems)
