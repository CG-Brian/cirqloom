"""Previous-token attention mask (checkpoint 3, AGENTS.md §7.2).

A previous-token head puts its attention on source k = q - 1. This is a positional
attention pattern score, not causal proof.
"""

import numpy as np

from cirqloom_analysis.detectors.masks import DEFAULT_BURN_IN, check_burn_in


def previous_token_mask(seq: int, *, burn_in: int = DEFAULT_BURN_IN) -> np.ndarray:
    """Boolean [seq, seq] mask that is True on the previous-token cells (q, q - 1), q >= burn_in.

    Design (checkpoint 3, burn-in unified in checkpoint 4):
      1. Validate seq and burn_in with the shared rule (masks.check_burn_in).
         seq is a dimension size (e.g. pattern.shape[-1]), so its own type is not checked.
      2. True on the diagonal just below the main diagonal (k = q - 1).
      3. Drop the first `burn_in` destination rows. The default (2) drops (1, 0): position 0 is
         an attention sink for many heads, so a sink head would otherwise score there as if it
         were a previous-token head. burn_in=1 keeps (1, 0); row 0 has no cell anyway.

    The result always satisfies masked_head_mean's mask rules: bool, (seq, seq), non-empty,
    and no future cells.
    """
    # Step 1: fail early with the real cause. Without this, a too-short seq would produce an
    # empty mask and only fail later in masked_head_mean ("mask selects no cells").
    check_burn_in(seq, burn_in)

    # Step 2: k=-1 puts True one below the main diagonal, i.e. cells (q, q - 1).
    # dtype=bool is required: masked_head_mean rejects numeric masks.
    mask = np.eye(seq, k=-1, dtype=bool)

    # Step 3: switch off the burned-in destination rows.
    mask[:burn_in] = False

    return mask
