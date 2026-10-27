"""First-token (position 0) attention mask (checkpoint 4, AGENTS.md §7.3).

Many heads dump attention on position 0 when they have nothing relevant to attend to
(an *attention sink*). Position 0 is the only source every row can see, and its value vector
tends to carry little, so heavy attention there often means the head is idle on this input.

This is a BOS/first-position attention score. It may reflect an attention sink rather than a
dedicated semantic circuit: do not infer an interpretation from the score alone.
"""

import numpy as np

from cirqloom_analysis.detectors.masks import DEFAULT_BURN_IN, check_burn_in


def first_token_mask(seq: int, *, burn_in: int = DEFAULT_BURN_IN) -> np.ndarray:
    """Boolean [seq, seq] mask that is True on cells (q, 0) for q >= burn_in.

    Design (checkpoint 4):
      1. Validate seq and burn_in with the shared rule (masks.check_burn_in).
      2. True on column 0 (source = position 0) for every destination row ...
      3. ... except the first `burn_in` rows. Row 0 is 1.0 for every head. At the default (2),
         row 1 is dropped too: a previous-token head also puts all its weight on (1, 0).

    Uses the same burn-in as previous_token_mask, so both scores cover the same destination
    rows and have the same support.
    """
    # Step 1: same validation as every mask builder.
    check_burn_in(seq, burn_in)

    # Steps 2–3: column 0, starting at row burn_in.
    mask = np.zeros((seq, seq), dtype=bool)
    mask[burn_in:, 0] = True

    return mask
