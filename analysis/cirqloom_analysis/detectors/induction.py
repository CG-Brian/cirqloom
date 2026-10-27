"""Induction-like attention mask on repeated random tokens (checkpoint 5, AGENTS.md §7.4).

An induction head completes [A][B] ... [A] → [B]: at the second A it attends to the token
*after* the earlier A. It relies on an earlier previous-token head (K-composition).

Input layout this mask assumes (the prompt generator must produce exactly this):

    position: 0    1 .. L                 L+1 .. 2L
    token:    BOS  X_0 .. X_{L-1}         X_0 .. X_{L-1}       seq = 1 + 2L

For a second-copy destination q (L+1 <= q <= 2L), the same token appeared at q - L, so the
induction source is the next position: k = q - L + 1.

Only second-copy rows are used. The full diagonal(1 - L) also contains (L-1, 0), an
attention-sink cell, and (L, 1), a first-copy cell with no earlier occurrence; averaging
the whole diagonal would mix those in (AGENTS.md §15).

This measures attention (QK) only. Whether the head actually copies the token (OV, logit
attribution) is a separate copy_support_score, and the label is induction_like_candidate.
"""

import numpy as np

from cirqloom_analysis.detectors.masks import check_integer

MIN_REPEAT_LEN = 3
"""Below 3 the induction cells coincide with another detector's cells:
L=1 gives k = q (self-attention), L=2 gives k = q - 1 (previous-token)."""


def induction_mask(repeat_len: int) -> np.ndarray:
    """Boolean [seq, seq] mask (seq = 1 + 2 * repeat_len) True on the induction cells.

    repeat_len is L, the length of each copy. The mask selects exactly L cells:
    (L+1, 2), (L+2, 3), ..., (2L, L+1).
    """
    # Step 1: repeat_len comes from a prompt-suite setting, so its type is checked.
    check_integer("repeat_len", repeat_len)
    if repeat_len < MIN_REPEAT_LEN:
        raise ValueError(
            f"repeat_len must be >= {MIN_REPEAT_LEN} so induction cells differ from "
            f"self-attention and previous-token cells, got {repeat_len}"
        )

    seq = 1 + 2 * repeat_len

    # Step 2: the diagonal k = q - L + 1, i.e. offset 1 - L below the main diagonal.
    mask = np.eye(seq, k=1 - repeat_len, dtype=bool)

    # Step 3: keep second-copy rows only. Rows 0..L are BOS and the first copy, where no
    # earlier occurrence exists. This is structural (fixed by L), not a tunable burn-in.
    mask[: repeat_len + 1] = False

    return mask


def induction_mask_for_seq(seq: int) -> np.ndarray:
    """induction_mask for a (BOS, X, X) input of length seq, so all detectors share build_mask(seq).

    repeat_len is recovered as (seq - 1) / 2. Arithmetic can only check that seq *could* be
    such an input: an odd-length natural sentence also passes. Checking that the content
    really repeats is the pipeline's job, via the spec's prompt_suite (checkpoint 7).
    """
    # Step 1: (BOS, X, X) always has odd length; an even seq cannot have this layout.
    if seq % 2 == 0:
        raise ValueError(f"induction input (BOS, X, X) needs odd seq = 1 + 2L, got {seq}")

    # Step 2: induction_mask checks the minimum length (repeat_len >= 3, i.e. seq >= 7).
    return induction_mask((seq - 1) // 2)
