"""Attention between two named token positions (M3 step 4, IOI prompts).

On a text suite whose prompts share named positions (e.g. IOI: IO, S1, S1+1, S2, END), the
score is the attention weight from one position (query row) to another (key column), per
prompt, averaged over prompts. This is an observed attention pattern. It does not show what the
head writes, whether it copies, or that it has the role a paper gives it (docs/spec.md §2, §7.6).
"""

import numpy as np

from cirqloom_analysis.detectors.masks import check_integer


def position_pair_mask(seq: int, *, query: int, key: int) -> np.ndarray:
    """Boolean [seq, seq] mask with the single cell (query, key); key must not be in the future."""
    check_integer("query", query)
    check_integer("key", key)
    if not 0 <= key <= query < seq:
        raise ValueError(f"need 0 <= key <= query < seq, got key={key}, query={query}, seq={seq}")
    mask = np.zeros((seq, seq), dtype=bool)
    mask[query, key] = True
    return mask
