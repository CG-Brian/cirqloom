"""Checkpoint 2: masked head mean."""

import numpy as np
import pytest
from attention_helpers import make_valid_pattern

from cirqloom_analysis.detectors.attention import AttentionPatternError
from cirqloom_analysis.detectors.scoring import masked_head_mean


def cells(seq: int, *qk: tuple[int, int]) -> np.ndarray:
    """A [seq, seq] bool mask with True at the given (q, k) cells."""
    mask = np.zeros((seq, seq), dtype=bool)
    for q, k in qk:
        mask[q, k] = True
    return mask


def test_hand_computed_mean():
    pattern = np.array([[[[1.0, 0.0, 0.0], [0.7, 0.3, 0.0], [0.1, 0.6, 0.3]]]])
    result = masked_head_mean(pattern, cells(3, (1, 0), (2, 1)))  # previous-token cells
    np.testing.assert_allclose(result.per_example, [[0.65]])  # (0.7 + 0.6) / 2


def test_output_shape_is_batch_by_head():
    result = masked_head_mean(make_valid_pattern(2, 3, 5), cells(5, (4, 0)))
    assert result.per_example.shape == (2, 3)


def test_heads_and_examples_are_scored_independently():
    pattern = make_valid_pattern(2, 3, 4)
    mask = cells(4, (3, 1))
    before = masked_head_mean(pattern, mask).per_example

    changed = pattern.copy()
    changed[1, 2, 3] = [0.0, 1.0, 0.0, 0.0]  # example 1, head 2: all attention on the masked cell
    after = masked_head_mean(changed, mask).per_example

    assert after[1, 2] == pytest.approx(1.0)
    untouched = np.ones((2, 3), dtype=bool)
    untouched[1, 2] = False
    np.testing.assert_array_equal(after[untouched], before[untouched])


def test_support_counts_mask_cells():
    mask = cells(5, (1, 0), (2, 1), (3, 2), (4, 3))
    assert masked_head_mean(make_valid_pattern(1, 1, 5), mask).support == 4


def test_rejects_empty_mask():
    with pytest.raises(ValueError, match="no cells"):
        masked_head_mean(make_valid_pattern(1, 1, 3), np.zeros((3, 3), dtype=bool))


def test_rejects_future_cells_in_mask():
    # (1, 2) is k > q: an off-by-one in the other direction.
    with pytest.raises(ValueError, match="future"):
        masked_head_mean(make_valid_pattern(1, 1, 3), cells(3, (2, 1), (1, 2)))


def test_rejects_mask_shape_mismatch():
    with pytest.raises(ValueError, match="does not match"):
        masked_head_mean(make_valid_pattern(1, 1, 5), cells(4, (1, 0)))


def test_rejects_non_bool_mask():
    # As ints, [[0, 0], [1, 0]] would be read as row indices by numpy's fancy indexing.
    with pytest.raises(ValueError, match="bool"):
        masked_head_mean(make_valid_pattern(1, 1, 2), cells(2, (1, 0)).astype(int))


def test_invalid_pattern_is_rejected_first():
    transposed = make_valid_pattern(1, 1, 3).transpose(0, 1, 3, 2)
    with pytest.raises(AttentionPatternError):
        masked_head_mean(transposed, cells(3, (1, 0)))
