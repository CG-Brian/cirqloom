"""Checkpoint 1: attention tensor contract."""

import numpy as np
import pytest
from attention_helpers import make_valid_pattern

from cirqloom_analysis.detectors.attention import (
    AttentionPatternError,
    validate_attention_pattern,
)


def test_valid_pattern_passes():
    validate_attention_pattern(make_valid_pattern(2, 3, 5))


def test_single_token_sequence_passes():
    # seq = 1 has no future cells at all; the only row is [1.0].
    validate_attention_pattern(np.ones((1, 1, 1, 1)))


def test_tolerates_float32_rounding():
    pattern = make_valid_pattern(2, 2, 64).astype(np.float32)
    validate_attention_pattern(pattern)


def test_rejects_wrong_ndim_and_reports_shape():
    with pytest.raises(AttentionPatternError, match=r"\(3, 5, 5\)"):
        validate_attention_pattern(np.zeros((3, 5, 5)))


def test_rejects_non_square():
    with pytest.raises(AttentionPatternError, match="differ"):
        validate_attention_pattern(np.zeros((1, 1, 4, 5)))


def test_rejects_future_attention():
    pattern = make_valid_pattern(1, 1, 3)
    pattern[0, 0, 1] = [0.4, 0.5, 0.1]  # row q=1 puts 0.1 on k=2 (the future); sum is still 1
    with pytest.raises(AttentionPatternError, match="future"):
        validate_attention_pattern(pattern)


def test_rejects_transposed_pattern():
    # A [k, q] pattern is upper-triangular, so it leaks into the future (and its rows break).
    pattern = make_valid_pattern(1, 2, 4).transpose(0, 1, 3, 2)
    with pytest.raises(AttentionPatternError, match="future"):
        validate_attention_pattern(pattern)


def test_rejects_bad_row_sum():
    pattern = make_valid_pattern(1, 1, 3)
    pattern[0, 0, 1] = [0.5, 0.4, 0.0]  # sums to 0.9
    with pytest.raises(AttentionPatternError, match="sum to 1"):
        validate_attention_pattern(pattern)


def test_rejects_negative_weights():
    pattern = make_valid_pattern(1, 1, 3)
    pattern[0, 0, 2] = [1.2, -0.2, 0.0]  # sums to 1 but has a negative weight
    with pytest.raises(AttentionPatternError, match="negative"):
        validate_attention_pattern(pattern)


def test_rejects_nan():
    pattern = make_valid_pattern(1, 2, 4)
    pattern[0, 1, 2, 0] = np.nan
    with pytest.raises(AttentionPatternError, match="non-finite") as exc_info:
        validate_attention_pattern(pattern)
    # The NaN cell is zeroed for the later checks, so its row also fails the sum check.
    # No other noise (e.g. a false "future" or "negative" report) should appear.
    assert len(exc_info.value.problems) == 2  # non-finite + that row no longer sums to 1


def test_reports_all_problems_at_once():
    pattern = make_valid_pattern(1, 1, 3)
    pattern[0, 0, 1] = [0.5, 0.4, 0.3]  # future leak AND sums to 1.2
    with pytest.raises(AttentionPatternError) as exc_info:
        validate_attention_pattern(pattern)
    assert len(exc_info.value.problems) == 2


def test_error_message_lists_problems():
    with pytest.raises(AttentionPatternError) as exc_info:
        validate_attention_pattern(np.zeros((2, 2)))
    assert "Invalid attention pattern (1 problems)" in str(exc_info.value)
