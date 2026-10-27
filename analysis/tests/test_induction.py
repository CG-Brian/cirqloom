"""Checkpoint 5: induction mask on repeated random tokens."""

import numpy as np
import pytest
from attention_helpers import (
    make_valid_pattern,
    previous_token_head,
    repeated_head,
    selected,
    sink_head,
    uniform_head,
)

from cirqloom_analysis.detectors.induction import MIN_REPEAT_LEN, induction_mask
from cirqloom_analysis.detectors.scoring import masked_head_mean


def score(pattern: np.ndarray, repeat_len: int) -> float:
    return masked_head_mean(pattern, induction_mask(repeat_len)).per_example.item()


# --- Which cells are selected ---------------------------------------------------------------


def test_hand_worked_example_l3():
    # The table filled in during the design: q=4,5,6 → k=2,3,4.
    assert selected(induction_mask(3)) == [(4, 2), (5, 3), (6, 4)]


@pytest.mark.parametrize("repeat_len", [3, 4, 7, 50])
def test_cells_follow_k_equals_q_minus_l_plus_1(repeat_len):
    cells = selected(induction_mask(repeat_len))
    assert [q for q, _ in cells] == list(range(repeat_len + 1, 2 * repeat_len + 1))  # 2nd copy
    assert all(k == q - repeat_len + 1 for q, k in cells)


@pytest.mark.parametrize("repeat_len", [3, 5, 50])
def test_never_selects_duplicate_token_cells(repeat_len):
    # Guards the missing "+1": k = q - L is the duplicate-token head, one column to the left.
    assert not any(k == q - repeat_len for q, k in selected(induction_mask(repeat_len)))


@pytest.mark.parametrize("repeat_len", [3, 5, 50])
def test_excludes_first_copy_part_of_the_diagonal(repeat_len):
    # diagonal(1 - L) also passes through (L-1, 0) (sink) and (L, 1) (first copy).
    mask = induction_mask(repeat_len)
    assert not mask[repeat_len - 1, 0]
    assert not mask[repeat_len, 1]
    full_diagonal = np.eye(1 + 2 * repeat_len, k=1 - repeat_len, dtype=bool)
    assert full_diagonal.sum() == repeat_len + 2  # the two cells above are really on it
    assert mask.sum() == repeat_len


@pytest.mark.parametrize("repeat_len", [3, 5, 50])
def test_never_selects_column_0_or_1(repeat_len):
    # k = q - L + 1 >= 2 for every second-copy row: no sink cell, no first-copy X_0 cell.
    assert not induction_mask(repeat_len)[:, :2].any()


# --- Shape, dtype and the masked_head_mean contract ----------------------------------------


@pytest.mark.parametrize("repeat_len", [3, 4, 20])
def test_shape_dtype_and_support(repeat_len):
    seq = 1 + 2 * repeat_len
    mask = induction_mask(repeat_len)
    assert mask.shape == (seq, seq)
    assert mask.dtype == bool
    result = masked_head_mean(make_valid_pattern(2, 3, seq), mask)
    assert result.support == repeat_len


def test_wrong_length_pattern_is_caught_by_scoring():
    # The mask assumes seq = 1 + 2L; a pattern of another length fails loudly, not silently.
    with pytest.raises(ValueError, match="does not match"):
        masked_head_mean(make_valid_pattern(1, 1, 8), induction_mask(3))


# --- Input validation ------------------------------------------------------------------------


def test_minimum_repeat_len_is_accepted():
    assert MIN_REPEAT_LEN == 3
    induction_mask(3)


@pytest.mark.parametrize("repeat_len", [2, 1, 0, -4])
def test_short_repeat_len_is_rejected(repeat_len):
    with pytest.raises(ValueError, match=rf"repeat_len must be >= 3.*got {repeat_len}"):
        induction_mask(repeat_len)


@pytest.mark.parametrize("repeat_len", [3.0, True, "3", None])
def test_non_integer_repeat_len_is_rejected(repeat_len):
    with pytest.raises(TypeError, match="repeat_len must be an integer"):
        induction_mask(repeat_len)  # type: ignore[arg-type]


def test_accepts_numpy_integer():
    assert induction_mask(np.int64(4)).shape == (9, 9)


# --- Why L >= 3: smaller L collides with other detectors ------------------------------------


def test_l2_would_coincide_with_previous_token_cells():
    # Documents the reason for MIN_REPEAT_LEN: with L=2, k = q - 2 + 1 = q - 1.
    l2_cells = np.eye(5, k=-1, dtype=bool)
    l2_cells[:3] = False
    assert selected(l2_cells) == [(3, 2), (4, 3)]  # exactly previous-token cells


# --- Scientific behaviour: does the score separate the right heads? ------------------------


@pytest.mark.parametrize("repeat_len", [3, 6, 25])
def test_ideal_induction_head_scores_one(repeat_len):
    assert score(repeated_head(repeat_len, offset=1), repeat_len) == 1.0


@pytest.mark.parametrize("repeat_len", [3, 6, 25])
def test_duplicate_token_head_scores_zero(repeat_len):
    # One column off from induction: the off-by-one this detector must not confuse.
    assert score(repeated_head(repeat_len, offset=0), repeat_len) == 0.0


@pytest.mark.parametrize("repeat_len", [3, 6, 25])
def test_previous_token_and_sink_heads_score_zero(repeat_len):
    seq = 1 + 2 * repeat_len
    assert score(previous_token_head(seq), repeat_len) == 0.0
    assert score(sink_head(seq), repeat_len) == 0.0


def test_full_diagonal_mean_would_reward_a_sink_head():
    # Why the second-copy mask matters: averaging the whole diagonal(1 - L) gives a pure sink
    # head (no induction behaviour at all) a free point from the (L-1, 0) cell.
    repeat_len = 3
    full_diagonal = np.eye(1 + 2 * repeat_len, k=1 - repeat_len, dtype=bool)
    naive = masked_head_mean(sink_head(1 + 2 * repeat_len), full_diagonal).per_example.item()
    assert naive == pytest.approx(1 / (repeat_len + 2))
    assert score(sink_head(1 + 2 * repeat_len), repeat_len) == 0.0


def test_uniform_head_baseline_is_small_and_nonzero():
    repeat_len = 10
    baseline = score(uniform_head(1 + 2 * repeat_len), repeat_len)
    assert 0 < baseline < 0.1
