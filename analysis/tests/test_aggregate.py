"""Checkpoint 6: aggregating per-example scores into per-head summaries."""

import numpy as np
import pytest
from attention_helpers import make_valid_pattern, repeated_head

from cirqloom_analysis.detectors.aggregate import summarize
from cirqloom_analysis.detectors.induction import induction_mask
from cirqloom_analysis.detectors.scoring import MaskedScore, masked_head_mean

# The 4 examples × 3 heads table from the checkpoint 6 explanation.
TABLE = np.array(
    [
        [0.91, 0.05, 0.12],
        [0.88, 0.07, 0.40],
        [0.93, 0.04, 0.02],
        [0.90, 0.06, 0.31],
    ]
)


def batch(per_example, support: int = 50) -> MaskedScore:
    return MaskedScore(per_example=np.asarray(per_example, dtype=float), support=support)


def test_mean_std_and_n_per_head():
    summary = summarize([batch(TABLE)])
    np.testing.assert_allclose(summary.mean, TABLE.mean(axis=0))
    np.testing.assert_allclose(summary.std, TABLE.std(axis=0, ddof=1))
    assert summary.n == 4
    assert summary.mean.shape == summary.std.shape == (3,)


def test_consistent_head_has_smaller_spread_than_erratic_head():
    # Same idea as the explanation: head 0 is steady, head 2 jumps around.
    summary = summarize([batch(TABLE)])
    assert summary.std[0] < 0.05 < summary.std[2]


def test_n_counts_examples_not_cells():
    # 6 prompts with 50 induction cells each: n is 6, not 300.
    summary = summarize([batch(np.full((6, 2), 0.5), support=50)])
    assert summary.n == 6
    assert summary.support.tolist() == [50] * 6


def test_batches_are_pooled_with_equal_weight_per_example():
    # L=20 batch scores 0.0, L=50 batch scores 1.0, four examples each.
    short = batch(np.zeros((4, 1)), support=20)
    long = batch(np.ones((4, 1)), support=50)
    summary = summarize([short, long])
    assert summary.n == 8
    assert summary.mean.item() == pytest.approx(0.5)  # not (20*0 + 50*1)/70 ≈ 0.71
    assert summary.support.tolist() == [20] * 4 + [50] * 4


def test_per_example_scores_are_kept_in_order():
    a, b = batch(TABLE[:2]), batch(TABLE[2:])
    np.testing.assert_array_equal(summarize([a, b]).per_example, TABLE)


def test_single_example_has_no_std():
    summary = summarize([batch([[0.7, 0.2]])])
    assert summary.n == 1
    assert summary.std is None  # undefined, not 0.0 and not NaN
    np.testing.assert_allclose(summary.mean, [0.7, 0.2])


def test_two_examples_have_a_std():
    summary = summarize([batch([[0.0], [1.0]])])
    assert summary.std is not None
    assert summary.std.item() == pytest.approx(np.sqrt(0.5))


def test_rejects_empty_input():
    with pytest.raises(ValueError, match="no scores"):
        summarize([])


def test_rejects_batches_with_no_examples():
    with pytest.raises(ValueError, match="no examples"):
        summarize([batch(np.empty((0, 3)))])


def test_rejects_mismatched_head_counts():
    with pytest.raises(ValueError, match="batch 1"):
        summarize([batch(np.zeros((2, 12))), batch(np.zeros((2, 11)))])


def test_rejects_non_finite_scores():
    with pytest.raises(ValueError, match="NaN"):
        summarize([batch([[0.5, np.nan]])])


def test_end_to_end_with_real_masks_and_different_lengths():
    # Two repeat lengths, ideal induction head on both → mean 1.0, spread 0, n = 2 examples.
    scores = []
    for repeat_len in (3, 6):
        pattern = repeated_head(repeat_len, offset=1)
        scores.append(masked_head_mean(pattern, induction_mask(repeat_len)))
    summary = summarize(scores)
    assert summary.n == 2
    assert summary.mean.item() == 1.0
    assert summary.std.item() == 0.0
    assert summary.support.tolist() == [3, 6]


def test_end_to_end_random_patterns_are_finite():
    seq = 1 + 2 * 5
    mask = induction_mask(5)
    scores = [masked_head_mean(make_valid_pattern(3, 12, seq, seed=s), mask) for s in range(3)]
    summary = summarize(scores)
    assert summary.n == 9
    assert summary.mean.shape == (12,)
    assert np.isfinite(summary.mean).all() and np.isfinite(summary.std).all()
