"""Checkpoint 3: previous-token mask (burn-in unified in checkpoint 4)."""

import numpy as np
import pytest
from attention_helpers import (
    make_valid_pattern,
    previous_token_head,
    selected,
    self_head,
    sink_head,
    uniform_head,
)

from cirqloom_analysis.detectors.previous_token import previous_token_mask
from cirqloom_analysis.detectors.scoring import masked_head_mean


def score(pattern: np.ndarray, **kwargs) -> float:
    mask = previous_token_mask(pattern.shape[-1], **kwargs)
    return masked_head_mean(pattern, mask).per_example.item()


# --- Which cells are selected ---------------------------------------------------------------


def test_default_burn_in_drops_cell_1_0():
    assert selected(previous_token_mask(5)) == [(2, 1), (3, 2), (4, 3)]


def test_burn_in_1_keeps_cell_1_0():
    assert selected(previous_token_mask(5, burn_in=1)) == [(1, 0), (2, 1), (3, 2), (4, 3)]


def test_larger_burn_in_drops_more_rows():
    assert selected(previous_token_mask(5, burn_in=3)) == [(3, 2), (4, 3)]


def test_every_cell_is_exactly_one_step_back():
    # Guards the k=+1 vs k=-1 bug: every selected source must be q - 1, never q + 1 or q.
    for burn_in in (1, 2, 3):
        cells = selected(previous_token_mask(8, burn_in=burn_in))
        assert all(k == q - 1 for q, k in cells)


def test_never_selects_cell_0_0():
    for burn_in in (1, 2):
        assert not previous_token_mask(6, burn_in=burn_in)[0, 0]


# --- Shape, dtype and the masked_head_mean contract ----------------------------------------


@pytest.mark.parametrize("seq", [3, 4, 17])
def test_shape_follows_seq(seq):
    # Guards the hard-coded np.eye(3, ...) bug: the mask must grow with seq.
    assert previous_token_mask(seq).shape == (seq, seq)


def test_dtype_is_bool():
    assert previous_token_mask(4).dtype == bool


@pytest.mark.parametrize("seq", [3, 4, 10])
@pytest.mark.parametrize("burn_in", [1, 2])
def test_always_accepted_by_masked_head_mean(seq, burn_in):
    # bool, (seq, seq), non-empty and no future cells: the mask rules from checkpoint 2.
    mask = previous_token_mask(seq, burn_in=burn_in)
    result = masked_head_mean(make_valid_pattern(2, 3, seq), mask)
    assert result.support == seq - burn_in  # one cell per row q = burn_in .. seq-1


# --- Minimum length and options --------------------------------------------------------------


def test_minimum_seq_is_accepted():
    # The minimum itself is valid (guards the <= vs < off-by-one).
    assert selected(previous_token_mask(3)) == [(2, 1)]
    assert selected(previous_token_mask(2, burn_in=1)) == [(1, 0)]


@pytest.mark.parametrize(("seq", "burn_in"), [(2, 2), (1, 1)])
def test_too_short_is_rejected_with_the_real_cause(seq, burn_in):
    with pytest.raises(ValueError, match=rf"needs seq >= {burn_in + 1}, got {seq}"):
        previous_token_mask(seq, burn_in=burn_in)


def test_invalid_burn_in_is_rejected():
    with pytest.raises(ValueError):
        previous_token_mask(5, burn_in=0)
    with pytest.raises(TypeError):
        previous_token_mask(5, burn_in=2.0)  # type: ignore[arg-type]


def test_burn_in_is_keyword_only():
    # A bare positional 2 would hide what it means at the call site.
    with pytest.raises(TypeError):
        previous_token_mask(5, 2)  # type: ignore[misc]


# --- Scientific behaviour: does the score separate the right heads? ------------------------


def test_ideal_previous_token_head_scores_one():
    assert score(previous_token_head(6)) == 1.0


def test_self_attention_head_scores_zero():
    # Every row attends to itself; including (0, 0) would have given it a free point.
    assert score(self_head(6)) == 0.0


def test_sink_head_scores_zero_by_default():
    # Not a previous-token head, so the default must give 0. burn_in=1 shows the (1, 0) leak.
    assert score(sink_head(3)) == 0.0
    assert score(sink_head(3), burn_in=1) == 0.5


def test_uniform_head_has_a_nonzero_baseline():
    # A head with no preference still scores above 0: A[q, q-1] = 1 / (q + 1).
    # Scores need a control baseline before being interpreted.
    assert score(uniform_head(4)) == pytest.approx((1 / 3 + 1 / 4) / 2)
