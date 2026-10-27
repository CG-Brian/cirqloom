"""Checkpoint 4: first-token (position 0) mask."""

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

from cirqloom_analysis.detectors.first_token import first_token_mask
from cirqloom_analysis.detectors.previous_token import previous_token_mask
from cirqloom_analysis.detectors.scoring import masked_head_mean


def score(pattern: np.ndarray, **kwargs) -> float:
    mask = first_token_mask(pattern.shape[-1], **kwargs)
    return masked_head_mean(pattern, mask).per_example.item()


# --- Which cells are selected ---------------------------------------------------------------


def test_default_selects_column_0_from_row_2():
    assert selected(first_token_mask(5)) == [(2, 0), (3, 0), (4, 0)]


def test_burn_in_1_keeps_cell_1_0():
    assert selected(first_token_mask(5, burn_in=1)) == [(1, 0), (2, 0), (3, 0), (4, 0)]


def test_only_column_0_is_ever_selected():
    for burn_in in (1, 2, 4):
        assert all(k == 0 for _, k in selected(first_token_mask(9, burn_in=burn_in)))


def test_never_selects_cell_0_0():
    for burn_in in (1, 2):
        assert not first_token_mask(6, burn_in=burn_in)[0, 0]


# --- Shape, dtype and the masked_head_mean contract ----------------------------------------


@pytest.mark.parametrize("seq", [3, 4, 17])
def test_shape_and_dtype(seq):
    mask = first_token_mask(seq)
    assert mask.shape == (seq, seq)
    assert mask.dtype == bool


@pytest.mark.parametrize("seq", [3, 4, 10])
@pytest.mark.parametrize("burn_in", [1, 2])
def test_same_support_as_previous_token(seq, burn_in):
    # Shared burn-in → same destination rows → scores can be compared side by side.
    pattern = make_valid_pattern(2, 3, seq)
    first = masked_head_mean(pattern, first_token_mask(seq, burn_in=burn_in))
    prev = masked_head_mean(pattern, previous_token_mask(seq, burn_in=burn_in))
    assert first.support == prev.support == seq - burn_in


def test_invalid_input_uses_the_shared_rule():
    with pytest.raises(ValueError, match="needs seq >= 3, got 2"):
        first_token_mask(2)
    with pytest.raises(ValueError):
        first_token_mask(5, burn_in=0)
    with pytest.raises(TypeError):
        first_token_mask(5, burn_in=True)  # type: ignore[arg-type]


# --- Scientific behaviour --------------------------------------------------------------------


def test_sink_head_scores_one():
    assert score(sink_head(6)) == 1.0


def test_previous_token_head_scores_zero_by_default():
    # Row 1's previous token is position 0, so with burn_in=1 a previous-token head gets a
    # free point on (1, 0): the mirror image of the sink leak in checkpoint 3.
    assert score(previous_token_head(4)) == 0.0
    assert score(previous_token_head(4), burn_in=1) == pytest.approx(1 / 3)


def test_self_attention_head_scores_zero():
    assert score(self_head(6)) == 0.0


def test_uniform_baseline_matches_previous_token_baseline():
    # A uniform row q gives 1/(q+1) to every source, so position 0 and q-1 get the same weight:
    # both detectors share a control baseline at the same burn-in.
    seq = 8
    first = score(uniform_head(seq))
    prev = masked_head_mean(uniform_head(seq), previous_token_mask(seq)).per_example.item()
    assert first == pytest.approx(prev)
    assert 0 < first < 0.5
