"""Checkpoint 8: controls. Does each score separate the heads it should?

Earlier tests showed each mask picks the intended cells. These ask the scientific question:
does a head that does the behaviour score clearly higher than heads that do not, and than a
no-preference (uniform) baseline? Scores are not mutually exclusive, so the expectation is a
dominant diagonal, not a 0/1 identity matrix.

Head-side controls only. The input-side control (same head, non-repeated input: a real
induction head should drop, a previous-token head should not) needs a model that reacts to
its input; hand-written patterns are fixed. That control runs on GPT-2 in M1 Phase 3.
"""

import numpy as np
import pytest
from attention_helpers import (
    make_valid_pattern,
    mix,
    previous_token_head,
    repeated_head,
    selected,
    sink_head,
    uniform_head,
)

from cirqloom_analysis.detectors.registry import default_registry
from cirqloom_analysis.detectors.scoring import masked_head_mean

REGISTRY = default_registry()
DETECTORS = ["previous_token_attention", "first_token_attention", "induction_attention"]
HEADS = ["previous_token", "sink", "induction"]  # same order: head i matches detector i


def heads_for(repeat_len: int) -> dict[str, np.ndarray]:
    seq = 1 + 2 * repeat_len
    return {
        "previous_token": previous_token_head(seq),
        "sink": sink_head(seq),
        "induction": repeated_head(repeat_len, offset=1),
    }


def score(pattern: np.ndarray, detector_id: str) -> float:
    spec = REGISTRY.get(detector_id)
    return masked_head_mean(pattern, spec.mask(pattern.shape[-1])).per_example.item()


def matrix(repeat_len: int) -> np.ndarray:
    """[head, detector] scores for the ideal heads."""
    heads = heads_for(repeat_len)
    return np.array([[score(heads[h], d) for d in DETECTORS] for h in HEADS])


# --- Discrimination matrix -------------------------------------------------------------------


def test_matrix_for_l3_matches_hand_calculation():
    # The table worked out by hand in the checkpoint 8 explanation (L=3, burn_in=2).
    expected = np.array(
        [
            [1.0, 0.0, 0.0],  # previous-token head
            [0.0, 1.0, 0.0],  # sink head
            [0.0, 0.4, 1.0],  # induction head: idle first-copy rows sink to BOS
        ]
    )
    np.testing.assert_allclose(matrix(3), expected)


@pytest.mark.parametrize("repeat_len", [3, 5, 10, 50])
def test_each_detector_ranks_its_own_head_first(repeat_len):
    # Per column: the matching head beats every other head. Compared within one detector
    # only; scores of different detectors are not on a common scale (AGENTS.md §5).
    m = matrix(repeat_len)
    for d in range(len(DETECTORS)):
        own = m[d, d]
        assert own == 1.0
        assert all(m[h, d] < own for h in range(len(HEADS)) if h != d)


@pytest.mark.parametrize("repeat_len", [3, 5, 10, 50])
def test_each_ideal_head_beats_the_uniform_baseline(repeat_len):
    uniform = uniform_head(1 + 2 * repeat_len)
    m = matrix(repeat_len)
    for d, detector_id in enumerate(DETECTORS):
        assert m[d, d] > score(uniform, detector_id)


# --- Off-diagonal cells that are expected to be non-zero -----------------------------------


@pytest.mark.parametrize("repeat_len", [3, 5, 10, 50])
def test_induction_head_bos_score_is_its_idle_fraction(repeat_len):
    # Rows 2..L (first copy, nothing to retrieve yet) sink to BOS; rows L+1..2L do not.
    # So the BOS score is (L-1)/(2L-1), approaching 0.5. Not evidence of a "BOS head".
    expected = (repeat_len - 1) / (2 * repeat_len - 1)
    assert score(heads_for(repeat_len)["induction"], "first_token_attention") == pytest.approx(
        expected
    )


@pytest.mark.parametrize("repeat_len", [3, 5, 10, 50])
def test_induction_head_does_not_score_as_previous_token(repeat_len):
    # Induction depends on an *earlier* previous-token head, but its own attention goes to
    # k = q - L + 1, never q - 1 when L >= 3.
    assert score(heads_for(repeat_len)["induction"], "previous_token_attention") == 0.0


# --- Uniform baseline ------------------------------------------------------------------------


def expected_uniform(detector_id: str, seq: int) -> float:
    # Row q of a uniform head puts 1/(q+1) on every allowed source.
    cells = selected(REGISTRY.get(detector_id).mask(seq))
    return float(np.mean([1 / (q + 1) for q, _ in cells]))


@pytest.mark.parametrize("repeat_len", [3, 10, 50])
@pytest.mark.parametrize("detector_id", DETECTORS)
def test_uniform_baseline_matches_formula(detector_id, repeat_len):
    seq = 1 + 2 * repeat_len
    assert score(uniform_head(seq), detector_id) == pytest.approx(
        expected_uniform(detector_id, seq)
    )


def test_uniform_baseline_for_l3_values():
    seq = 7
    assert score(uniform_head(seq), "previous_token_attention") == pytest.approx(0.2186, abs=1e-4)
    assert score(uniform_head(seq), "first_token_attention") == pytest.approx(0.2186, abs=1e-4)
    assert score(uniform_head(seq), "induction_attention") == pytest.approx(0.1698, abs=1e-4)


@pytest.mark.parametrize("detector_id", DETECTORS)
def test_uniform_baseline_shrinks_with_length(detector_id):
    # Chance level depends on seq: a score must be read against the baseline of its own
    # input length, and one fixed threshold cannot fit every length or detector (§15).
    baselines = [score(uniform_head(1 + 2 * L), detector_id) for L in (3, 10, 50)]
    assert baselines[0] > baselines[1] > baselines[2]


def test_baselines_differ_between_detectors():
    seq = 7
    assert score(uniform_head(seq), "induction_attention") != pytest.approx(
        score(uniform_head(seq), "previous_token_attention")
    )


# --- Mixed heads: scores are linear in the pattern -----------------------------------------


def test_mixed_head_example_from_explanation():
    # 70% induction + 30% sink, L=3: worked out by hand as 0.70 / 0.58 / 0.
    heads = heads_for(3)
    head = mix([heads["induction"], heads["sink"]], [0.7, 0.3])
    assert score(head, "induction_attention") == pytest.approx(0.70)
    assert score(head, "first_token_attention") == pytest.approx(0.58)
    assert score(head, "previous_token_attention") == 0.0


@pytest.mark.parametrize("detector_id", DETECTORS)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_score_of_mixture_is_weighted_sum(detector_id, seed):
    # A masked mean only adds and scales, so score(a·P + b·Q) = a·score(P) + b·score(Q).
    # This lets a mixed head's score be predicted from its parts.
    seq = 1 + 2 * 6
    rng = np.random.default_rng(seed)
    p = make_valid_pattern(1, 1, seq, seed=seed)
    q = make_valid_pattern(1, 1, seq, seed=seed + 100)
    a = float(rng.random())
    mixed = mix([p, q], [a, 1 - a])
    assert score(mixed, detector_id) == pytest.approx(
        a * score(p, detector_id) + (1 - a) * score(q, detector_id)
    )


@pytest.mark.parametrize("weight", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_induction_score_tracks_induction_share(weight):
    # Dilute an induction head with a sink: the induction score equals the induction share.
    heads = heads_for(10)
    head = mix([heads["induction"], heads["sink"]], [weight, 1 - weight])
    assert score(head, "induction_attention") == pytest.approx(weight)
