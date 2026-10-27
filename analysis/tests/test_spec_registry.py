"""Checkpoint 7: detector specs and the registry."""

import dataclasses

import numpy as np
import pytest
from attention_helpers import previous_token_head, repeated_head, selected, sink_head

from cirqloom_analysis.detectors.induction import induction_mask, induction_mask_for_seq
from cirqloom_analysis.detectors.previous_token import previous_token_mask
from cirqloom_analysis.detectors.registry import DetectorRegistry, default_registry
from cirqloom_analysis.detectors.scoring import masked_head_mean
from cirqloom_analysis.detectors.spec import DetectorSpec

SPEC_FIELDS = dict(
    id="toy_detector",
    version="1.0.0",
    evidence_type="heuristic_attention",
    method_reference="https://example.org/method",
    description="Toy detector for tests.",
    prompt_suite="toy_suite",
    build_mask=previous_token_mask,
    default_parameters={"burn_in": 2},
)


def make_spec(**changes) -> DetectorSpec:
    return DetectorSpec(**{**SPEC_FIELDS, **changes})


# --- induction_mask_for_seq ------------------------------------------------------------------


@pytest.mark.parametrize("repeat_len", [3, 4, 50])
def test_seq_wrapper_matches_induction_mask(repeat_len):
    seq = 1 + 2 * repeat_len
    np.testing.assert_array_equal(induction_mask_for_seq(seq), induction_mask(repeat_len))


@pytest.mark.parametrize("seq", [8, 20, 102])
def test_seq_wrapper_rejects_even_seq(seq):
    # An even length cannot be (BOS, X, X); rounding L would silently shift every cell.
    with pytest.raises(ValueError, match=rf"odd seq = 1 \+ 2L, got {seq}"):
        induction_mask_for_seq(seq)


@pytest.mark.parametrize("seq", [1, 3, 5])
def test_seq_wrapper_rejects_too_short_seq(seq):
    with pytest.raises(ValueError, match="repeat_len must be >= 3"):
        induction_mask_for_seq(seq)


# --- DetectorSpec validation -----------------------------------------------------------------


def test_valid_spec_is_created():
    spec = make_spec()
    assert spec.id == "toy_detector"
    assert spec.maturity == "experimental"  # default: not validated on real model data


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"id": ""}, "id must be snake_case"),
        ({"id": "Induction-Attention"}, "id must be snake_case"),
        ({"version": "1.0"}, "version must be X.Y.Z"),
        ({"version": "v1.0.0"}, "version must be X.Y.Z"),
        ({"evidence_type": "heuristic_atention"}, "unknown evidence_type"),
        ({"maturity": "candidate"}, "maturity must be one of"),
        ({"method_reference": "Olsson et al."}, "method_reference must be a URL"),
        ({"description": "  "}, "description must not be empty"),
        ({"prompt_suite": ""}, "prompt_suite must not be empty"),
        ({"default_parameters": {"burnin": 2}}, "do not fit build_mask"),
        ({"build_mask": induction_mask_for_seq}, "do not fit build_mask"),
    ],
)
def test_invalid_spec_fails_at_creation(changes, message):
    with pytest.raises(ValueError, match=message):
        make_spec(**changes)


def test_all_problems_are_reported_together():
    with pytest.raises(ValueError, match="version.*evidence_type"):
        make_spec(version="1", evidence_type="nope")


def test_spec_and_its_defaults_are_immutable():
    spec = make_spec()
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.version = "2.0.0"  # type: ignore[misc]
    with pytest.raises(TypeError):
        spec.default_parameters["burn_in"] = 5  # type: ignore[index]


def test_defaults_are_copied_not_shared():
    defaults = {"burn_in": 2}
    spec = make_spec(default_parameters=defaults)
    defaults["burn_in"] = 9
    assert spec.default_parameters["burn_in"] == 2


# --- Parameters ------------------------------------------------------------------------------


def test_resolve_parameters_applies_overrides():
    spec = make_spec()
    assert spec.resolve_parameters() == {"burn_in": 2}
    assert spec.resolve_parameters(burn_in=1) == {"burn_in": 1}


def test_unknown_override_is_rejected():
    with pytest.raises(ValueError, match=r"no parameter\(s\) \['burnin'\]"):
        make_spec().resolve_parameters(burnin=1)


def test_mask_uses_defaults_and_overrides():
    spec = make_spec()
    assert selected(spec.mask(5)) == [(2, 1), (3, 2), (4, 3)]
    assert selected(spec.mask(5, burn_in=1)) == [(1, 0), (2, 1), (3, 2), (4, 3)]


# --- Registry --------------------------------------------------------------------------------


def test_register_and_get():
    registry = DetectorRegistry()
    spec = registry.register(make_spec())
    assert registry.get("toy_detector") is spec
    assert "toy_detector" in registry
    assert len(registry) == 1


def test_duplicate_id_is_rejected_and_original_kept():
    registry = DetectorRegistry()
    original = registry.register(make_spec())
    with pytest.raises(ValueError, match="already registered"):
        registry.register(make_spec(version="2.0.0"))
    assert registry.get("toy_detector") is original


def test_unknown_id_lists_registered_ids():
    registry = DetectorRegistry()
    registry.register(make_spec())
    with pytest.raises(KeyError, match="unknown detector 'induction_attn'.*toy_detector"):
        registry.get("induction_attn")


def test_iterates_in_registration_order():
    registry = DetectorRegistry()
    registry.register(make_spec(id="b_detector"))
    registry.register(make_spec(id="a_detector"))
    assert [s.id for s in registry] == registry.ids() == ["b_detector", "a_detector"]


# --- Default registry ------------------------------------------------------------------------


def test_default_registry_has_the_three_m1_detectors():
    registry = default_registry()
    assert registry.ids() == [
        "previous_token_attention",
        "first_token_attention",
        "induction_attention",
    ]
    for spec in registry:
        assert spec.evidence_type == "heuristic_attention"
        assert spec.maturity == "experimental"  # no real-model controls have run yet
        assert spec.prompt_suite == "repeated_tokens_v1"


def test_default_registry_is_fresh_each_call():
    a, b = default_registry(), default_registry()
    a.register(make_spec())
    assert "toy_detector" not in b


def test_only_burn_in_detectors_are_tunable():
    registry = default_registry()
    assert dict(registry.get("previous_token_attention").default_parameters) == {"burn_in": 2}
    assert dict(registry.get("first_token_attention").default_parameters) == {"burn_in": 2}
    assert dict(registry.get("induction_attention").default_parameters) == {}


@pytest.mark.parametrize("repeat_len", [3, 10])
def test_every_detector_runs_through_one_call_shape(repeat_len):
    # The pipeline only needs spec.mask(seq): no detector-specific code paths.
    seq = 1 + 2 * repeat_len
    heads = {
        "previous_token_attention": previous_token_head(seq),
        "first_token_attention": sink_head(seq),
        "induction_attention": repeated_head(repeat_len, offset=1),
    }
    registry = default_registry()
    for detector_id, ideal_head in heads.items():
        spec = registry.get(detector_id)
        own = masked_head_mean(ideal_head, spec.mask(seq)).per_example.item()
        assert own == 1.0, detector_id
        for other in registry:
            if other.id != detector_id:
                score = masked_head_mean(ideal_head, other.mask(seq)).per_example.item()
                assert score < 1.0, (detector_id, other.id)
