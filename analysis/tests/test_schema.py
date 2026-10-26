import math

import pytest
from pydantic import ValidationError

from cirqloom_analysis.export.fixture import build_fixture
from cirqloom_analysis.schema.bundle import Experiment, Node, Observation
from cirqloom_analysis.schema.validation import BundleValidationError, validate_bundle

VALID_OBS = dict(
    node_id="model:gpt2-small:head:L0H0",
    experiment_id="exp-test",
    detector_id="previous_token",
    detector_version="0.0.0",
    evidence_type="heuristic_attention",
    score=0.5,
    status="measured",
    sample_count=3,
    method_reference="test",
)


@pytest.fixture
def bundle():
    return build_fixture()


def with_observation(bundle, **overrides):
    manifest, graph = bundle
    exp = Experiment(
        id="exp-test", prompt_suite="t", seed=0, sequence_length=8, tokens_uri="t.json"
    )
    obs = Observation(**{**VALID_OBS, **overrides})
    return (
        manifest.model_copy(update={"experiments": [exp]}),
        graph.model_copy(update={"observations": [obs]}),
    )


def test_fixture_is_valid(bundle):
    validate_bundle(*bundle)


def test_valid_observation_passes(bundle):
    validate_bundle(*with_observation(bundle))


def test_rejects_missing_head(bundle):
    manifest, graph = bundle
    graph = graph.model_copy(update={"nodes": graph.nodes[:-1]})
    with pytest.raises(BundleValidationError, match="expected 144 head nodes, found 143"):
        validate_bundle(manifest, graph)


def test_rejects_duplicate_node(bundle):
    manifest, graph = bundle
    nodes = [*graph.nodes[:-1], graph.nodes[0]]
    with pytest.raises(BundleValidationError, match="duplicate node id"):
        validate_bundle(manifest, graph.model_copy(update={"nodes": nodes}))


def test_rejects_fake_zero_for_unavailable(bundle):
    with pytest.raises(BundleValidationError, match="null if and only if"):
        validate_bundle(*with_observation(bundle, score=0.0, status="unavailable"))


def test_rejects_nan_score():
    with pytest.raises(ValidationError):
        Observation(**{**VALID_OBS, "score": math.nan})


def test_rejects_unknown_field():
    with pytest.raises(ValidationError):
        Node(id="x", type="attention_head", layer=0, head=0, colour="red")


def test_rejects_membership_to_missing_node(bundle):
    manifest, graph = bundle
    bad = graph.memberships[0].model_copy(update={"node_id": "model:gpt2-small:head:L99H0"})
    graph = graph.model_copy(update={"memberships": [bad]})
    with pytest.raises(BundleValidationError, match="unknown node"):
        validate_bundle(manifest, graph)


def test_fixture_requires_warning(bundle):
    manifest, graph = bundle
    with pytest.raises(BundleValidationError, match="display_warning"):
        validate_bundle(manifest.model_copy(update={"display_warning": None}), graph)


def test_reports_all_problems_at_once(bundle):
    manifest, graph = bundle
    manifest = manifest.model_copy(update={"display_warning": None, "bundle_id": "other"})
    with pytest.raises(BundleValidationError) as exc_info:
        validate_bundle(manifest, graph)
    assert len(exc_info.value.problems) == 2
