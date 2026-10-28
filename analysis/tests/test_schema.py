import math

import pytest
from pydantic import ValidationError

from cirqloom_analysis.export.fixture import build_fixture
from cirqloom_analysis.schema.bundle import AttentionAsset, Experiment, Node, Observation
from cirqloom_analysis.schema.validation import BundleValidationError, validate_bundle

VALID_OBS = dict(
    node_id="model:gpt2-small:head:L0H0",
    experiment_id="exp-test",
    detector_id="previous_token",
    detector_version="0.0.0",
    evidence_type="heuristic_attention",
    parameters={"burn_in": 2},
    score=0.5,
    score_std=0.1,
    per_example_scores=[0.4, 0.5, 0.6],
    status="measured",
    sample_count=3,
    method_reference="test",
)


@pytest.fixture
def bundle():
    return build_fixture()


def make_experiment(**overrides) -> Experiment:
    fields = dict(
        id="exp-test",
        prompt_suite="t",
        kind="measurement",
        description="test",
        seeds=[0, 1, 2],
        sequence_length=8,
        tokens_uri="t.json",
        attention=None,
    )
    return Experiment(**{**fields, **overrides})


def with_observation(bundle, experiment=None, **overrides):
    manifest, graph = bundle
    obs = Observation(**{**VALID_OBS, **overrides})
    return (
        manifest.model_copy(update={"experiments": [experiment or make_experiment()]}),
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


def test_unavailable_observation_has_no_examples(bundle):
    validate_bundle(
        *with_observation(
            bundle,
            score=None,
            score_std=None,
            per_example_scores=[],
            sample_count=0,
            status="unavailable",
        )
    )


def test_rejects_per_example_count_mismatch(bundle):
    with pytest.raises(BundleValidationError, match="one score per example"):
        validate_bundle(*with_observation(bundle, per_example_scores=[0.5]))


def test_std_is_null_exactly_below_two_examples(bundle):
    with pytest.raises(BundleValidationError, match="score_std must be null"):
        validate_bundle(
            *with_observation(bundle, sample_count=1, per_example_scores=[0.5], score_std=0.0)
        )
    with pytest.raises(BundleValidationError, match="score_std must be null"):
        validate_bundle(*with_observation(bundle, score_std=None))
    validate_bundle(
        *with_observation(bundle, sample_count=1, per_example_scores=[0.5], score_std=None)
    )


def test_rejects_duplicate_observation(bundle):
    manifest, graph = with_observation(bundle)
    graph = graph.model_copy(update={"observations": graph.observations * 2})
    with pytest.raises(BundleValidationError, match="duplicate observation"):
        validate_bundle(manifest, graph)


def test_attention_asset_must_match_model_and_length(bundle):
    asset = AttentionAsset(uri="a.bin", dtype="float16", shape=[12, 12, 8, 8], prompt_index=0)
    validate_bundle(*with_observation(bundle, make_experiment(attention=asset)))
    bad = asset.model_copy(update={"shape": [12, 12, 9, 9], "prompt_index": 3})
    with pytest.raises(BundleValidationError) as exc_info:
        validate_bundle(*with_observation(bundle, make_experiment(attention=bad)))
    assert len(exc_info.value.problems) == 2


def test_rejects_duplicate_experiment_and_seeds(bundle):
    manifest, graph = bundle
    exp = make_experiment(seeds=[1, 1])
    manifest = manifest.model_copy(update={"experiments": [exp, exp]})
    with pytest.raises(BundleValidationError) as exc_info:
        validate_bundle(manifest, graph)
    assert len(exc_info.value.problems) == 3  # duplicate id + seeds twice


def test_experiment_needs_a_seed():
    with pytest.raises(ValidationError):
        make_experiment(seeds=[])


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
