"""M1 Phase 3: capture runner, input-side controls and measured bundle export (no weights)."""

import numpy as np
import pytest
from fake_adapter import INDUCTION_HEAD, INFO, PREVIOUS_TOKEN_HEAD, ContentAwareAdapter

from cirqloom_analysis.capture.controls import compare, controls_markdown
from cirqloom_analysis.capture.runner import (
    capture,
    detectors_for,
    run_experiment,
    score,
    unflatten_heads,
)
from cirqloom_analysis.detectors.registry import default_registry
from cirqloom_analysis.export.bundle_writer import (
    read_bundle,
    validate_bundle_files,
    write_bundle,
)
from cirqloom_analysis.export.measured import build_measured_bundle, observation_status
from cirqloom_analysis.prompts.repeated import CONTROL_SUITE, REPEATED_SUITE, build_suite
from cirqloom_analysis.schema.validation import BundleValidationError, head_node_id

REGISTRY = default_registry()
SEEDS = [0, 1, 2, 3]


@pytest.fixture(scope="module")
def runs():
    adapter = ContentAwareAdapter()
    measured = run_experiment(adapter, build_suite(REPEATED_SUITE, 8, SEEDS, INFO), REGISTRY)
    control = run_experiment(adapter, build_suite(CONTROL_SUITE, 8, SEEDS, INFO), REGISTRY)
    return measured, control


def head_mean(run, detector_id, layer, head):
    mean = unflatten_heads(run.results[detector_id].summary.mean, INFO.n_layers, INFO.n_heads)
    return float(mean[layer, head])


# --- Runner ----------------------------------------------------------------------------------


def test_all_three_detectors_run_on_both_suites(runs):
    measured, control = runs
    ids = {"previous_token_attention", "first_token_attention", "induction_attention"}
    assert set(measured.results) == set(control.results) == ids
    assert measured.experiment_id == "repeated_tokens_v1:L8"
    assert measured.patterns.shape == (4, 2, 3, 17, 17)


def test_scores_land_on_the_right_heads(runs):
    # The off-origin heads catch a layer/head swap in flattening.
    measured, _ = runs
    assert head_mean(measured, "induction_attention", *INDUCTION_HEAD) == 1.0
    assert head_mean(measured, "previous_token_attention", *PREVIOUS_TOKEN_HEAD) == 1.0
    induction = unflatten_heads(
        measured.results["induction_attention"].summary.mean, INFO.n_layers, INFO.n_heads
    )
    assert np.unravel_index(np.argmax(induction), induction.shape) == INDUCTION_HEAD


def test_sample_count_is_prompts_and_parameters_are_recorded(runs):
    measured, _ = runs
    result = measured.results["previous_token_attention"]
    assert result.summary.n == len(SEEDS)
    assert result.parameters == {"burn_in": 2}
    assert measured.results["induction_attention"].parameters == {}


def test_unknown_suite_has_no_detectors_and_is_rejected():
    assert detectors_for(REGISTRY, "natural_text_v1") == []
    patterns = np.zeros((1, 2, 3, 7, 7))
    with pytest.raises(ValueError, match="not valid on suite"):
        score(patterns, "natural_text_v1", list(REGISTRY))


def test_capture_refuses_mixed_lengths_or_suites():
    adapter = ContentAwareAdapter()
    mixed = build_suite(REPEATED_SUITE, 5, [0], INFO) + build_suite(REPEATED_SUITE, 6, [0], INFO)
    with pytest.raises(ValueError, match="one suite and one length"):
        capture(adapter, mixed)
    both = build_suite(REPEATED_SUITE, 5, [0], INFO) + build_suite(CONTROL_SUITE, 5, [1], INFO)
    with pytest.raises(ValueError, match="one suite and one length"):
        capture(adapter, both)


def test_unflatten_heads_inverts_c_order():
    flat = np.arange(6)
    np.testing.assert_array_equal(unflatten_heads(flat, 2, 3), [[0, 1, 2], [3, 4, 5]])
    with pytest.raises(ValueError, match="expected 6 heads"):
        unflatten_heads(np.arange(5), 2, 3)


# --- Input-side control ----------------------------------------------------------------------


def test_induction_drops_on_control_previous_token_does_not(runs):
    measured, control = runs
    assert head_mean(control, "induction_attention", *INDUCTION_HEAD) < 0.2
    assert head_mean(control, "previous_token_attention", *PREVIOUS_TOKEN_HEAD) == 1.0


def test_comparison_reports_top_heads_with_their_control_scores(runs):
    comparisons = {c.detector_id: c for c in compare(*runs, n_heads=INFO.n_heads, top_k=2)}
    top = comparisons["induction_attention"].top_heads[0]
    assert top.label == "L1H2" and top.measured == 1.0 and top.control < 0.2
    assert "`induction_attention`" in controls_markdown(list(comparisons.values()))


def test_comparison_requires_paired_runs(runs):
    measured, _ = runs
    other = run_experiment(
        ContentAwareAdapter(), build_suite(CONTROL_SUITE, 8, [5, 6, 7, 8], INFO), REGISTRY
    )
    with pytest.raises(ValueError, match="same lengths and seeds"):
        compare(measured, other, INFO.n_heads)


# --- Export ----------------------------------------------------------------------------------


def build(runs):
    return build_measured_bundle(
        list(runs), INFO, ContentAwareAdapter().token_strings, bundle_id="toy:test:v1",
        generator="test", environment={"device": "cpu"}, notes="test bundle",
    )


def test_status_rule():
    assert observation_status(None, "experimental") == "unavailable"
    assert observation_status(0.3, "experimental") == "experimental"


def test_export_has_one_observation_per_experiment_detector_head(runs):
    manifest, graph, _ = build(runs)
    assert len(graph.observations) == 2 * 3 * INFO.n_layers * INFO.n_heads
    assert {e.kind for e in manifest.experiments} == {"measurement", "control"}
    assert not manifest.synthetic_fixture and graph.circuits == []
    assert {o.status for o in graph.observations} == {"experimental"}


def test_exported_observation_matches_the_summary(runs):
    measured, _ = runs
    _, graph, _ = build(runs)
    node = head_node_id("toy", *INDUCTION_HEAD)
    obs = next(
        o for o in graph.observations
        if o.node_id == node and o.detector_id == "induction_attention"
        and o.experiment_id == measured.experiment_id
    )
    assert obs.score == 1.0 and obs.score_std == 0.0
    assert obs.per_example_scores == [1.0] * len(SEEDS)
    assert obs.sample_count == len(SEEDS) and obs.parameters == {}


def test_every_experiment_detector_pair_exports_the_mask_it_scored_with(runs):
    manifest, graph, _ = build(runs)
    registry = default_registry()
    seq = {e.id: e.sequence_length for e in manifest.experiments}
    masks = {(m.experiment_id, m.detector_id): m for m in graph.masks}
    pairs = {(o.experiment_id, o.detector_id) for o in graph.observations}
    assert set(masks) == pairs
    for (exp_id, detector_id), mask in masks.items():
        spec = registry.get(detector_id)
        expected = spec.mask(seq[exp_id], **mask.parameters)
        assert [list(c) for c in mask.cells_qk] == np.argwhere(expected).tolist()
        assert mask.detector_version == spec.version


def test_induction_mask_is_the_second_copy_stripe(runs):
    measured, _ = runs
    _, graph, _ = build(runs)
    mask = next(
        m for m in graph.masks
        if m.detector_id == "induction_attention" and m.experiment_id == measured.experiment_id
    )
    repeat_len = measured.prompts[0].repeat_len
    second_copy = range(repeat_len + 1, 2 * repeat_len + 1)
    assert mask.cells_qk == [(q, q - repeat_len + 1) for q in second_copy]


def test_attention_asset_only_for_measurement_and_decodes_back(runs):
    measured, _ = runs
    manifest, _, assets = build(runs)
    exp = {e.kind: e for e in manifest.experiments}
    assert exp["control"].attention is None
    asset = exp["measurement"].attention
    raw = np.frombuffer(assets[asset.uri], dtype="<f2").reshape(asset.shape)
    np.testing.assert_allclose(raw, measured.patterns[0], atol=1e-3)


def test_written_bundle_reads_back_and_files_validate(runs, tmp_path):
    manifest, graph, assets = build(runs)
    write_bundle(manifest, graph, tmp_path, assets, readme="# toy\n")
    read_manifest, read_graph = read_bundle(tmp_path)
    assert read_graph == graph
    validate_bundle_files(tmp_path, read_manifest)
    assert (tmp_path / "README.md").read_text() == "# toy\n"


def test_rewrite_removes_stale_assets(runs, tmp_path):
    manifest, graph, assets = build(runs)
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "old.bin").write_bytes(b"x")
    write_bundle(manifest, graph, tmp_path, assets)
    assert not (tmp_path / "assets" / "old.bin").exists()


def test_writer_refuses_missing_or_escaping_assets(runs, tmp_path):
    manifest, graph, assets = build(runs)
    with pytest.raises(BundleValidationError, match="not written"):
        write_bundle(manifest, graph, tmp_path, {})
    with pytest.raises(ValueError, match="inside assets/"):
        write_bundle(manifest, graph, tmp_path, {**assets, "../evil.txt": b""})


def test_file_validation_catches_truncated_attention(runs, tmp_path):
    manifest, graph, assets = build(runs)
    write_bundle(manifest, graph, tmp_path, assets)
    uri = next(e.attention.uri for e in manifest.experiments if e.attention)
    (tmp_path / uri).write_bytes(b"\0\0")
    with pytest.raises(BundleValidationError, match="expected"):
        validate_bundle_files(tmp_path, manifest)
