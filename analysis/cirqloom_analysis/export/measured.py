"""Build a measured bundle from captured experiment runs (M1 Phase 3, AGENTS.md §8).

Pure: takes runs and model info, returns (manifest, graph, assets). Nothing here touches torch,
so it is tested with a fake adapter.

Observation status: "unavailable" when there is no score, otherwise the detector's maturity
("experimental" until real-model controls validate the method). Never "measured" by default.

Numbers: scores and stds are rounded to 6 decimals (well above float32 noise, keeps rebuilds
stable). Attention assets are float16, one example prompt per experiment.
"""

import json
from collections.abc import Callable

import numpy as np

from cirqloom_analysis.capture.runner import ExperimentRun, unflatten_heads
from cirqloom_analysis.models.base import AdapterInfo
from cirqloom_analysis.prompts.repeated import CONTROL_SUITE, REPEATED_SUITE
from cirqloom_analysis.schema.bundle import (
    SCHEMA_VERSION,
    AttentionAsset,
    DetectorMask,
    EvidenceStatus,
    Experiment,
    Graph,
    Manifest,
    Node,
    Observation,
    Provenance,
)
from cirqloom_analysis.schema.validation import head_node_id

SCORE_DECIMALS = 6
ATTENTION_PROMPT_INDEX = 0  # which prompt of each experiment gets a stored attention matrix

SUITE_KIND = {REPEATED_SUITE: "measurement", CONTROL_SUITE: "control"}
SUITE_DESCRIPTION = {
    REPEATED_SUITE: "(BOS, X, X): X is {L} random tokens repeated once.",
    CONTROL_SUITE: "(BOS, X, Y): same BOS and X as the repeated suite, then {L} fresh random "
    "tokens. Control: no repetition.",
}


def _round(value: float) -> float:
    return round(float(value), SCORE_DECIMALS)


def observation_status(score: float | None, maturity: str) -> EvidenceStatus:
    return "unavailable" if score is None else maturity  # type: ignore[return-value]


def asset_slug(experiment_id: str) -> str:
    """File-name-safe form of an experiment id: "repeated_tokens_v1:L50" → "...-L50"."""
    return experiment_id.replace(":", "-")


def build_observations(run: ExperimentRun, info: AdapterInfo) -> list[Observation]:
    observations: list[Observation] = []
    for detector_id, result in run.results.items():
        summary = result.summary
        mean = unflatten_heads(summary.mean, info.n_layers, info.n_heads)
        std = None if summary.std is None else unflatten_heads(
            summary.std, info.n_layers, info.n_heads
        )
        # per_example is [n, flat heads]; move heads first to unflatten.
        per_example = unflatten_heads(summary.per_example.T, info.n_layers, info.n_heads)
        for layer in range(info.n_layers):
            for head in range(info.n_heads):
                score = _round(mean[layer, head])
                observations.append(
                    Observation(
                        node_id=head_node_id(info.model_id, layer, head),
                        experiment_id=run.experiment_id,
                        detector_id=detector_id,
                        detector_version=result.spec.version,
                        evidence_type=result.spec.evidence_type,
                        parameters=dict(result.parameters),
                        score=score,
                        score_std=None if std is None else _round(std[layer, head]),
                        per_example_scores=[_round(v) for v in per_example[layer, head]],
                        status=observation_status(score, result.spec.maturity),
                        sample_count=summary.n,
                        method_reference=result.spec.method_reference,
                    )
                )
    return observations


def build_masks(run: ExperimentRun) -> list[DetectorMask]:
    """One mask per detector: the cells its scores averaged, as sorted [q, k] pairs."""
    return [
        DetectorMask(
            experiment_id=run.experiment_id,
            detector_id=detector_id,
            detector_version=result.spec.version,
            parameters=dict(result.parameters),
            # argwhere walks the mask in C order, so cells come out sorted by (q, k).
            cells_qk=[(int(q), int(k)) for q, k in np.argwhere(result.mask)],
        )
        for detector_id, result in run.results.items()
    ]


def _dump_json(payload: object) -> bytes:
    text = json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return (text + "\n").encode("utf-8")


def build_measured_bundle(
    runs: list[ExperimentRun],
    info: AdapterInfo,
    token_strings: Callable[[np.ndarray], list[str]],
    *,
    bundle_id: str,
    generator: str,
    environment: dict[str, str],
    notes: str,
) -> tuple[Manifest, Graph, dict[str, bytes]]:
    """Manifest, graph and asset files (relative path → bytes) for a measured bundle."""
    if not runs:
        raise ValueError("no experiment runs to export")
    assets: dict[str, bytes] = {}
    experiments: list[Experiment] = []
    observations: list[Observation] = []
    masks: list[DetectorMask] = []

    for run in runs:
        if run.suite not in SUITE_KIND:
            raise ValueError(f"no export description for suite {run.suite!r}")
        repeat_len = run.prompts[0].repeat_len
        seq = run.patterns.shape[-1]
        slug = asset_slug(run.experiment_id)

        tokens_uri = f"assets/{slug}.tokens.json"
        assets[tokens_uri] = _dump_json(
            {
                "experiment_id": run.experiment_id,
                "prompts": [
                    {
                        "id": p.id,
                        "seed": p.seed,
                        "token_ids": [int(i) for i in p.token_ids],
                        "tokens": token_strings(p.token_ids),
                    }
                    for p in run.prompts
                ],
            }
        )

        attention = None
        if SUITE_KIND[run.suite] == "measurement":
            uri = f"assets/{slug}.attention.f16"
            example = run.patterns[ATTENTION_PROMPT_INDEX]  # [layer, head, q, k]
            assets[uri] = np.ascontiguousarray(example, dtype="<f2").tobytes()
            attention = AttentionAsset(
                uri=uri, dtype="float16", shape=list(example.shape), prompt_index=0
            )

        experiments.append(
            Experiment(
                id=run.experiment_id,
                prompt_suite=run.suite,
                kind=SUITE_KIND[run.suite],
                description=SUITE_DESCRIPTION[run.suite].format(L=repeat_len),
                seeds=[p.seed for p in run.prompts],
                sequence_length=seq,
                tokens_uri=tokens_uri,
                attention=attention,
            )
        )
        observations.extend(build_observations(run, info))
        masks.extend(build_masks(run))

    nodes = [
        Node(id=head_node_id(info.model_id, layer, head), type="attention_head", layer=layer,
             head=head)
        for layer in range(info.n_layers)
        for head in range(info.n_heads)
    ]
    manifest = Manifest(
        schema_version=SCHEMA_VERSION,
        bundle_id=bundle_id,
        synthetic_fixture=False,
        display_warning=None,
        model=info.to_bundle_model(),
        experiments=experiments,
        provenance=Provenance(
            generator=generator, generator_version="1", notes=notes, environment=environment
        ),
        graph_uri="graph.json",
    )
    graph = Graph(
        schema_version=SCHEMA_VERSION,
        bundle_id=bundle_id,
        nodes=nodes,
        observations=observations,
        masks=masks,
        circuits=[],
        memberships=[],
        edges=[],
    )
    return manifest, graph, assets
