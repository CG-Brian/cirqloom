"""Capture attention for one experiment and score it with every applicable detector (M1 Phase 3).

One experiment = one prompt suite at one length = one batch. Prompts of different lengths are
never mixed (no padding, see models/base.py).

Head bookkeeping is the riskiest step here: detectors see heads on one flat axis of
n_layers * n_heads, and a score attached to the wrong head is the worst possible bug. The flat
index is always layer * n_heads + head (numpy C order of [layer, head]); `unflatten_heads` is
the only place that undoes it.
"""

from dataclasses import dataclass

import numpy as np

from cirqloom_analysis.detectors.aggregate import HeadSummary, summarize
from cirqloom_analysis.detectors.registry import DetectorRegistry
from cirqloom_analysis.detectors.scoring import masked_head_mean
from cirqloom_analysis.detectors.spec import DetectorSpec
from cirqloom_analysis.models.base import BaseModelAdapter
from cirqloom_analysis.prompts.repeated import Prompt


@dataclass(frozen=True)
class DetectorResult:
    spec: DetectorSpec
    parameters: dict[str, int]
    """The exact settings used (defaults + overrides): recorded with every observation."""
    summary: HeadSummary
    """Per-head summary on the flat [n_layers * n_heads] axis."""
    mask: np.ndarray
    """The exact bool [seq, seq] mask the scores were averaged over (exported with them)."""


@dataclass(frozen=True)
class ExperimentRun:
    experiment_id: str
    suite: str
    prompts: list[Prompt]
    patterns: np.ndarray
    """[n_prompts, layer, head, q, k], float32, validated by the adapter."""
    results: dict[str, DetectorResult]


def experiment_id(suite: str, repeat_len: int) -> str:
    return f"{suite}:L{repeat_len}"


def detectors_for(registry: DetectorRegistry, suite: str) -> list[DetectorSpec]:
    """Registered detectors that may run on `suite` (measurement or declared control)."""
    return [spec for spec in registry if spec.accepts_suite(suite)]


def capture(adapter: BaseModelAdapter, prompts: list[Prompt]) -> np.ndarray:
    """Attention [n, layer, head, q, k] for prompts of one suite and one length."""
    if not prompts:
        raise ValueError("no prompts to capture")
    suites = {p.suite for p in prompts}
    lengths = {p.token_ids.shape[0] for p in prompts}
    if len(suites) != 1 or len(lengths) != 1:
        raise ValueError(
            f"one experiment needs one suite and one length, got suites {sorted(suites)} "
            f"and lengths {sorted(lengths)}"
        )
    return adapter.run_attention(np.stack([p.token_ids for p in prompts]))


def score(
    patterns: np.ndarray, suite: str, specs: list[DetectorSpec]
) -> dict[str, DetectorResult]:
    """Score [n, layer, head, q, k] patterns with each spec, on a flat head axis."""
    n, n_layers, n_heads, seq, _ = patterns.shape
    flat = patterns.reshape(n, n_layers * n_heads, seq, seq)
    results: dict[str, DetectorResult] = {}
    for spec in specs:
        if not spec.accepts_suite(suite):
            raise ValueError(
                f"detector {spec.id!r} is not valid on suite {suite!r} "
                f"(measures {spec.prompt_suite!r}, controls {list(spec.control_suites)})"
            )
        parameters = spec.resolve_parameters()
        mask = spec.build_mask(seq, **parameters)
        summary = summarize([masked_head_mean(flat, mask)])
        results[spec.id] = DetectorResult(
            spec=spec, parameters=parameters, summary=summary, mask=mask
        )
    return results


def run_experiment(
    adapter: BaseModelAdapter, prompts: list[Prompt], registry: DetectorRegistry
) -> ExperimentRun:
    suite = prompts[0].suite if prompts else ""
    patterns = capture(adapter, prompts)
    specs = detectors_for(registry, suite)
    if not specs:
        raise ValueError(f"no registered detector accepts suite {suite!r}")
    return ExperimentRun(
        experiment_id=experiment_id(suite, prompts[0].repeat_len),
        suite=suite,
        prompts=prompts,
        patterns=patterns,
        results=score(patterns, suite, specs),
    )


def unflatten_heads(values: np.ndarray, n_layers: int, n_heads: int) -> np.ndarray:
    """[n_layers * n_heads, ...] → [n_layers, n_heads, ...]; inverse of the reshape in score."""
    if values.shape[0] != n_layers * n_heads:
        raise ValueError(f"expected {n_layers * n_heads} heads, got {values.shape[0]}")
    return values.reshape(n_layers, n_heads, *values.shape[1:])
