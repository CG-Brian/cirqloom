"""Detector registry: one place that lists every detector by id (checkpoint 7, AGENTS.md §14).

The pipeline runs "every registered detector" instead of naming detectors throughout the
code, and an id typo fails at lookup instead of producing an Observation the frontend cannot
match. Duplicate ids are rejected: silently replacing a spec would make results depend on
import order.
"""

from collections.abc import Iterator

from cirqloom_analysis.detectors.first_token import first_token_mask
from cirqloom_analysis.detectors.induction import induction_mask_for_seq
from cirqloom_analysis.detectors.masks import DEFAULT_BURN_IN
from cirqloom_analysis.detectors.previous_token import previous_token_mask
from cirqloom_analysis.detectors.spec import DetectorSpec

INDUCTION_HEADS_PAPER = (
    "https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html"
)
ATTENTION_SINKS_PAPER = "https://arxiv.org/abs/2309.17453"
REPEATED_TOKENS_SUITE = "repeated_tokens_v1"


class DetectorRegistry:
    """id → DetectorSpec. Iterates in registration order."""

    def __init__(self) -> None:
        self._specs: dict[str, DetectorSpec] = {}

    def register(self, spec: DetectorSpec) -> DetectorSpec:
        if spec.id in self._specs:
            raise ValueError(f"detector {spec.id!r} is already registered")
        self._specs[spec.id] = spec
        return spec

    def get(self, detector_id: str) -> DetectorSpec:
        try:
            return self._specs[detector_id]
        except KeyError:
            raise KeyError(
                f"unknown detector {detector_id!r}; registered: {self.ids()}"
            ) from None

    def ids(self) -> list[str]:
        return list(self._specs)

    def __contains__(self, detector_id: object) -> bool:
        return detector_id in self._specs

    def __iter__(self) -> Iterator[DetectorSpec]:
        return iter(self._specs.values())

    def __len__(self) -> int:
        return len(self._specs)


def default_registry() -> DetectorRegistry:
    """A fresh registry with the M1 attention-pattern detectors (AGENTS.md §7.2-7.4)."""
    registry = DetectorRegistry()
    registry.register(
        DetectorSpec(
            id="previous_token_attention",
            version="1.0.0",
            evidence_type="heuristic_attention",
            method_reference=INDUCTION_HEADS_PAPER,
            description="Mean attention on k = q - 1 over rows q >= burn_in.",
            prompt_suite=REPEATED_TOKENS_SUITE,
            build_mask=previous_token_mask,
            default_parameters={"burn_in": DEFAULT_BURN_IN},
        )
    )
    registry.register(
        DetectorSpec(
            id="first_token_attention",
            version="1.0.0",
            evidence_type="heuristic_attention",
            method_reference=ATTENTION_SINKS_PAPER,
            description="Mean attention on k = 0 over rows q >= burn_in; may be a sink.",
            prompt_suite=REPEATED_TOKENS_SUITE,
            build_mask=first_token_mask,
            default_parameters={"burn_in": DEFAULT_BURN_IN},
        )
    )
    registry.register(
        DetectorSpec(
            id="induction_attention",
            version="1.0.0",
            evidence_type="heuristic_attention",
            method_reference=INDUCTION_HEADS_PAPER,
            description="Mean attention on k = q - L + 1 over second-copy rows of (BOS, X, X).",
            prompt_suite=REPEATED_TOKENS_SUITE,
            build_mask=induction_mask_for_seq,
        )
    )
    return registry
