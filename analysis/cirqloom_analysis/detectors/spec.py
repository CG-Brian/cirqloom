"""Detector specs: what a score means and how it was made (checkpoint 7, AGENTS.md §0.3, §14).

A spec is the fixed description of one detector, known before any model is run. When results
are exported (M1 Phase 3) it is combined with a HeadSummary and run details into Observations:

    Observation field   comes from
    -----------------   ----------------------------------------------------------
    detector_id         spec.id
    detector_version    spec.version
    evidence_type       spec.evidence_type
    method_reference    spec.method_reference
    score               HeadSummary.mean            (run)
    sample_count        HeadSummary.n               (run)
    status              "unavailable" if score is None, else spec.maturity

Parameters actually used (e.g. burn_in) must be recorded with each result; the spec only holds
the defaults. Bump `version` whenever the same call could give a different number: a changed
mask rule or a changed default. Docstring or error-message edits do not need a bump.

Classification (candidate status, thresholds) is not part of a spec yet: thresholds must be
benchmarked against controls per detector (AGENTS.md §7.4-7.5, §15).
"""

import inspect
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, get_args

import numpy as np

from cirqloom_analysis.schema.bundle import EvidenceType

Maturity = Literal["measured", "experimental"]
"""How far the method is validated. "experimental" until positive and negative controls on
real model data pass; never upgraded by hand-picking results (AGENTS.md §7.7, §12)."""

MaskBuilder = Callable[..., np.ndarray]
"""Called as build_mask(seq, **parameters) and returns a bool [seq, seq] mask."""

_ID = re.compile(r"[a-z][a-z0-9_]*")
_VERSION = re.compile(r"\d+\.\d+\.\d+")


@dataclass(frozen=True)
class DetectorSpec:
    """Fixed description of one attention-pattern detector. Validated when it is created."""

    id: str
    """Stable snake_case id, e.g. "induction_attention". Written to every Observation."""

    version: str
    """X.Y.Z. Results with different versions are different measurements."""

    evidence_type: EvidenceType
    """What kind of evidence the score is. Attention masks are "heuristic_attention"."""

    method_reference: str
    """URL of the paper or definition the method follows."""

    description: str
    """One-line definition of what the score averages."""

    prompt_suite: str
    """The prompt suite whose inputs this mask assumes; other inputs give meaningless scores."""

    build_mask: MaskBuilder
    """Builds the mask for a given sequence length and parameters."""

    maturity: Maturity = "experimental"

    default_parameters: Mapping[str, int] = field(default_factory=dict)
    """Tunable settings and their defaults, e.g. {"burn_in": 2}. Only these may be overridden."""

    def __post_init__(self) -> None:
        # Fail when the spec is defined, not after a model run: a typo here would otherwise
        # surface only at export time, or silently reach the bundle.
        problems: list[str] = []

        # Check 1: identifiers and text fields.
        if not _ID.fullmatch(self.id):
            problems.append(f"id must be snake_case, got {self.id!r}")
        if not _VERSION.fullmatch(self.version):
            problems.append(f"version must be X.Y.Z, got {self.version!r}")
        if not self.method_reference.startswith(("https://", "http://")):
            problems.append(f"method_reference must be a URL, got {self.method_reference!r}")
        if not self.description.strip():
            problems.append("description must not be empty")
        if not self.prompt_suite.strip():
            problems.append("prompt_suite must not be empty")

        # Check 2: enum values. Type hints are not enforced at runtime, so check them here.
        if self.evidence_type not in get_args(EvidenceType):
            problems.append(f"unknown evidence_type {self.evidence_type!r}")
        if self.maturity not in get_args(Maturity):
            problems.append(f"maturity must be one of {get_args(Maturity)}, got {self.maturity!r}")

        # Check 3: the defaults must be accepted by build_mask(seq, **defaults).
        try:
            inspect.signature(self.build_mask).bind(0, **self.default_parameters)
        except TypeError as error:
            problems.append(f"default_parameters do not fit build_mask: {error}")

        if problems:
            raise ValueError(f"invalid detector spec {self.id!r}: " + "; ".join(problems))

        # Freeze the defaults too: frozen=True alone would still allow mutating the dict.
        object.__setattr__(
            self, "default_parameters", MappingProxyType(dict(self.default_parameters))
        )

    def resolve_parameters(self, **overrides: int) -> dict[str, int]:
        """Defaults with overrides applied: the exact settings to record with the result."""
        unknown = sorted(overrides.keys() - self.default_parameters.keys())
        if unknown:
            raise ValueError(
                f"detector {self.id!r} has no parameter(s) {unknown}; "
                f"tunable: {sorted(self.default_parameters)}"
            )
        return {**self.default_parameters, **overrides}

    def mask(self, seq: int, **overrides: int) -> np.ndarray:
        """The mask for a sequence of length seq, with defaults plus any overrides."""
        return self.build_mask(seq, **self.resolve_parameters(**overrides))
