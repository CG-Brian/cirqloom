"""Input-side control: the same heads on repeated vs non-repeated input (M1 Phase 3, §7.4).

The two experiments are paired: same seeds, same length, same BOS and first half, so every
mask selects the same cells. Only whether the content repeats differs. A content-dependent
head (induction) should drop on the control; a positional head (previous-token) should not.

This module only reports numbers. It does not decide that a detector "passed": that judgement,
and any threshold, is left to a person reading the report (AGENTS.md §12, §15).
"""

from dataclasses import dataclass

import numpy as np

from cirqloom_analysis.capture.runner import ExperimentRun


@dataclass(frozen=True)
class HeadComparison:
    label: str
    measured: float
    control: float


@dataclass(frozen=True)
class ControlComparison:
    detector_id: str
    measurement_id: str
    control_id: str
    measured_median: float
    control_median: float
    top_heads: list[HeadComparison]
    """Heads with the highest score on the measurement suite, with their control score."""


def compare(
    measurement: ExperimentRun, control: ExperimentRun, n_heads: int, top_k: int = 5
) -> list[ControlComparison]:
    """One comparison per detector that ran on both experiments."""
    m_prompts = [(p.repeat_len, p.seed) for p in measurement.prompts]
    c_prompts = [(p.repeat_len, p.seed) for p in control.prompts]
    if m_prompts != c_prompts:
        raise ValueError("control must use the same lengths and seeds as the measurement")
    if not np.array_equal(
        measurement.patterns.shape, control.patterns.shape
    ):
        raise ValueError("measurement and control patterns differ in shape")

    comparisons = []
    for detector_id, m_result in measurement.results.items():
        c_result = control.results.get(detector_id)
        if c_result is None:
            continue
        m_mean, c_mean = m_result.summary.mean, c_result.summary.mean
        order = np.argsort(-m_mean, kind="stable")[:top_k]
        comparisons.append(
            ControlComparison(
                detector_id=detector_id,
                measurement_id=measurement.experiment_id,
                control_id=control.experiment_id,
                measured_median=float(np.median(m_mean)),
                control_median=float(np.median(c_mean)),
                top_heads=[
                    HeadComparison(
                        label=f"L{i // n_heads}H{i % n_heads}",
                        measured=float(m_mean[i]),
                        control=float(c_mean[i]),
                    )
                    for i in order
                ],
            )
        )
    return comparisons


def controls_markdown(comparisons: list[ControlComparison]) -> str:
    lines = [
        "| Detector | Measurement → control | Top heads on measurement (measured / control) "
        "| Median of all heads (measured / control) |",
        "| --- | --- | --- | --- |",
    ]
    for c in comparisons:
        heads = ", ".join(f"{h.label} {h.measured:.3f} / {h.control:.3f}" for h in c.top_heads)
        lines.append(
            f"| `{c.detector_id}` | `{c.measurement_id}` → `{c.control_id}` | {heads} "
            f"| {c.measured_median:.3f} / {c.control_median:.3f} |"
        )
    return "\n".join(lines) + "\n"
