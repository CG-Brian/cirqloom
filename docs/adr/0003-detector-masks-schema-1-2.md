# ADR 0003: Detector masks in the bundle (schema 1.2.0)

Status: accepted, 2026-10-28

## Context

M2 draws an attention heatmap and needs to show which cells a detector score averaged. The mask rule lives in Python (`DetectorSpec.build_mask`). Re-implementing it in TypeScript would let the drawn mask drift from the scored one (for example after a version or default change) with nothing on screen to show it.

## Decision

- **Python exports the exact mask it scored with**, and the frontend only draws it. The runner keeps the bool mask in `DetectorResult`; the exporter turns it into a record.
- **One mask per (experiment, detector)**, in a top-level `graph.masks` list next to `observations`. A mask depends only on detector id, version, parameters and `seq`, never on the head or the prompt content. One experiment has one suite and one length (enforced by `capture`), so every prompt in it shares the mask.
- **Record:** `experiment_id`, `detector_id`, `detector_version`, `parameters`, `cells_qk`. `cells_qk` is a sparse, sorted, unique list of `[q, k]` = `[destination, source]` pairs. The current masks have at most `seq - 2` cells, so sparse pairs are far smaller than a dense `seq × seq` matrix.
- **Validator rules:** the experiment exists; one mask per pair; cells are non-empty, sorted, unique and satisfy `0 <= k <= q < seq`; every observation of the pair has the mask's version and parameters; a mask with no observation is rejected. An observation without a mask is allowed, because future detectors (for example logit attribution) may not be masks. The export test requires a mask for every M1 detector.

## Consequences

- 1.1.0 bundles no longer load. The fixture and the measured bundle were regenerated. Scores and assets are byte-identical; only `schema_version` and `masks` changed.
- The frontend can look a mask up by (experiment, detector) without checking settings: the Python validator already guarantees they match the observations.
- Experiments that mix prompt lengths would need per-prompt masks and a schema change.
