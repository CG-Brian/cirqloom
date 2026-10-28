# ADR 0002: Measured bundles and schema 1.1.0

Status: accepted, 2026-10-27

## Context

M1 exports real GPT-2 Small attention scores. Schema 1.0.0 could not say which detector settings produced a score, had no spread or per-prompt values, described an experiment by a single seed, and had nowhere to put attention matrices.

## Decision

- **Experiment = one prompt suite at one sequence length**, scored as one batch. It lists all prompt `seeds`, its `kind` (`measurement` or `control`) and a tokens file. Prompts are never padded.
- **Observation** adds `parameters` (exact detector settings), `score_std` (null below 2 prompts) and `per_example_scores`. `sample_count` counts prompts.
- **Provenance** adds an `environment` map (library versions, device, dtype, BOS policy, precision).
- **Attention assets** are raw little-endian float16 `[layer, head, q, k]` for one example prompt per measurement experiment, one file per experiment. The browser fetches it once on first selection and decodes one head's slice. Control experiments keep scores only. This keeps the committed bundle near 5 MB with only a handful of files.
- **Status** is `"unavailable"` when there is no score, otherwise the detector spec's `maturity`. All M1 detectors are `experimental`.
- The measured bundle is committed and validated in CI, but not rebuilt there (that needs model weights). The fixture is still rebuilt and diffed.

## Consequences

- 1.0.0 bundles no longer load; the fixture was regenerated. Both validators changed together.
- Rebuilds are byte-identical on one machine; another BLAS or device may change low-order digits, so rounding to 6 decimals is the tolerance.
- Per-head attention files or HTTP range requests can replace the per-experiment file later without changing the inspector.
