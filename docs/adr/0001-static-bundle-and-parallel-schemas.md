# ADR 0001: Static JSON bundles, parallel schemas, artificial fixture

- **Status:** Accepted
- **Date:** 2026-10-26
- **Milestone:** M0

## Context

V1 is an offline-precomputed explorer (AGENTS.md §1, §4). Analysis runs in Python, while the explorer is a static React app that must not depend on PyTorch or a running server. Both sides need the same data contract, with runtime validation, so a malformed or fabricated value never reaches the screen. M0 has to work without model weights.

## Decisions

### 1. Static bundles split into `manifest.json` + `graph.json`

A bundle is a directory of JSON files served as static assets. The small manifest loads first and carries the fixture flag, display warning, model info, experiments and provenance. The graph holds nodes, observations, circuits, memberships and edges. Large numeric assets (attention matrices) will be separate, lazily loaded files referenced from the manifest (M1).

- Matches the `DataProvider` split (`loadManifest`, `loadGraph`, `loadHeadAttention`).
- No database, API server or GPU at runtime. An HTTP provider can be added later without UI changes.
- Vite serves `data/` directly as `publicDir`, so there is no copy step and no stale duplicate.

### 2. Parallel Pydantic and zod schemas, tied by a contract test

The schema is defined twice: `analysis/cirqloom_analysis/schema/bundle.py` (Pydantic, `extra="forbid"`, `allow_inf_nan=False`) and `frontend/src/data/schema.ts` (zod `strictObject`). `frontend/src/tests/schema.test.ts` parses the fixture that Python generates, so any drift fails a test on one side.

Alternatives considered:

- **Generate TypeScript from JSON Schema:** adds a codegen step and tooling for a schema of about a dozen types. Revisit if the schema grows substantially.
- **Validate only in Python:** the browser would trust any file it is given, including hand-edited bundles.

Semantic rules that span objects (complete head grid, deterministic ids, dangling references, null-iff-unavailable scores, fixture labelling) live in `schema/validation.py`, with the subset the UI relies on duplicated in `crossCheckBundle` and a zod `superRefine`.

### 3. Deterministic output

Writers validate before writing, sort keys, use fixed indentation and `ensure_ascii=False`, and include no timestamps. Rebuilding an unchanged bundle is byte-identical, and CI enforces this with `git diff --exit-code data/`.

### 4. A deliberately artificial fixture

The demo fixture's circuit is the diagonal L*k*H*k* with alternating roles, `origin: "synthetic_fixture"`, `evidence_status: "experimental"`, and no observations or experiments. An obviously artificial pattern cannot be mistaken for a GPT-2 finding (AGENTS.md §15). The validator requires fixture bundles to carry a display warning and forbids fixture circuits in real bundles.

## Consequences

- Every schema change touches two files plus the fixture, and the contract test makes forgetting one fail loudly.
- The fixture exercises every UI path except measured data and attention. M1 must add tests for those using real or explicitly synthetic observations.
- Bundles larger than the demo should not be committed; they belong in release assets (AGENTS.md §13).
