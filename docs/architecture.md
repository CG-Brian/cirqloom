# Architecture

## Layers

```text
analysis/ (Python)                         frontend/ (TypeScript)
──────────────────                         ──────────────────────
schema/bundle.py      Pydantic models      data/schema.ts           zod mirror + types
schema/validation.py  cross-object rules   data/DataProvider.ts     interface + typed errors
models/               adapters (torch)     data/StaticBundleProvider.ts  fetch + validate + attention
prompts/              prompt suites        data/float16.ts          attention asset decoding
detectors/            numpy masks, specs   features/experiment/     graphView, useBundle, active experiment
capture/              run + score, controls features/selection/     Zustand store (ids + flags)
export/               fixture, measured,   features/overlays/       pure highlight selectors
                      validate → write     scene/                   R3F canvas, camera, heads
published/            reviewed circuit
                      annotations → records
scripts/              build / validate     panels/ + app/           sidebar, list, inspector, shell
            │
            ▼
data/fixtures/gpt2-small-demo/        manifest.json, graph.json
data/bundles/gpt2-small/              manifest.json, graph.json, assets/, README.md
            │  (served as Vite publicDir)
            └──────────────────────────────▶ StaticBundleProvider
```

**Layering rule:** the frontend consumes model-agnostic, versioned nodes, memberships and observations. It never imports PyTorch or knows tensor shapes or hook names, and detectors never contain rendering logic. On the Python side only `models/gpt2.py` imports torch / TransformerLens; everything after the adapter is numpy.

## Analysis pipeline

`build_gpt2_bundle.py`: `GPT2Adapter` (pinned revision, checks 12×12 / n_ctx / vocab / BOS after loading) → for each length, `build_suite` for the repeated and control suites → `run_experiment` (one batch per experiment, `run_attention` validates token ids before and attention after the model, every registered detector that accepts the suite scores it) → `compare` (control report) → `import_ioi` (published annotations, no model involved; refuses any model but GPT-2 Small) → `build_measured_bundle` (adds published circuits as given; they never change observations, masks or assets) → `write_bundle` → re-read and `validate_bundle_files`. Detectors see heads on one flat axis (`layer * n_heads + head`); `unflatten_heads` is the only place that undoes it.

## Data flow in the app

1. `StaticBundleProvider` fetches `manifest.json`, then `graph.json`. Each is parsed with zod, then `crossCheckBundle` checks the two together. Failures throw typed errors (`BundleLoadError`, `BundleSchemaError`) that the app shows verbatim.
2. `buildGraphView` indexes the bundle once: nodes sorted by (layer, head), positions from `scene/layout.ts`, and lookups for nodes, memberships and observations per head.
3. `getActiveExperiment` picks the experiment to show (the user's pick, else the first measurement run); the inspector filters observations to it, because scores from different suites are not comparable.
4. `App` computes the active `CircuitOverlay` once from the store's `activeCircuitId` and passes it to the scene, sidebar and status bar.
5. Components read UI state from `useExplorerStore` with narrow selectors, then call pure selectors (`getNodeDisplayStyle`, `getLayerFrameStyle`, `getHeadEvidence`, …) to decide what to draw.

## State

| State | Lives in | Why |
| --- | --- | --- |
| Bundle data | `useBundle` result → `GraphView` | Immutable once loaded |
| Selection, hover, active circuit, experiment, mask detector, view mode, camera command | Zustand store | Shared by 3D scene, list, inspector; narrow subscriptions keep re-renders small |
| Camera pose | `CameraController` (three.js objects) | Changes every frame; must not re-render React |
| Panel collapse | `App` local state | Layout only |

The camera is driven by events, not state: `resetCamera()` bumps a nonce in the store, and `CameraController` animates toward the canonical pose.

## Bundle format

- `manifest.json`: schema version (1.3.0), bundle id, `synthetic_fixture` flag, display warning, model info, experiments (suite, kind, seeds, length, tokens file, optional attention asset), provenance (including an `environment` map), and `graph_uri`.
- `graph.json`: nodes, observations (with `parameters`, `score_std`, `per_example_scores`), masks (the `[q, k]` cells each detector averaged per experiment, see [ADR 0003](adr/0003-detector-masks-schema-1-2.md)), circuits (with role definitions and an origin: `published`, `measured_reconstruction` or `synthetic_fixture`), memberships (with an optional note), and evidence-typed edges: endpoints are a head, a role group or `logits`/`embed`, with channel, token positions, evidence kind and provenance, stored at the granularity the evidence was measured at (see [ADR 0004](adr/0004-circuit-edges-schema-1-3.md)).
- `assets/*.tokens.json`: every prompt of one experiment, ids and display strings (leading spaces kept).
- `assets/*.attention.f16`: one example prompt's attention for all heads, little-endian float16 `[layer, head, q, k]`. Fetched once per experiment on first head selection; the provider decodes only the selected head's slice.

Schema rules shared by both languages: unknown keys are rejected, NaN and Infinity are rejected, `score` is `null` if and only if `status` is `"unavailable"`, and node ids are `model:{model}:head:L{layer}H{head}` (zero-based). See [ADR 0001](adr/0001-static-bundle-and-parallel-schemas.md).

## Extension seams

- **New data source:** implement `DataProvider`. The UI needs no change.
- **New model:** the layout and scene take layer and head counts from the manifest; nothing assumes 12×12.
- **New detector:** emit `Observation` records. The inspector already renders measured, candidate, experimental and unavailable states.
- **Circuit graph (M3 step 3):** `features/circuits/edgeGroups.ts` bundles `Graph.edges` into `VisualEdgeGroup`s without modifying a record and provides labels, the heads a relation is about, and line styles; `features/circuits/circuitLayout.ts` places role-group hubs, subset anchors, `logits`, and curve points in the band above the layers (`circuitBandY`; `sceneBounds` frames it). `scene/CircuitEdgeLayer.tsx` draws them; `panels/EdgeInspector.tsx` shows one relation and its records. The store adds `hoveredEdgeGroupId` / `selectedEdgeGroupId`; selecting a head or a relation clears the other. Nothing is IOI-specific: any circuit with role groups and edges uses the same path.
- **Attention heatmap (M2 P1):** `useHeadAttention` loads one head through `DataProvider.loadHeadAttention` and drops answers for a head or experiment that is no longer selected. `AttentionPanel` picks the honest state; `AttentionHeatmap` only draws (canvas, one pixel per cell scaled up). Pure helpers (`features/attention/heatmapModel.ts`) do mask lookup, colours, geometry and cell readouts. Masks come from `graph.masks` (ADR 0003), never recomputed in TypeScript.
- **New prompt suite or detector:** add a suite builder and a `DetectorSpec` (with `prompt_suite` / `control_suites`); the runner only pairs detectors with suites they accept.
