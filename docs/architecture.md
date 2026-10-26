# Architecture

## Layers

```text
analysis/ (Python)                         frontend/ (TypeScript)
──────────────────                         ──────────────────────
schema/bundle.py      Pydantic models      data/schema.ts           zod mirror + types
schema/validation.py  cross-object rules   data/DataProvider.ts     interface + typed errors
export/fixture.py     synthetic fixture    data/StaticBundleProvider.ts  fetch + validate
export/bundle_writer  validate → write     features/experiment/     graphView (indexing), useBundle
scripts/              build / validate     features/selection/      Zustand store (ids + flags)
                                           features/overlays/       pure highlight selectors
            │                              scene/                   R3F canvas, camera, heads
            ▼                              panels/ + app/           sidebar, list, inspector, shell
data/fixtures/<bundle>/manifest.json, graph.json
            │  (served as Vite publicDir)
            └──────────────────────────────▶ StaticBundleProvider
```

**Layering rule:** the frontend consumes model-agnostic, versioned nodes, memberships and observations. It never imports PyTorch or knows tensor shapes or hook names, and detectors never contain rendering logic.

## Data flow in the app

1. `StaticBundleProvider` fetches `manifest.json`, then `graph.json`. Each is parsed with zod, then `crossCheckBundle` checks the two together. Failures throw typed errors (`BundleLoadError`, `BundleSchemaError`) that the app shows verbatim.
2. `buildGraphView` indexes the bundle once: nodes sorted by (layer, head), positions from `scene/layout.ts`, and lookups for nodes, memberships and observations per head.
3. `App` computes the active `CircuitOverlay` once from the store's `activeCircuitId` and passes it to the scene, sidebar and status bar.
4. Components read UI state from `useExplorerStore` with narrow selectors, then call pure selectors (`getNodeDisplayStyle`, `getLayerFrameStyle`, `getHeadEvidence`, …) to decide what to draw.

## State

| State | Lives in | Why |
| --- | --- | --- |
| Bundle data | `useBundle` result → `GraphView` | Immutable once loaded |
| Selection, hover, active circuit, view mode, camera command | Zustand store | Shared by 3D scene, list, inspector; narrow subscriptions keep re-renders small |
| Camera pose | `CameraController` (three.js objects) | Changes every frame; must not re-render React |
| Panel collapse | `App` local state | Layout only |

The camera is driven by events, not state: `resetCamera()` bumps a nonce in the store, and `CameraController` animates toward the canonical pose.

## Bundle format

- `manifest.json`: schema version, bundle id, `synthetic_fixture` flag, display warning, model info, experiments, provenance, and `graph_uri`.
- `graph.json`: nodes, observations, circuits (with role definitions), memberships, and edges.

Schema rules shared by both languages: unknown keys are rejected, NaN and Infinity are rejected, `score` is `null` if and only if `status` is `"unavailable"`, and node ids are `model:{model}:head:L{layer}H{head}` (zero-based). See [ADR 0001](adr/0001-static-bundle-and-parallel-schemas.md).

## Extension seams

- **New data source:** implement `DataProvider`. The UI needs no change.
- **New model:** the layout and scene take layer and head counts from the manifest; nothing assumes 12×12.
- **New detector:** emit `Observation` records. The inspector already renders measured, candidate, experimental and unavailable states.
- **Attention heatmaps (M1/M2):** `DataProvider.loadHeadAttention` exists. The static provider throws `AttentionUnavailableError` until bundles carry attention assets.
