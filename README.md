# Cirqloom

An evidence-aware, interactive 3D atlas of attention heads and circuits in pretrained transformers, starting with GPT-2 Small.

> **Status: M0 (foundation).** The app currently runs on a **synthetic demo fixture**. It is not measured model data: the fixture has no experiments, no detector scores and no attention matrices, and its "circuit" is a deliberately artificial diagonal pattern (L*k*H*k*). Real GPT-2 measurements arrive in M1.

## What works today

- A 3D scene of GPT-2 Small's 12 layers × 12 heads, with orbit, pan and zoom, plus Reset view.
- A circuit overlay with subtle role rings. Non-members recede and spanned layers are outlined. No edges are drawn.
- Head selection from the 3D scene, or from a keyboard-navigable head list. Both stay in sync.
- A head inspector that keeps four kinds of evidence separate: circuit membership, published annotations, measured observations and attention. Missing data is stated as missing, never shown as zero.
- A versioned bundle schema, defined in Python (Pydantic) and TypeScript (zod), with a cross-language contract test.

## Quickstart: frontend with the bundled fixture

No Python and no model weights needed.

```bash
cd frontend
npm ci
npm run dev
```

Open the URL Vite prints (default http://localhost:5173).

| Action | Control |
| --- | --- |
| Rotate | Left-drag |
| Pan | Right-drag |
| Zoom | Scroll / pinch |
| Select a head | Click it, or use the head list |
| Move in the head list | Arrow keys (Left/Right = head, Up/Down = layer), Home/End |
| Clear selection | Esc, or click empty space |

## Quickstart: regenerate the fixture (Python)

```bash
python3 -m venv .venv
.venv/bin/pip install -e "analysis[dev]"
.venv/bin/python analysis/scripts/build_fixture_bundle.py
.venv/bin/python analysis/scripts/validate_bundle.py data/fixtures/gpt2-small-demo
```

The build is deterministic: rebuilding without code changes yields byte-identical files, and CI checks this.

## Checks

```bash
# Python (from analysis/)
../.venv/bin/ruff check .
../.venv/bin/pytest

# Frontend (from frontend/)
npm run lint
npm run typecheck
npm test
npm run build
npx playwright install chromium   # first time only
npm run e2e                       # browser smoke tests; starts its own dev server on port 5199
```

## Tested versions

| Tool | Version | Note |
| --- | --- | --- |
| Python | 3.14.7 | `requires-python >=3.12`; only 3.14 is tested |
| Node / npm | 26.8 / 11.19 | |
| TypeScript | ~6.0 | pinned: typescript-eslint 8.x does not support TS 7 |
| React | ~19.3 | pinned: @react-three/fiber 9.8 requires React < 19.4 |

## Repository layout

```text
analysis/   Python: bundle schema, validation, deterministic writer, fixture generator
data/       Generated bundles (only the small demo fixture is committed)
frontend/   React + Vite + TypeScript app (React Three Fiber scene, panels, tests)
docs/       Architecture, scientific methods, UI guidelines, ADRs
```

## Documentation

- [Architecture](docs/architecture.md)
- [Scientific methods](docs/scientific-methods.md)
- [UI guidelines](docs/ui-guidelines.md)
- [ADR 0001: static bundles and parallel schemas](docs/adr/0001-static-bundle-and-parallel-schemas.md)

## Scientific stance

Cirqloom keeps observed correlations, published circuit annotations and causal evidence visibly distinct. High attention weight is not a connection between heads, a detector score is not a verified role, and a fixture is not a finding. See [docs/scientific-methods.md](docs/scientific-methods.md).
