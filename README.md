# Cirqloom

An evidence-aware, interactive 3D atlas of attention heads and circuits in pretrained transformers, starting with GPT-2 Small.

> **Status: M1 (measured GPT-2 bundle).** The app opens a **measured GPT-2 Small bundle** by default: previous-token, first-token (BOS) and induction attention scores on repeated random tokens, with a paired non-repeated control. Every score is labelled `experimental`: it is a heuristic attention measurement, not causal evidence. No published circuits (IOI) are included yet. The synthetic demo fixture is still available at `?bundle=fixture`.

## What works today

- A 3D scene of GPT-2 Small's 12 layers × 12 heads, with orbit, pan and zoom, plus Reset view.
- A circuit overlay with subtle role rings. Non-members recede and spanned layers are outlined. No edges are drawn.
- Head selection from the 3D scene, or from a keyboard-navigable head list. Both stay in sync.
- A head inspector that keeps four kinds of evidence separate: circuit membership, published annotations, measured observations and attention. Missing data is stated as missing, never shown as zero.
- An experiment picker. The inspector shows one experiment's scores at a time (mean ± sd over prompts, n, detector settings), and labels control runs.
- A Python pipeline: GPT-2 adapter on TransformerLens `TransformerBridge` (pinned revision), deterministic prompt suites, numpy detectors, and a bundle exporter with lazily loaded float16 attention assets.
- A versioned bundle schema, defined in Python (Pydantic) and TypeScript (zod), with a cross-language contract test.

## Quickstart: frontend with the committed bundles

No Python and no model weights needed: both bundles are committed under `data/`.

```bash
cd frontend
npm ci
npm run dev
```

Open the URL Vite prints (default http://localhost:5173). Add `?bundle=fixture` for the synthetic demo fixture.

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

## Regenerate the measured GPT-2 bundle (Python, downloads weights)

```bash
.venv/bin/pip install -e "analysis[dev,model]"
.venv/bin/python analysis/scripts/build_gpt2_bundle.py
.venv/bin/python analysis/scripts/validate_bundle.py data/bundles/gpt2-small-repeated-v1
```

The first run downloads the pinned GPT-2 weights (~500 MB) into the Hugging Face cache; a CPU run takes about 10 seconds after that. The output is byte-identical across rebuilds on the same machine. The bundle's `README.md` records the environment and the control comparison. CI validates the committed bundle but does not rebuild it.

Real-model tests are opt-in:

```bash
cd analysis && CIRQLOOM_REAL_MODEL=1 ../.venv/bin/pytest tests/test_gpt2_adapter.py
```

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
| torch / transformers / transformer-lens | 2.14.1 / 5.21.0 / 4.3.0 | real-model extra only; `HookedTransformer` is gone in 4.x |

## Repository layout

```text
analysis/   Python: schema, detectors, model adapters, prompt suites, capture, bundle export
data/       Generated bundles: the demo fixture and the measured GPT-2 bundle (~5 MB)
frontend/   React + Vite + TypeScript app (React Three Fiber scene, panels, tests)
docs/       Architecture, scientific methods, UI guidelines, ADRs
```

## Documentation

- [Architecture](docs/architecture.md)
- [Scientific methods](docs/scientific-methods.md)
- [UI guidelines](docs/ui-guidelines.md)
- [ADR 0001: static bundles and parallel schemas](docs/adr/0001-static-bundle-and-parallel-schemas.md)
- [ADR 0002: measured bundles, schema 1.1.0](docs/adr/0002-measured-bundles-schema-1-1.md)

## Scientific stance

Cirqloom keeps observed correlations, published circuit annotations and causal evidence visibly distinct. High attention weight is not a connection between heads, a detector score is not a verified role, and a fixture is not a finding. See [docs/scientific-methods.md](docs/scientific-methods.md).
