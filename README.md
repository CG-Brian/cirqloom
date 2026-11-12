# Cirqloom

An evidence-aware, interactive 3D atlas of attention heads and circuits in pretrained transformers, starting with GPT-2 Small.

**Live demo:** https://cg-brian.github.io/cirqloom/ (static site; no install, no model weights). The synthetic demo fixture is at `?bundle=fixture`.

> **Status: M3 release.** The app opens the GPT-2 Small bundle: the published IOI circuit of Wang et al. (2022) drawn as a 3D circuit graph (26 heads, 7 roles, 27 sourced relations), and GPT-2's own attention measured on repeated random tokens and on the paper's IOI prompts, each with a paired control. Next milestones are in the [roadmap](docs/roadmap.md).

### Three kinds of evidence, kept apart

| Kind | Where it comes from | How the app labels it |
| --- | --- | --- |
| Published | Wang et al. (2022): IOI membership, roles and relations, each with a figure or section locator. Cirqloom did not run these experiments. | "Published annotation" / "Published experiment" badges, citation and locator |
| Measured here | Attention-pattern scores computed by this repository's pipeline on GPT-2 Small, with paired controls | "Experimental" badge, n, settings, control runs |
| Synthetic | The demo fixture used for tests (`?bundle=fixture`) | "Demo fixture — not measured model data." |

No score in the app is causal evidence produced by Cirqloom, and no measured score verifies a published role.

## First visit: three things to try

The same guide appears in the right-hand panel until you select something.

1. **Explore an attention head.** Click a head in the 3D view, or in the Heads list on the left (arrow keys work there). The inspector shows its scores for the selected experiment and an attention heatmap: rows are destination tokens, columns are source tokens. "Show cells" outlines the exact cells a score averaged. The Detectors section on the left lists each detector's top heads as shortcuts.
2. **Pick an experiment.** The Experiment menu at the top switches between repeated random tokens (L = 10, 25, 50), IOI sentences built from the paper's templates, and the paired control for each. Compare a head on a measurement run and on its control: an induction head such as L5H5 drops to near zero when nothing repeats.
3. **Inspect a published relation.** Turn on "Indirect Object Identification (IOI)" in the left panel, then open a relation from its Relations list or click a ribbon. The relation inspector shows the paper's claim (channel, positions, locator, every evidence record) separately from "Observed attention (measured here)" on an IOI prompt.

## What works today

- A 3D scene of GPT-2 Small's 12 layers × 12 heads, with orbit, pan and zoom, Reset view, and camera focus on a circuit, a relation or a head.
- Circuit overlays with subtle role rings: members get a thin ring, non-members recede, spanned layers are outlined.
- The published IOI circuit (Wang et al., 2022) as a circuit graph: 7 role-group hubs above the layers and 8 evidence-typed ribbons bundling its 27 sourced records, stored at the paper's granularity (head → role group, one group → group, head → logits; never expanded into head pairs), with Q/K/V channel labels. A relation inspector lists every record with its figure.
- Head selection from the 3D scene or a keyboard-navigable head list; both stay in sync.
- A head inspector that keeps circuit membership, published annotations, measured observations and attention separate. Missing data is stated as missing, never shown as zero.
- An experiment picker. Scores are shown for one experiment at a time (mean ± sd over prompts, n, detector settings); control runs are labelled.
- An attention heatmap for the selected head on one stored prompt: row = destination, column = source, fixed 0–1 colour scale, hatched future cells, exact weight and both tokens on hover or arrow keys, named IOI positions, and "Show cells" for the exact cells a detector averaged (exported by the Python pipeline).
- Observed attention alignment in the relation inspector: a relation's heads at their IOI positions on one prompt (observations only, not information flow).
- Detector discovery: each detector's top three heads on the selected experiment, ranked on that detector's own scale.
- A first-visit guide, and a fallback that keeps every panel usable if WebGL is unavailable.
- A Python pipeline: GPT-2 adapter on TransformerLens `TransformerBridge` (pinned revision), deterministic prompt suites with paired controls, numpy detectors, and a deterministic bundle exporter with lazily loaded float16 attention assets.
- A versioned bundle schema in Python (Pydantic) and TypeScript (zod), tied by a cross-language contract test.

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
.venv/bin/python analysis/scripts/validate_bundle.py data/bundles/gpt2-small
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
CIRQLOOM_REAL_MODEL=1 ../.venv/bin/pytest tests/test_gpt2_adapter.py   # optional, needs weights

# Frontend (from frontend/)
npm run lint
npm run typecheck
npm test
npm run build
npx playwright install chromium   # first time only
npm run e2e                       # browser smoke tests; starts its own dev server on port 5199
npm run e2e:pages                 # the same tests against a production build under /cirqloom/
```

## Deployment (GitHub Pages)

The demo is a static site: `npm run build` writes the app and both bundles to `frontend/dist`. `CIRQLOOM_BASE_PATH` sets the subpath it is served under (default `/`). `.github/workflows/pages.yml` builds for `/<repo>/`, runs the browser tests against that build, and deploys it. One-time setup: **Settings → Pages → Build and deployment → Source: GitHub Actions**. See [ADR 0006](docs/adr/0006-github-pages-deployment.md).

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
data/       Generated bundles: the demo fixture and the measured GPT-2 bundle (~12 MB; ~0.5 MB of it gzipped JSON)
frontend/   React + Vite + TypeScript app (React Three Fiber scene, panels, tests)
docs/       Architecture, scientific methods, UI guidelines, ADRs
```

## Documentation

- [Specification](docs/spec.md): scope, scientific rules, data contract and acceptance criteria
- [Roadmap](docs/roadmap.md): what is built and what comes next
- [Architecture](docs/architecture.md)
- [Scientific methods](docs/scientific-methods.md): prompt suites, detectors, controls and the committed results
- [UI guidelines](docs/ui-guidelines.md)
- [Source record: IOI circuit (Wang et al., 2022)](docs/sources/ioi-wang-2022.md)
- ADRs: [0001 static bundles and parallel schemas](docs/adr/0001-static-bundle-and-parallel-schemas.md) · [0002 measured bundles](docs/adr/0002-measured-bundles-schema-1-1.md) · [0003 detector masks](docs/adr/0003-detector-masks-schema-1-2.md) · [0004 evidence-typed circuit edges](docs/adr/0004-circuit-edges-schema-1-3.md) · [0005 IOI prompts and positions](docs/adr/0005-ioi-prompts-and-positions-schema-1-4.md) · [0006 GitHub Pages deployment](docs/adr/0006-github-pages-deployment.md)

## Scientific stance

Cirqloom keeps observed correlations, published circuit annotations and causal evidence visibly distinct. High attention weight is not a connection between heads, a detector score is not a verified role, and a fixture is not a finding. See [docs/scientific-methods.md](docs/scientific-methods.md).
