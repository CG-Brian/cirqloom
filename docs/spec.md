# Cirqloom specification

The public specification for Cirqloom: scope, scientific rules, data contract, interface rules and acceptance criteria. Code comments cite it as `docs/spec.md §N`; section numbers are stable. What is built today and what comes next is in the [roadmap](roadmap.md); design decisions are in the [ADRs](adr/).

## 0. Principles

1. **Implement vertical slices.** Each step delivers working end-to-end behaviour, starting from a deterministic fixture and then real model data.
2. **Prefer simple, testable implementations.** Cirqloom is an offline-precomputed explorer, not an online inference service.
3. **Keep scientific claims auditable.** Every score, circuit label and relation has a named method, run, parameters, provenance and evidence status. Detector results are never manufactured, and attention weights never become causal edges.
4. **Heads are understated circuit components.** A head gets a subtle ring or halo when relevant; the circuit, the selected context and the evidence are the visual focus.
5. **Separate analysis, data and frontend.** The browser app never imports PyTorch; detectors contain no rendering logic.
6. **Use versioned schemas and stable ids**, so other models, circuit types and evidence kinds can be added later.
7. **Do not overclaim.** "Induction-like attention", "published IOI role" and "measured causal effect" are different evidence states.
8. **Circuits are the product; detectors are instruments.** Head scores, masks and heatmaps exist to build, support and audit circuit-level claims.

## 1. Scope

**Model:** GPT-2 Small (`openai-community/gpt2`, pinned revision): 12 layers × 12 attention heads.

**Product:** a 3D explorer where a user chooses a circuit, sees its role groups, member heads and evidence-typed relations, selects a head or relation, and inspects the evidence (citation, measured scores with controls, real token attention) and its provenance in a 2D panel.

**Data mode:** experiment bundles precomputed in Python and served as static files.

**Out of scope:** in-browser patching or live inference; SAE features or MLP/residual nodes; model training; multi-model comparison; automatic circuit discovery; any claim that a detector score establishes a head's general role. The only non-head graph endpoints are `logits` (and `embed` if a sourced relation needs it).

## 2. Scientific distinctions

| Term | Meaning | Visual rule |
| --- | --- | --- |
| Head | One attention head (layer, index) | Small neutral node; subtle emphasis when selected or in the shown circuit |
| Functional observation | A measurement that a head shows a pattern on specified prompts | Score in the inspector with provenance |
| Circuit | A task-conditioned set of role groups, member heads and evidence-typed relations | The primary object of the UI |
| Role group | The heads that play one role in one circuit (e.g. IOI S-Inhibition) | Group hub; group-level relations attach here |
| Published annotation | Membership or relation taken from a cited paper | "Published" badge and citation |
| Published relation | A relation asserted in a paper: `published_schematic` (diagram only) or `published_experimental` (the paper's own experiment, e.g. path patching) | Source-linked, styled by kind |
| Weight composition | Input-independent score of how strongly one head's output can feed another's Q, K or V (Elhage et al., 2021), with a baseline | Never called information flow or causation (planned) |
| Observed attention alignment | On one prompt, several heads' attention rows lined up along a token path | A 2D view only; never a 3D edge, never "information transferred" |
| Causal edge (ours) | Our own precomputed intervention with a recorded design and controls; valid only under those conditions | Reserved; nothing emits it today |

- **Edge granularity follows the evidence.** A result measured from one sender head into a whole role group is a head → group relation; it is never expanded into head pairs that were not measured individually.
- **Roles are not exclusive.** A head may score high on several detectors and belong to several circuits. IOI is a circuit; Name Mover and S-Inhibition are roles in it. Copy suppression (L10H7) is not the IOI S-Inhibition role.
- **Attention is not causation.** A large attention weight does not establish a logit contribution or a connection between heads.
- **Observations are conditional** on model revision, tokenizer, prompt suite, processing and analysis version. A GPT-2 annotation never applies to another model automatically.

## 4. Stack

React + TypeScript + Vite; Three.js via React Three Fiber and drei; Tailwind CSS; Zustand for shared UI state; zod for runtime validation. Analysis in Python with PyTorch and TransformerLens (`TransformerBridge`), numpy for detectors, Pydantic for the schema. pytest + Ruff; Vitest + Testing Library; Playwright. No database, server or GPU at runtime.

## 5. Architecture and data contract

```text
GPT-2 weights / tokenizer → model adapter (only place that imports torch)
  → prompt suites + capture → numpy detectors + controls
  → published circuit importer (no model)        ↘
  → bundle exporter (validate → deterministic JSON + float16 attention assets)
  → DataProvider (zod-validated) → React / 3D explorer
```

The frontend consumes model-agnostic, versioned nodes, memberships, observations, masks and edges. It never needs tensor shapes or hook names.

- Stable ids: `model:gpt2-small:head:L9H6` (zero-based), `circuit:gpt2-small:ioi:wang-2022`. Node and edge ids are independent of positions and geometry.
- Raw scores are floats with a definition, never a probability of head identity.
- Observation evidence types: `heuristic_attention`, `logit_attribution`, `weight_composition`, `published_annotation`, `structural`, `causal_intervention`. Edge evidence kinds: `published_schematic`, `published_experimental`, `weight_composition`, `causal_intervention`.
- Provenance covers source, model revision, tokenizer, prompt suite, seeds, detector version and environment.
- Missing values stay missing (`null`), never invented. Scores from different detectors or prompt suites are not ranked together.

## 7. Analysis methods

Implementation details and the committed results are in [scientific-methods.md](scientific-methods.md).

### 7.1 Capture protocol

1. Pin model id, revision and tokenizer; evaluation mode; seeded prompt generation.
2. Deterministic prompt suites, each with a paired control.
3. Cache only what is needed (attention patterns).
4. Attention is `[batch, head, destination, source]`; rows are checked to be causal, finite and to sum to 1.
5. Scoring excludes future cells and the burn-in rows its definition requires.
6. Export the tokens needed to label a heatmap; attention matrices load lazily per experiment.
7. Aggregate per prompt: mean, sample standard deviation, n and every per-prompt score.
8. Document numeric precision (scores rounded to 6 decimals; assets float16).

### 7.2 Previous-token attention

Mean `A[q, q-1]` over rows `q >= burn_in`. A positional attention pattern, not causal proof.

### 7.3 First-token attention

Mean `A[q, 0]` over rows `q >= burn_in`. May be an attention sink rather than a role; never interpreted from the score alone.

### 7.4 Induction-like attention

On `(BOS, X, X)` with `|X| = L`, the cells `(q, q-L+1)` for second-copy rows `q = L+1 … 2L` only. Never the mean of the whole `diagonal(1-L)`, which mixes in first-copy and BOS cells. Report the mask's cell count, test several lengths and seeds, and compare against a non-repeated control. Attention alone does not show copying (§7.10).

### 7.5 Copy suppression (planned)

Defined as copy suppression, not generic suppression: attention to an earlier token that earlier components predict, plus a signed, token-specific negative direct logit contribution, reported as separate quantities against random and negative controls. L10H7 is a positive control to measure, never hard-coded. Ships as `experimental` if validation is inconclusive.

### 7.6 Published circuits (IOI)

Imported only from a verified primary source (`docs/sources/ioi-wang-2022.md`): circuit id, citation, annotation version, expected model; members with their published role group and notes; relations only where the paper supports them, typed `published_schematic` or `published_experimental`, with a locator. Each relation is stored at the granularity the paper measured it, which in Wang et al. is mostly one sender head into a whole receiving group through a named Q/K/V input at named positions (ADR 0004). Imported membership and locally measured attention stay visibly separate. Attention measured on IOI prompts is an observation, never a verification of a role.

### 7.7 Result states

`published` (cited annotation), `measured` (validated method on a specified run), `candidate` (a rule suggests the behaviour; not causally confirmed), `experimental` (exploratory method), `unavailable` (no reliable measurement; score must be `null`, never 0). No score maps to a percent certainty; a threshold only ever applies within one detector's scale.

### 7.8 Weight composition (planned)

Q-, K- and V-composition scores between an earlier and a later head from the weights, with the exact matrix product, weight processing, a random-matrix baseline and the distribution over head pairs at the same layer gap. Capacity in weight space only; never labelled causal or information flow.

### 7.9 Offline path patching (planned, gated)

A written design (clean/corrupt prompts, senders and receivers, patched input and positions, metric, controls and thresholds) is reviewed in an ADR before any implementation. Results become `causal_intervention` edges only if the controls separate them, and each edge states its conditions.

### 7.10 Direct logit analysis (planned)

Per head, the direct contribution of its output to the logit of a specified token on a specified suite, with the final-LayerNorm and unembedding convention recorded. Reported next to attention scores, never merged with them.

## 8. Bundle schema

Parallel runtime validators in Python (Pydantic) and TypeScript (zod), tied by a contract test on the generated bundles (ADR 0001). A bundle is `manifest.json` + `graph.json` + `assets/` (tokens files and float16 attention). Rules: unknown keys and NaN/Infinity are rejected; `score` is null if and only if status is `unavailable`; a synthetic fixture says `synthetic_fixture: true` and the UI shows "Demo fixture — not measured model data."; output is deterministic; no model weights are redistributed. Schema history: ADRs 0001–0005.

## 9. Visualization and interaction

- **Priorities:** evidence and legibility before spectacle; the circuit, not a glowing head, is the focus; a 2D inspector carries the details 3D cannot show.
- **Scene:** layers left to right as slim frames, 12 small neutral heads each; labels on hover, not permanent. Members of the shown circuit get a thin role ring and faint halo; non-members recede; selection adds a neutral bright ring. Heads are never permanently coloured by one category.
- **Relations:** sparse ribbons between role-group hubs, styled by evidence kind with line pattern as well as colour; arrows only where the evidence has a direction. Unknown or dense connections are never drawn.
- **Colour:** an accessible palette paired with line patterns and marker shapes; a legend; never colour alone.
- **Camera:** orbit, pan and zoom with damping; Reset view; focus on a head, relation or circuit without taking control away. Click selects (with a drag threshold), hover shows a tooltip, Escape deselects.
- **Accessibility:** a keyboard-navigable head list and relation list parallel the 3D scene; if the 3D view cannot start, every head and relation stays reachable from the panels.
- **Inspector:** identity, circuit memberships with provenance, published annotations with citations, measured scores (with n, settings, status and reference), a real attention heatmap with true token labels and exact weights, and plain-language caveats. Honest empty states for missing data.
- **Performance:** ordinary meshes for 144 heads, memoised geometry, lazily loaded attention, capped device pixel ratio; smooth orbiting is a requirement.

## 10. Frontend state

Domain data is immutable after loading and indexed once (`buildGraphView`). Shared UI state (selection, hover, active circuit, experiment, mask detector, camera command) lives in a Zustand store holding ids and flags only, read with narrow selectors. Camera pose stays in three.js objects and never re-renders React. Highlight and ranking rules are pure selectors in `features/overlays/selectors.ts`. Data access goes through the `DataProvider` interface (`StaticBundleProvider` today).

## 12. Acceptance criteria and tests

**Scientific correctness**
- Toy attention matrices check the previous-token diagonal, the first-token column and the second-copy-only induction stripe, including off-by-one and BOS cases. No future cells are ever scored.
- Shapes are validated, with clear errors.
- The same seed, model revision and configuration reproduce the bundle byte for byte on one machine (rounding is the cross-machine tolerance).
- Aggregation reports mean, n and spread; no NaN is serialised.
- IOI membership, roles and relations are checked against an independent copy of the paper's tables, with a locator on every relation; no head-pair relation exists without head-pair evidence.
- Every result carries model revision, method version and evidence status.

**UI behaviour**
- 12 × 12 heads render with deterministic ids; orbit, pan, zoom, reset and focus work, and selection survives camera moves.
- Selecting a head opens the inspector and loads only the needed attention asset.
- A circuit overlay changes subtle emphasis and the legend; it never generates edges. Relations open from the 3D scene and from a keyboard list.
- Heatmap labels match the real tokens and the row/column orientation.
- No data, failed assets and unverified states show informative messages.

**Tooling:** `ruff check`, `pytest`, `tsc --noEmit`, ESLint, Vitest, production build and Playwright smoke tests before a release. Normal test runs never download model weights; real-model tests are opt-in (`CIRQLOOM_REAL_MODEL=1`) with a pinned revision.

## 13. Development practice

- The README gives copy-paste setup for the frontend alone and for regenerating bundles.
- Lockfiles; critical versions pinned; tested Python and Node versions documented.
- Model weights stay in the Hugging Face cache and are never committed. Bundles are committed only while the static demo needs them (ADR 0006).
- The public demo contains no secrets or private prompts.
- ADRs record method, schema and visualisation decisions.
- No silent failures: failed experiments, unsupported configurations and missing assets are surfaced.

## 14. Extension contracts

- **New model:** implement `BaseModelAdapter` (architecture metadata, tokenization, validated attention) and build a bundle; the UI takes layer and head counts from the manifest.
- **New detector:** register a `DetectorSpec` (id, version, prompt suite, control suites, mask builder, evidence type, maturity); it emits observations and never touches the scene.
- **New published circuit:** import from a traceable source through a reviewed annotation file; validation checks model compatibility and a locator on every relation; the overlay, graph and inspectors are reused.
- **Causal interventions:** a separately typed `causal_intervention` record from a specified clean/corrupt contrast, patch site, metric and effect. Existing published or composition edges are never upgraded to causal.

## 15. Anti-patterns

- Making every head glow, or constant moving particles.
- Treating a head as belonging to only one circuit.
- Calling `A[q, k]` a connection between two heads: it is token-to-token attention inside one head.
- Implying a wiring diagram from attention matrices.
- Using `diagonal(1-L).mean()` over the whole matrix as an induction score.
- Confusing copy suppression with IOI S-Inhibition, or name attention with verified name moving.
- Adding unsourced relations to make a flow graph look complete, or expanding a group relation into head pairs.
- One arbitrary threshold (e.g. 0.4) across detectors.
- Presenting fixture data as GPT-2 findings.
- Calling weight composition or attention alignment "information flow", or showing a causal edge outside its measured conditions.
- Switching to an incompatible TransformerLens API without the adapter, a pin and verification.
