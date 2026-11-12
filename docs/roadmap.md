# Roadmap

Status of Cirqloom's milestones. The [specification](spec.md) defines the rules every milestone follows; the [ADRs](adr/) record the decisions.

## Current release: M3 (IOI circuit explorer)

Cirqloom 1.0 is the M3 release: the published IOI circuit as an explorable 3D graph, next to attention measured on GPT-2 Small with paired controls.

| Milestone | Scope | Status |
| --- | --- | --- |
| M0 | Data contract (parallel Pydantic / zod schemas, validators, deterministic writer), synthetic fixture, 3D scene of 12 × 12 heads, head list, inspector, CI | Done |
| M1 | GPT-2 adapter (`TransformerBridge`, pinned revision), repeated-token suites with a paired control, previous-token / first-token / induction detectors with masks, measured bundle | Done |
| M2 | Detector masks in the bundle and an attention heatmap that outlines the scored cells | Done (per-detector 3D overlays and thresholds moved to a later step) |
| M3 | Evidence-typed edge contract; the published IOI circuit (26 heads, 7 role groups, 27 sourced relations); role-group hubs and relation ribbons; relation inspector; IOI prompt suite with named positions and a paired control; observed attention alignment; camera focus | Done |
| Release hardening | Public spec and roadmap, GitHub Pages deployment under a subpath, first-time visitor guide, WebGL fallback, detector top heads in the sidebar, wording review | Done in this release |

### What the release shows, and what it does not

- **Published (from Wang et al., 2022):** IOI membership, roles and relations, each with a figure or section locator. Cirqloom did not run these experiments.
- **Measured here:** attention-pattern scores on repeated random tokens and on IOI template prompts, each with a paired control, all labelled `experimental`.
- **Not measured yet:** what any head writes (direct logit contribution), weight composition, or any intervention. No relation in the app is causal evidence produced by Cirqloom.

## Next (not started)

| Milestone | Scope | Notes |
| --- | --- | --- |
| M4 | Induction circuit reconstruction: weight composition with baselines (spec §7.8), direct logit analysis of copying (§7.10), an attention-alignment view on repeated tokens, a `measured_reconstruction` circuit | Offline path patching (§7.9) only after its design ADR is reviewed |
| M5 | Copy suppression around L10H7 (McDougall et al., 2024): source record, QK and OV evidence on probe prompts with controls, link to its IOI Negative Name Mover annotation | Ships as `experimental` if validation is inconclusive |
| M6 | Detector instruments (per-detector 3D overlays and thresholds), accessibility audit, performance (code-splitting, compact bundle JSON) | |

## Not planned

SAE features, MLP or residual-stream nodes, other models, in-browser patching or live inference, and automatic circuit discovery (spec §1).
