# Scientific methods

## Current state (M1)

Two bundles are committed:

- `data/bundles/gpt2-small/` (default in the app, bundle id `gpt2-small:v2`): measured attention scores for GPT-2 Small (`openai-community/gpt2` at a pinned revision), plus the **published** IOI circuit (Wang et al., 2022: 7 roles, 26 members, 27 relations) imported from `analysis/cirqloom_analysis/published/ioi/annotations.json`. The two are kept apart: published records have status `published`, a citation and a locator, and no numbers; measured records are `experimental` and come from repeated random tokens, which do not test any IOI role. Provenance lists the annotation version under `published circuit:gpt2-small:ioi:wang-2022`.
- `data/fixtures/gpt2-small-demo/` (`?bundle=fixture`): the synthetic fixture. Zero experiments and observations; one deliberately artificial circuit (diagonal heads L*k*H*k*, `origin: "synthetic_fixture"`). The UI labels it "Demo fixture — not measured model data."

### Prompt suites

- `repeated_tokens_v1`: `(BOS, X, X)`, X = L random token ids drawn uniformly from every id except `<|endoftext|>` (50256), which is used as BOS by convention.
- `nonrepeated_tokens_v1` (control): `(BOS, X, Y)` with the **same** BOS and X as the repeated prompt of the same (L, seed), and Y fresh. Every mask selects the same cells in both; only repetition differs.
- Each prompt has its own generator seeded by (salt, L, seed), so adding lengths or seeds never changes existing prompts. The committed bundle uses L = 10, 25, 50 and 8 seeds each. One suite at one length is one experiment and one batch (no padding).

### Detectors (`analysis/cirqloom_analysis/detectors/`, all v1.0.0, `heuristic_attention`)

Score = mean attention over a boolean mask of cells, per prompt; the reported score is the mean over prompts, with sample std (ddof=1) and every per-prompt score. `sample_count` is the number of prompts, not cells.

| Detector | Cells | Settings |
| --- | --- | --- |
| `previous_token_attention` | `(q, q-1)` for `q >= burn_in` | `burn_in = 2` |
| `first_token_attention` | `(q, 0)` for `q >= burn_in`; may be an attention sink, not a role | `burn_in = 2` |
| `induction_attention` | `(q, q-L+1)` for second-copy rows `q = L+1 … 2L` only; never the whole `diagonal(1-L)` | `L >= 3`, from `seq = 1 + 2L` |

Every observation has status `experimental`. A detector moves to `measured` only after its controls are reviewed; thresholds and `candidate` status come after that, per detector (never a shared cut-off).

### Controls

- **Head-side (unit tests):** ideal previous-token, sink and induction heads, uniform baseline and mixtures; each detector ranks its own head first, and the uniform baseline differs per detector and shrinks with sequence length.
- **Input-side (real model, recorded in the bundle README, not judged in code):** on the committed run, the top induction heads at L = 25 and 50 (L5H5, L6H9, L7H10, L5H1, L7H2; scores about 0.78–0.94) drop to below 0.01 on the non-repeated control, while previous-token scores (L4H11 about 0.97–0.99) do not change. At L = 10, L10H7 also enters the induction top five (0.685, control 0.008); that is an attention pattern only, and is not read as evidence about its published copy-suppression role. First-token scores are high for many heads (median about 0.5) on both suites: an attention sink is common, so a BOS score alone says little.

### Known caveats

- X can contain the same token twice (about 2.5% of prompts at L = 50); an ideal induction head may then attend to either match, which slightly lowers its score.
- Attention assets are float16 (about 3 significant digits) and cover one example prompt per measurement experiment. Scores are computed in float32 and rounded to 6 decimals.
- These are attention patterns only. Whether a head copies (OV / logit attribution) is not measured yet.

## Evidence vocabulary

| Status | Meaning |
| --- | --- |
| `published` | Taken from a cited paper annotation |
| `measured` | A score from a specified run with a validated method |
| `candidate` | A threshold or rule suggests the behaviour; not causally confirmed |
| `experimental` | The method is incomplete or exploratory (fixture memberships and all M1 detector scores) |
| `unavailable` | No reliable measurement; `score` must be `null`, never 0 |

Evidence types: `heuristic_attention`, `logit_attribution`, `published_annotation`, `structural`, and `causal_intervention` (reserved; nothing emits it in V1).

## Invariants enforced in code

- `score` is null if and only if `status` is `"unavailable"`. A non-null score requires `sample_count > 0`. Enforced by both the Python and TypeScript validators.
- A bundle marked `synthetic_fixture` must carry a display warning, and may only contain `synthetic_fixture` circuits. A real bundle may not contain fixture circuits.
- No NaN or Infinity can be serialized.
- `per_example_scores` holds one value per prompt; `score_std` is null exactly when there are fewer than 2 prompts. One observation per (experiment, detector, head).
- Each (experiment, detector) with observations has one mask: the exact `[q, k]` cells it averaged, sorted, unique and causal (`0 <= k <= q < seq`), with the same version and parameters as its observations. A mask with no observation is rejected.
- Edges are stored at the granularity their evidence was measured at (head → role group, group → group, head → logits); a group relation is never expanded into head pairs. Published edges need a locator in the citation and never carry a number read off a figure; edges we measure (weight composition, causal intervention) need a method and a magnitude, and causal edges list their conditions. See ADR 0004.
- An attention asset's shape must equal `[layers, heads, seq, seq]` for its experiment, and its file size must match (checked by `validate_bundle.py`).

## Planned (M2+)

- **Copy-suppression candidate:** attention and signed token-specific direct logit contribution, reported separately. L10H7 is a positive control to measure, not to hard-code.
- **Published IOI circuit:** memberships and roles imported from Wang et al. (2022), only after checking against the primary source. Edges are included only where the paper supports them.
