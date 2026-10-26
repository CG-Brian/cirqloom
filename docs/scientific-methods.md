# Scientific methods

## Current state (M0)

**No detectors have been implemented or run.** The only bundle is the synthetic fixture `data/fixtures/gpt2-small-demo/`. It contains:

- 144 head nodes with GPT-2 Small's shape (12 layers × 12 heads, zero-based ids);
- zero experiments, zero observations, zero edges;
- one circuit, `circuit:gpt2-small:fixture-demo`, with `origin: "synthetic_fixture"`. Its members are the diagonal heads L*k*H*k*, with roles alternating by layer parity. The pattern is deliberately artificial, so it cannot be mistaken for an induction, IOI or copy-suppression finding.
- `model.revision: null`, because no weights were loaded.

The UI labels this bundle everywhere ("Demo fixture — not measured model data.") and shows "no experiments" / "no attention data" instead of any number.

## Evidence vocabulary

| Status | Meaning |
| --- | --- |
| `published` | Taken from a cited paper annotation |
| `measured` | A score from a specified run with a validated method |
| `candidate` | A threshold or rule suggests the behaviour; not causally confirmed |
| `experimental` | The method is incomplete or exploratory (fixture memberships use this) |
| `unavailable` | No reliable measurement; `score` must be `null`, never 0 |

Evidence types: `heuristic_attention`, `logit_attribution`, `published_annotation`, `structural`, and `causal_intervention` (reserved; nothing emits it in V1).

## Invariants enforced in code

- `score` is null if and only if `status` is `"unavailable"`. A non-null score requires `sample_count > 0`. Enforced by both the Python and TypeScript validators.
- A bundle marked `synthetic_fixture` must carry a display warning, and may only contain `synthetic_fixture` circuits. A real bundle may not contain fixture circuits.
- No NaN or Infinity can be serialized.

## Planned detectors (M1+)

These follow AGENTS.md §7; each will get a versioned spec and controls here before results are trusted.

- **Previous-token attention:** mean `A[q, q-1]` over valid `q ≥ 1`.
- **First-token / BOS attention:** mean `A[q, 0]` over non-BOS destinations. This may reflect an attention sink, not a semantic role.
- **Induction-like attention:** on `(BOS, X, X)` with `|X| = L`, the mean over second-copy destinations only (`q = 1+L … 2L`, source `k = q-L+1`), with the mask support count reported. Shuffled and non-repeated controls are required.
- **Copy-suppression candidate:** attention and signed token-specific direct logit contribution, reported separately. L10H7 is a positive control to measure, not to hard-code.
- **Published IOI circuit:** memberships and roles imported from Wang et al. (2022), only after checking against the primary source. Edges are included only where the paper supports them.
