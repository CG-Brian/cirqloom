# gpt2-small:v2

GPT-2 Small bundle: measured attention experiments and published circuit annotations, kept
apart by provenance and evidence status. Regenerate with:

    python analysis/scripts/build_gpt2_bundle.py --lengths 10 25 50 \
        --seeds 8 --device cpu

## What this is

Measured here: attention-pattern detectors on repeated random tokens, with a paired non-repeated control; every score is 'experimental' (heuristic attention, not causal evidence). Published, not measured here: the IOI circuit of Wang et al. (2022), imported from reviewed annotations. Repeated-token scores do not test any head's IOI role.

- Model: `openai-community/gpt2` at revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`
  (12 layers × 12 heads).
- Detectors: previous-token, first-token (BOS) and induction attention masks
  (`analysis/cirqloom_analysis/detectors/`). Scores are mean masked attention per prompt,
  averaged over prompts (n = prompts, not cells). Status is `experimental` for every score.
- Attention assets: one prompt per measurement experiment, float16
  `[layer, head, q, k]`. Weights from the GPT-2 model card license apply to the model; this
  bundle contains derived attention values only, no weights.

## Published circuits (imported, not measured here)

- `circuit:gpt2-small:ioi:wang-2022` (published): 7 roles, 26 member heads, 27 relations, all `published_experimental` with a locator and no magnitude. Wang, Variengien, Conmy, Shlegeris, Steinhardt (2022). Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small. arXiv:2211.00593v1. https://arxiv.org/abs/2211.00593

Imported from `analysis/cirqloom_analysis/published/`; sources and checks in `docs/sources/`.
The repeated-token experiments below do not test these roles.

## Experiments

- `repeated_tokens_v1:L10` (measurement, seq 21, 8 prompts): (BOS, X, X): X is 10 random tokens repeated once.
- `nonrepeated_tokens_v1:L10` (control, seq 21, 8 prompts): (BOS, X, Y): same BOS and X as the repeated suite, then 10 fresh random tokens. Control: no repetition.
- `repeated_tokens_v1:L25` (measurement, seq 51, 8 prompts): (BOS, X, X): X is 25 random tokens repeated once.
- `nonrepeated_tokens_v1:L25` (control, seq 51, 8 prompts): (BOS, X, Y): same BOS and X as the repeated suite, then 25 fresh random tokens. Control: no repetition.
- `repeated_tokens_v1:L50` (measurement, seq 101, 8 prompts): (BOS, X, X): X is 50 random tokens repeated once.
- `nonrepeated_tokens_v1:L50` (control, seq 101, 8 prompts): (BOS, X, Y): same BOS and X as the repeated suite, then 50 fresh random tokens. Control: no repetition.

## Input-side control (recorded, not judged)

Same heads on repeated input and on the paired non-repeated control. A content-dependent
head should drop on the control; a positional head should not. These numbers are reported
as measured; no pass/fail threshold is applied.

| Detector | Measurement → control | Top heads on measurement (measured / control) | Median of all heads (measured / control) |
| --- | --- | --- | --- |
| `previous_token_attention` | `repeated_tokens_v1:L10` → `nonrepeated_tokens_v1:L10` | L4H11 0.974 / 0.984, L3H7 0.538 / 0.542, L2H2 0.525 / 0.493, L6H8 0.453 / 0.327, L3H2 0.415 / 0.408 | 0.046 / 0.051 |
| `first_token_attention` | `repeated_tokens_v1:L10` → `nonrepeated_tokens_v1:L10` | L5H7 0.926 / 0.907, L4H10 0.891 / 0.884, L6H1 0.864 / 0.858, L8H11 0.855 / 0.756, L8H4 0.827 / 0.834 | 0.540 / 0.630 |
| `induction_attention` | `repeated_tokens_v1:L10` → `nonrepeated_tokens_v1:L10` | L5H5 0.908 / 0.005, L7H10 0.841 / 0.004, L6H9 0.825 / 0.004, L5H1 0.818 / 0.000, L10H7 0.685 / 0.008 | 0.038 / 0.014 |
| `previous_token_attention` | `repeated_tokens_v1:L25` → `nonrepeated_tokens_v1:L25` | L4H11 0.990 / 0.984, L3H7 0.527 / 0.545, L6H8 0.508 / 0.323, L2H2 0.506 / 0.485, L5H6 0.406 / 0.354 | 0.036 / 0.041 |
| `first_token_attention` | `repeated_tokens_v1:L25` → `nonrepeated_tokens_v1:L25` | L5H7 0.940 / 0.930, L8H4 0.898 / 0.901, L7H8 0.880 / 0.895, L6H1 0.862 / 0.874, L7H9 0.858 / 0.944 | 0.491 / 0.567 |
| `induction_attention` | `repeated_tokens_v1:L25` → `nonrepeated_tokens_v1:L25` | L5H5 0.918 / 0.004, L6H9 0.900 / 0.003, L7H10 0.888 / 0.002, L5H1 0.884 / 0.000, L7H2 0.777 / 0.003 | 0.012 / 0.003 |
| `previous_token_attention` | `repeated_tokens_v1:L50` → `nonrepeated_tokens_v1:L50` | L4H11 0.990 / 0.987, L3H7 0.539 / 0.546, L6H8 0.494 / 0.323, L2H2 0.490 / 0.479, L5H6 0.408 / 0.340 | 0.031 / 0.038 |
| `first_token_attention` | `repeated_tokens_v1:L50` → `nonrepeated_tokens_v1:L50` | L5H7 0.928 / 0.932, L8H4 0.908 / 0.918, L7H8 0.892 / 0.887, L7H4 0.887 / 0.916, L7H9 0.872 / 0.958 | 0.460 / 0.536 |
| `induction_attention` | `repeated_tokens_v1:L50` → `nonrepeated_tokens_v1:L50` | L5H5 0.940 / 0.002, L6H9 0.918 / 0.002, L7H10 0.911 / 0.002, L5H1 0.911 / 0.000, L7H2 0.836 / 0.003 | 0.005 / 0.001 |

## Environment

- bos_token: 50256 (<|endoftext|>, used as BOS by convention)
- device: cpu
- dtype: float32
- numpy: 2.5.4
- precision: scores rounded to 6 decimals; attention assets float16
- prepend_bos: true
- published circuit:gpt2-small:ioi:wang-2022: IOI annotations v1.0.0 (arXiv:2211.00593v1; code https://github.com/redwoodresearch/Easy-Transformer@e168301; verified 2026-10-28, see docs/sources/ioi-wang-2022.md)
- python: 3.14.7
- torch: 2.14.1
- transformer-lens: 4.3.0
- transformers: 5.21.0
