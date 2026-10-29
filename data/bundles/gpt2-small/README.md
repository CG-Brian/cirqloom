# gpt2-small:v2

GPT-2 Small bundle: measured attention experiments and published circuit annotations, kept
apart by provenance and evidence status. Regenerate with:

    python analysis/scripts/build_gpt2_bundle.py --lengths 10 25 50 \
        --seeds 8 --ioi-templates 0 5 6 \
        --ioi-seeds 8 --device cpu

## What this is

Measured here: attention-pattern detectors on repeated random tokens, with a paired non-repeated control, and attention between named positions on IOI prompts built from the paper's templates, with a paired control where S2 is a third name. Every score is 'experimental' (heuristic attention, not causal evidence). Published, not measured here: the IOI circuit of Wang et al. (2022), imported from reviewed annotations. No measured score verifies a published role.

- Model: `openai-community/gpt2` at revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`
  (12 layers × 12 heads).
- Detectors: previous-token, first-token (BOS) and induction attention masks on repeated
  tokens; on IOI prompts, attention between named positions (END→IO, END→S1, END→S2, S2→S1,
  S2→S1+1). Scores are mean masked attention per prompt, averaged over prompts (n = prompts,
  not cells). Status is `experimental` for every score. IOI attention scores are observations:
  they do not verify the paper's roles (no copying, output or causal test is run).
- IOI prompts: `analysis/cirqloom_analysis/prompts/ioi.py` (paper templates and the official
  code's names, places and objects; every filler is one GPT-2 token; positions IO, S1, S1+1,
  S2, END are checked against the tokenizer and stored in each tokens file).
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
- `ioi_templates_v1:BABA-t00` (measurement, seq 16, 8 prompts): IOI template BABA-t00 (Wang et al. 2022, Appendix E), e.g. "Then, Andrew and Thomas went to the house. Andrew gave a computer to". Names, place and object vary per prompt; positions IO, S1, S1+1, S2, END are the same for all.
- `ioi_abc_v1:BABA-t00` (control, seq 16, 8 prompts): Paired control for IOI template BABA-t00: same IO, S1, place and object as the IOI prompt of the same seed, but the S2 slot holds a third name, so no name repeats (pABC-style), e.g. "Then, Andrew and Thomas went to the house. James gave a computer to". Positions name the same template slots.
- `ioi_templates_v1:ABBA-t00` (measurement, seq 16, 8 prompts): IOI template ABBA-t00 (Wang et al. 2022, Appendix E), e.g. "Then, Amy and Shannon went to the hospital. Shannon gave a kiss to". Names, place and object vary per prompt; positions IO, S1, S1+1, S2, END are the same for all.
- `ioi_abc_v1:ABBA-t00` (control, seq 16, 8 prompts): Paired control for IOI template ABBA-t00: same IO, S1, place and object as the IOI prompt of the same seed, but the S2 slot holds a third name, so no name repeats (pABC-style), e.g. "Then, Amy and Shannon went to the hospital. Mark gave a kiss to". Positions name the same template slots.
- `ioi_templates_v1:BABA-t05` (measurement, seq 15, 8 prompts): IOI template BABA-t05 (Wang et al. 2022, Appendix E), e.g. "After Kimberly and Stephanie went to the garden, Kimberly gave a computer to". Names, place and object vary per prompt; positions IO, S1, S1+1, S2, END are the same for all.
- `ioi_abc_v1:BABA-t05` (control, seq 15, 8 prompts): Paired control for IOI template BABA-t05: same IO, S1, place and object as the IOI prompt of the same seed, but the S2 slot holds a third name, so no name repeats (pABC-style), e.g. "After Kimberly and Stephanie went to the garden, Charles gave a computer to". Positions name the same template slots.
- `ioi_templates_v1:ABBA-t05` (measurement, seq 15, 8 prompts): IOI template ABBA-t05 (Wang et al. 2022, Appendix E), e.g. "After Justin and Stephen went to the hospital, Stephen gave a ring to". Names, place and object vary per prompt; positions IO, S1, S1+1, S2, END are the same for all.
- `ioi_abc_v1:ABBA-t05` (control, seq 15, 8 prompts): Paired control for IOI template ABBA-t05: same IO, S1, place and object as the IOI prompt of the same seed, but the S2 slot holds a third name, so no name repeats (pABC-style), e.g. "After Justin and Stephen went to the hospital, Lisa gave a ring to". Positions name the same template slots.
- `ioi_templates_v1:BABA-t06` (measurement, seq 18, 8 prompts): IOI template BABA-t06 (Wang et al. 2022, Appendix E), e.g. "When Cody and Nicholas got a necklace at the school, Cody decided to give it to". Names, place and object vary per prompt; positions IO, S1, S1+1, S2, END are the same for all.
- `ioi_abc_v1:BABA-t06` (control, seq 18, 8 prompts): Paired control for IOI template BABA-t06: same IO, S1, place and object as the IOI prompt of the same seed, but the S2 slot holds a third name, so no name repeats (pABC-style), e.g. "When Cody and Nicholas got a necklace at the school, Jeremy decided to give it to". Positions name the same template slots.
- `ioi_templates_v1:ABBA-t06` (measurement, seq 18, 8 prompts): IOI template ABBA-t06 (Wang et al. 2022, Appendix E), e.g. "When Benjamin and Kenneth got a necklace at the garden, Kenneth decided to give it to". Names, place and object vary per prompt; positions IO, S1, S1+1, S2, END are the same for all.
- `ioi_abc_v1:ABBA-t06` (control, seq 18, 8 prompts): Paired control for IOI template ABBA-t06: same IO, S1, place and object as the IOI prompt of the same seed, but the S2 slot holds a third name, so no name repeats (pABC-style), e.g. "When Benjamin and Kenneth got a necklace at the garden, Jeremy decided to give it to". Positions name the same template slots.

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
| `ioi_end_to_io_attention` | `ioi_templates_v1:BABA-t00` → `ioi_abc_v1:BABA-t00` | L10H7 0.815 / 0.255, L9H9 0.795 / 0.141, L9H6 0.745 / 0.220, L11H10 0.577 / 0.206, L10H0 0.378 / 0.201 | 0.008 / 0.008 |
| `ioi_end_to_s1_attention` | `ioi_templates_v1:BABA-t00` → `ioi_abc_v1:BABA-t00` | L10H0 0.174 / 0.298, L11H2 0.137 / 0.148, L11H9 0.130 / 0.196, L10H3 0.128 / 0.166, L10H1 0.122 / 0.277 | 0.006 / 0.005 |
| `ioi_end_to_s2_attention` | `ioi_templates_v1:BABA-t00` → `ioi_abc_v1:BABA-t00` | L8H6 0.802 / 0.471, L7H9 0.482 / 0.286, L8H10 0.398 / 0.236, L8H5 0.308 / 0.109, L9H7 0.209 / 0.124 | 0.019 / 0.028 |
| `ioi_s2_to_s1_attention` | `ioi_templates_v1:BABA-t00` → `ioi_abc_v1:BABA-t00` | L3H0 0.743 / 0.003, L1H11 0.704 / 0.010, L0H5 0.611 / 0.000, L0H1 0.519 / 0.001, L0H10 0.240 / 0.070 | 0.009 / 0.010 |
| `ioi_s2_to_s1plus1_attention` | `ioi_templates_v1:BABA-t00` → `ioi_abc_v1:BABA-t00` | L5H5 0.845 / 0.007, L6H9 0.457 / 0.005, L5H8 0.395 / 0.049, L5H10 0.246 / 0.085, L5H9 0.213 / 0.123 | 0.029 / 0.026 |
| `ioi_end_to_io_attention` | `ioi_templates_v1:ABBA-t00` → `ioi_abc_v1:ABBA-t00` | L10H7 0.832 / 0.325, L9H9 0.731 / 0.346, L9H6 0.706 / 0.316, L11H10 0.674 / 0.474, L10H1 0.414 / 0.304 | 0.007 / 0.006 |
| `ioi_end_to_s1_attention` | `ioi_templates_v1:ABBA-t00` → `ioi_abc_v1:ABBA-t00` | L9H8 0.332 / 0.234, L10H2 0.225 / 0.257, L11H2 0.172 / 0.177, L10H3 0.141 / 0.194, L10H0 0.136 / 0.258 | 0.008 / 0.007 |
| `ioi_end_to_s2_attention` | `ioi_templates_v1:ABBA-t00` → `ioi_abc_v1:ABBA-t00` | L8H6 0.736 / 0.489, L7H9 0.428 / 0.328, L8H10 0.398 / 0.215, L9H7 0.229 / 0.134, L8H5 0.228 / 0.128 | 0.021 / 0.026 |
| `ioi_s2_to_s1_attention` | `ioi_templates_v1:ABBA-t00` → `ioi_abc_v1:ABBA-t00` | L3H0 0.827 / 0.001, L1H11 0.739 / 0.014, L0H5 0.594 / 0.000, L0H1 0.525 / 0.001, L5H10 0.414 / 0.036 | 0.013 / 0.014 |
| `ioi_s2_to_s1plus1_attention` | `ioi_templates_v1:ABBA-t00` → `ioi_abc_v1:ABBA-t00` | L5H5 0.942 / 0.043, L5H8 0.629 / 0.308, L5H0 0.275 / 0.181, L6H3 0.125 / 0.122, L10H4 0.123 / 0.055 | 0.025 / 0.024 |
| `ioi_end_to_io_attention` | `ioi_templates_v1:BABA-t05` → `ioi_abc_v1:BABA-t05` | L9H9 0.880 / 0.216, L10H7 0.879 / 0.341, L9H6 0.786 / 0.234, L11H10 0.763 / 0.284, L10H0 0.512 / 0.316 | 0.007 / 0.007 |
| `ioi_end_to_s1_attention` | `ioi_templates_v1:BABA-t05` → `ioi_abc_v1:BABA-t05` | L11H9 0.156 / 0.235, L11H2 0.150 / 0.127, L10H0 0.146 / 0.268, L10H10 0.113 / 0.280, L10H2 0.107 / 0.152 | 0.007 / 0.007 |
| `ioi_end_to_s2_attention` | `ioi_templates_v1:BABA-t05` → `ioi_abc_v1:BABA-t05` | L8H6 0.852 / 0.530, L7H9 0.521 / 0.300, L8H10 0.377 / 0.269, L8H5 0.367 / 0.169, L9H7 0.257 / 0.187 | 0.014 / 0.022 |
| `ioi_s2_to_s1_attention` | `ioi_templates_v1:BABA-t05` → `ioi_abc_v1:BABA-t05` | L1H11 0.765 / 0.019, L3H0 0.744 / 0.043, L0H5 0.622 / 0.001, L0H1 0.535 / 0.003, L0H10 0.254 / 0.067 | 0.009 / 0.012 |
| `ioi_s2_to_s1plus1_attention` | `ioi_templates_v1:BABA-t05` → `ioi_abc_v1:BABA-t05` | L5H5 0.844 / 0.027, L5H8 0.366 / 0.054, L6H9 0.342 / 0.004, L5H10 0.282 / 0.097, L5H9 0.238 / 0.119 | 0.028 / 0.021 |
| `ioi_end_to_io_attention` | `ioi_templates_v1:ABBA-t05` → `ioi_abc_v1:ABBA-t05` | L10H7 0.735 / 0.209, L9H9 0.661 / 0.253, L11H10 0.576 / 0.289, L9H6 0.510 / 0.194, L10H1 0.410 / 0.223 | 0.007 / 0.007 |
| `ioi_end_to_s1_attention` | `ioi_templates_v1:ABBA-t05` → `ioi_abc_v1:ABBA-t05` | L9H8 0.372 / 0.236, L10H2 0.198 / 0.258, L10H0 0.188 / 0.309, L11H10 0.165 / 0.305, L9H6 0.159 / 0.177 | 0.008 / 0.007 |
| `ioi_end_to_s2_attention` | `ioi_templates_v1:ABBA-t05` → `ioi_abc_v1:ABBA-t05` | L8H6 0.770 / 0.473, L7H9 0.394 / 0.276, L8H10 0.391 / 0.261, L8H5 0.266 / 0.141, L9H7 0.219 / 0.122 | 0.021 / 0.024 |
| `ioi_s2_to_s1_attention` | `ioi_templates_v1:ABBA-t05` → `ioi_abc_v1:ABBA-t05` | L3H0 0.778 / 0.001, L1H11 0.722 / 0.016, L0H5 0.594 / 0.000, L0H1 0.537 / 0.001, L5H10 0.314 / 0.049 | 0.012 / 0.013 |
| `ioi_s2_to_s1plus1_attention` | `ioi_templates_v1:ABBA-t05` → `ioi_abc_v1:ABBA-t05` | L5H5 0.887 / 0.034, L5H8 0.570 / 0.248, L5H0 0.283 / 0.227, L6H3 0.263 / 0.226, L7H11 0.183 / 0.190 | 0.024 / 0.020 |
| `ioi_end_to_io_attention` | `ioi_templates_v1:BABA-t06` → `ioi_abc_v1:BABA-t06` | L9H9 0.946 / 0.215, L10H7 0.876 / 0.268, L9H6 0.811 / 0.163, L11H10 0.752 / 0.284, L10H10 0.531 / 0.255 | 0.007 / 0.006 |
| `ioi_end_to_s1_attention` | `ioi_templates_v1:BABA-t06` → `ioi_abc_v1:BABA-t06` | L11H2 0.149 / 0.185, L10H0 0.105 / 0.271, L9H2 0.102 / 0.120, L11H9 0.091 / 0.176, L10H2 0.083 / 0.127 | 0.007 / 0.007 |
| `ioi_end_to_s2_attention` | `ioi_templates_v1:BABA-t06` → `ioi_abc_v1:BABA-t06` | L8H6 0.932 / 0.542, L7H9 0.699 / 0.521, L8H10 0.420 / 0.336, L9H7 0.353 / 0.261, L8H5 0.202 / 0.090 | 0.012 / 0.020 |
| `ioi_s2_to_s1_attention` | `ioi_templates_v1:BABA-t06` → `ioi_abc_v1:BABA-t06` | L3H0 0.784 / 0.003, L1H11 0.756 / 0.007, L0H5 0.615 / 0.000, L0H1 0.501 / 0.001, L0H10 0.233 / 0.060 | 0.007 / 0.009 |
| `ioi_s2_to_s1plus1_attention` | `ioi_templates_v1:BABA-t06` → `ioi_abc_v1:BABA-t06` | L5H5 0.754 / 0.009, L5H8 0.396 / 0.060, L6H9 0.351 / 0.007, L5H10 0.296 / 0.067, L5H9 0.165 / 0.104 | 0.025 / 0.016 |
| `ioi_end_to_io_attention` | `ioi_templates_v1:ABBA-t06` → `ioi_abc_v1:ABBA-t06` | L11H10 0.750 / 0.375, L10H7 0.737 / 0.208, L9H9 0.715 / 0.282, L9H6 0.589 / 0.222, L10H0 0.418 / 0.227 | 0.006 / 0.006 |
| `ioi_end_to_s1_attention` | `ioi_templates_v1:ABBA-t06` → `ioi_abc_v1:ABBA-t06` | L9H8 0.313 / 0.189, L8H3 0.295 / 0.274, L11H2 0.218 / 0.236, L10H2 0.209 / 0.234, L11H6 0.161 / 0.221 | 0.006 / 0.006 |
| `ioi_end_to_s2_attention` | `ioi_templates_v1:ABBA-t06` → `ioi_abc_v1:ABBA-t06` | L8H6 0.852 / 0.578, L7H9 0.622 / 0.489, L8H10 0.527 / 0.341, L9H7 0.440 / 0.249, L8H5 0.214 / 0.087 | 0.016 / 0.019 |
| `ioi_s2_to_s1_attention` | `ioi_templates_v1:ABBA-t06` → `ioi_abc_v1:ABBA-t06` | L3H0 0.828 / 0.002, L1H11 0.655 / 0.014, L0H5 0.587 / 0.000, L0H1 0.517 / 0.001, L5H10 0.297 / 0.042 | 0.010 / 0.011 |
| `ioi_s2_to_s1plus1_attention` | `ioi_templates_v1:ABBA-t06` → `ioi_abc_v1:ABBA-t06` | L5H5 0.927 / 0.059, L5H8 0.632 / 0.281, L6H3 0.216 / 0.225, L5H0 0.190 / 0.162, L10H4 0.190 / 0.103 | 0.016 / 0.014 |

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
