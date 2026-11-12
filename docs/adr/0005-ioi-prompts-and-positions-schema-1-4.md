# ADR 0005: IOI prompt suite and named token positions (schema 1.4.0)

Status: accepted, 2026-10-28

## Context

M3 step 4 runs GPT-2 Small on the IOI task itself, so the published IOI circuit (ADR 0004) can be compared with attention measured on the paper's own kind of prompt. IOI analysis is about named token positions: the indirect object IO, the subject's two occurrences S1 and S2, the token after S1 (S1+1), and END (the last token before the answer). The frontend must not infer these positions from text or suite names. The detectors need them to choose cells.

## Decision

- **Prompts** (`prompts/ioi.py`). The 15 BABA templates of Figure 14 (identical to the official code's `BABA_TEMPLATES`), plus ABBA (first `[B]` and `[A]` swapped). Names, places and objects come from the official code (`ioi_dataset.py` @ `e168301`). Appendix E mentions 20 places and objects; the code lists 8 of each, and the code's lists are used. A prompt stops before the answer, so its last token is " to".
- **Positions are decided in Python and checked against the tokenizer.** Every filler must be one token after a space; fillers that are not are dropped and reported. IO must occur exactly once, S exactly twice, and the last token must be " to". All prompts of one experiment must have the same length and positions, or building fails. GPT-2 keeps all 99 names, 8 places and 8 objects, and all 15 templates × 2 patterns pass (opt-in real-tokenizer test).
- **Experiments**: one template and pattern = one experiment, `ioi_templates_v1:{BABA|ABBA}-tNN`, kind `measurement`, attention stored for prompt 0. The default build uses templates 0, 5 and 6 × both patterns, with 8 prompts each.
- **Schema 1.4.0**: `TokenizedPrompt` gains optional `text` and `positions` (name → index). They are written only for prompts that have them, so the repeated-token tokens files stay byte-identical. `validate_bundle_files` checks that positions are in range and identical across an experiment's prompts.
- **Position-reading detectors**: `DetectorSpec.position_parameters` maps `build_mask` arguments to position names, e.g. `{"query": "END", "key": "IO"}`. The runner fills them from the prompts' shared positions, and they are recorded as integers in each observation's `parameters` and in the exported mask (a single cell). Five detectors are defined, all `heuristic_attention` and `experimental`: END→IO, END→S1, END→S2, S2→S1, S2→S1+1. Their descriptions say that the paper reports the pattern for a class, and that attention alone does not test the role.
- **Frontend**: the heatmap labels named positions on both axes and in the readout, and shows the prompt text. The relation inspector gains "Observed attention (measured here)". On the selected experiment's prompt, each sender head's row comes from the relation's sender position and each receiving member's row from its receiver position, across all named positions. The view is explicitly labelled as separate observations that do not show information passing between heads. `DataProvider.loadExperimentPositions` reads positions from the tokens file, so the UI can offer experiments that have the needed positions without guessing from names.

## Consequences

- 1.3.0 bundles no longer load. Both bundles were regenerated. In the GPT-2 bundle, every repeated-token observation, mask, experiment and asset is unchanged. Six IOI experiments were added: 4,320 observations, 30 masks and 12 asset files. The bundle grows from 5.4 MB to 8.7 MB, mostly from per-prompt scores in `graph.json`.
- ~~No IOI control suite yet.~~ Added in M3 step 5 (below).
- Attention on one stored prompt is shown, while scores average 8 prompts. Both are labelled.

## Addendum (M3 step 5): paired ABC control

- Suite `ioi_abc_v1`, one experiment per IOI experiment (`ioi_abc_v1:{pattern}-tNN`, kind `control`, scores only). Each prompt shares its seed, template, IO, S1, place and object with the IOI prompt it pairs with. The S2 slot holds a third name C (drawn after the shared draws), so no name repeats. The positions are identical: they name the same template slots, and in the control S2 is C's position.
- This differs from the paper's pABC, which draws three fresh names. Keeping IO and S1 makes the comparison paired, like the repeated-token control. All five IOI detectors list `ioi_abc_v1` as their control suite, and the bundle README reports measurement vs control per pair, without a pass/fail rule.
- The frontend's control hint offers every same-length measurement run (BABA and ABBA share lengths), instead of guessing a single pair from ids.
- With the control the bundle is 12 MB (`graph.json` 7.5 MB, about 0.5 MB gzipped); see ADR 0006.
