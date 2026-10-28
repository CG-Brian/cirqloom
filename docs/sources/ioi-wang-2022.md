# Source record: GPT-2 Small IOI circuit (Wang et al., 2022)

Status: draft for review (M3 step 1, 2026-10-28). Nothing has been imported into a bundle yet.

## Sources checked

| Source | Identifier | What was read |
| --- | --- | --- |
| Paper | Wang, Variengien, Conmy, Shlegeris, Steinhardt. *Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small.* arXiv:2211.00593 **v1** (submitted 2022-11-01; the only arXiv version). https://arxiv.org/abs/2211.00593 | §3 intro and Figure 2 (p. 4–5), §3.1 + Figure 3 (p. 5–6), §3.2 + Figure 4 (p. 7), §3.3 + Figure 5 (p. 7–8), §3.4 (p. 9), Appendix C + Figure 12 (p. 17–18), Appendix D + Figure 13 (p. 18), Appendix F (p. 19–20), Appendix K table (p. 24–25) |
| Official code | https://github.com/redwoodresearch/Easy-Transformer (the README states it holds the code for all experiments in the paper) | `easy_transformer/ioi_circuit_extraction.py`, `CIRCUIT` dict, lines 168–201. Last change to that file: commit `e168301aa05154a14d52e5b6f98b6518200f5b1f` (2022-11-18). Repository HEAD when checked: `ea15315dd24481e9e2ac5c3ef335d82907a1dc34` |

Page numbers are those printed on the arXiv v1 PDF. Head notation in both sources is `layer.head`, zero-based (paper `9.9` = code `(9, 9)` = Cirqloom `L9H9`).

## Head membership: 26 heads, 7 classes

The paper names seven classes (p. 4–5). Figure 2 lists every head; the code's `CIRCUIT` dict holds the same 26 heads.

| Class (paper) | Heads (paper Figure 2) | Code key | Where the paper identifies them |
| --- | --- | --- | --- |
| Name Mover Heads | 9.9, 9.6, 10.0 | `"name mover"` (first 3 entries) | §3.1, Figure 3a: path patching h → logits at END |
| Negative Name Mover Heads | 10.7, 11.10 | `"negative"` | §3.1, Figure 3a (increase logit difference when patched) |
| Backup Name Mover Heads | 9.0, 9.7, 10.1, 10.2, 10.6, 10.10, 11.2, 11.9 | `"name mover"` (entries 4–11) | §3.4, Appendix F: top 8 direct effects after knocking out the Name Movers |
| S-Inhibition Heads | 7.3, 7.9, 8.6, 8.10 | `"s2 inhibition"` | §3.2, Figure 4b: path patching h → Name Mover queries |
| Induction Heads | 5.5, 6.9; in parentheses 5.8, 5.9 | `"induction"` (all four) | §3.3, Figure 5b (5.5, 6.9); Appendix C, Figure 12b (5.8, 5.9, "fuzzy") |
| Duplicate Token Heads | 0.1, 3.0; in parentheses 0.10 | `"duplicate token"` (all three) | §3.3, Figure 5b (0.1, 3.0); Appendix C, Figure 12b (0.10, "fuzzy") |
| Previous Token Heads | 2.2, 4.11 | `"previous token"` | Appendix D, Figure 13: path patching h → Induction keys at S1+1 |

Count: 3 + 2 + 8 + 4 + 4 + 3 + 2 = **26**. Paper and code agree on the set.

Differences to carry into the import:

1. **Backup Name Movers.** The code merges them into `"name mover"`; the paper keeps them as a separate class. Follow the paper (7 classes).
2. **Parenthesised heads** (0.10, 5.8, 5.9) are members, but the paper calls them "fuzzy" and finds them through S-Inhibition *keys* (Appendix C), not values. Keep this as a per-member note, not a separate class.
3. **Commented-out heads** in the code (`(7, 1)` under duplicate token; `(2, 9)`, `(4, 3)`, `(4, 7)`, `(5, 6)`, `(3, 3)`, `(3, 7)`, `(3, 6)` under previous token) are **not** members. Do not import them.

## Relations the paper supports

All relations are between **classes**, through a named input of the receiving heads (query, key or value), at named token positions. Each was found with path patching: one sender head at a time into the whole receiving class. The paper gives **no per-head-pair** results.

| From class (position) | To | Through | Evidence in paper |
| --- | --- | --- | --- |
| Name Mover Heads (END) | logits | direct effect, positive | §3.1, Figure 3a |
| Negative Name Mover Heads (END) | logits | direct effect, negative | §3.1, Figure 3a |
| S-Inhibition Heads (END) | Name Mover Heads | queries | §3.2, Figure 4b; Figure 4c patches all four S-Inhibition heads together |
| Duplicate Token Heads 0.1, 3.0 and Induction Heads 5.5, 6.9 (S2) | S-Inhibition Heads | values | §3.3, Figure 5b (four heads; the text groups them by attention pattern, and names them in Appendix C for 0.1, 3.0 and Appendix H for 5.5, 6.9) |
| Fuzzy Duplicate Token 0.10 and fuzzy Induction 5.8, 5.9 (S2) | S-Inhibition Heads | keys | Appendix C, Figure 12b |
| Previous Token Heads (S1+1) | Induction Heads | keys | Appendix D, Figure 13 |
| Backup Name Mover Heads (END) | logits | direct effect, **only after** the Name Movers are knocked out | §3.4, Appendix F |

Figure 2 itself is a schematic: its arrows show which residual streams heads read from and write to, not measured head-to-head links.

## Caveats stated by the paper

- The circuit is for the IOI task distribution (pIOI templates, Appendix E), not a general role of these heads.
- MLPs, layer norms and embeddings were not intervened on (p. 5).
- Duplicate Token and Induction functions were only partly validated (§3.3, "Missing validations and caveats"). In this circuit the Induction Heads' output acts as a position signal, unlike the original induction-head description (p. 9).
- The circuit reaches 87% of the full model's logit difference (faithfulness, §4), and Backup Name Movers show that faithfulness alone is incomplete (§3.4, §4.1).

## Notes for Cirqloom (not claims from the paper)

- Locally measured M1 scores agree with two classes: the induction detector's top heads include L5H5 and L6H9, and the previous-token detector's top head is L4H11. This is a correlation on our repeated-token suite. It does not verify the paper's IOI roles, and the two must stay separate in the UI.
- L10H7 is a Negative Name Mover here. It is also the copy-suppression example in McDougall et al. (2024). Keep these as two separate annotations with their own citations, and never label it S-Inhibition.

## Import plan (agreed 2026-10-28, schema 1.3.0)

Edges are stored at the granularity each experiment measured: one sender head at a time into a whole receiving group, except Figure 4c (all four S-Inhibition heads patched together). All are `published_experimental` with status `published`, `magnitude: null` (no numbers are read off figures), and a `locator`. `effect.direction` is the direction of the sender's own contribution, not the sign of the patching effect.

| # | Source | Target | Channel | Positions | Edges | Locator |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | head 9.9, 9.6, 10.0 | `logits` | `direct_logit`, effect logit difference ↑ | END | 3 | §3.1, Figure 3a |
| 2 | head 10.7, 11.10 | `logits` | `direct_logit`, effect logit difference ↓ | END | 2 | §3.1, Figure 3a |
| 3 | head 7.3, 7.9, 8.6, 8.10 | group Name Mover | `q` | END → END | 4 | §3.2, Figure 4b |
| 4 | group S-Inhibition | group Name Mover | `q` | END → END | 1 | §3.2, Figure 4c |
| 5 | head 0.1, 3.0, 5.5, 6.9 | group S-Inhibition | `v` | S2 → END | 4 | §3.3, Figure 5b; Appendix C (0.1, 3.0), Appendix H (5.5, 6.9) |
| 6 | head 0.10, 5.8, 5.9 | group S-Inhibition | `k` | S2 → END | 3 | Appendix C, Figure 12b |
| 7 | head 2.2, 4.11 | group Induction | `k` | S1+1 → S2 | 2 | Appendix D, Figure 13 |
| 8 | head 9.0, 9.7, 10.1, 10.2, 10.6, 10.10, 11.2, 11.9 | `logits` | `direct_logit` | END | 8, each with condition `after_knockout:name_mover` | §3.4; Appendix F, Figure 15 |

Total: 27 edges, drawn as 8 ribbons.

Not imported:

- Figure 2 arrows as `published_schematic` edges: every arrow is already covered by an experimental edge above.
- The Appendix C sentence that Duplicate Token Heads 3.0 and 0.1 influence the S-Inhibition heads with a reversed effect: it sits in the keys section but says "values", so its channel is ambiguous.
- Head-pair edges (for example 7.3 → 9.9): no head-pair result exists in the paper. They need our own weight-composition or patching evidence (§7.8, §7.9) and would be stored as separate measured edges.
