# ADR 0004: Evidence-typed circuit edges (schema 1.3.0)

Status: accepted, 2026-10-28

## Context

Cirqloom is a circuit explorer: circuits are drawn as role groups, member heads and the relations between them. The M0 edge was `source_node_id → target_node_id` with an `edge_type`, so it could only connect two heads. Published evidence does not have that shape. In Wang et al. (2022) almost every relation was measured by patching **one sender head at a time into a whole receiving role group**, through one input of the receivers (query, key or value), at named token positions. One experiment (Figure 4c) patched a whole group into a group. Effects on the output go into the logits, not into a head.

Storing these as head pairs would claim pair-level results the paper never measured. Storing them only as group pairs would over-cover members that were found by a different experiment (for example the "fuzzy" IOI heads, found through S-Inhibition keys rather than values).

## Decision

- **Endpoints** are a discriminated union on `kind`: `head` (`node_id`), `role_group` (`circuit_id`, `role`) or `special` (`logits` or `embed`).
- **An edge is stored at the granularity its evidence was measured at**: head → group, group → group, head → logits, and head → head only for head-pair evidence. The UI may bundle edges that share receiver, channel and source into one ribbon, but the stored edges stay separate.
- **Edge fields:** `circuit_id`, `source`, `target`, `channel` (`q`, `k`, `v`, `direct_logit`), `positions` (token roles), `evidence_kind`, `evidence_status`, `effect` (metric and the direction of the **source's contribution**, not the sign of a patching effect), `magnitude`, `conditions`, `locator` (where in the circuit's citation), `method_reference`, `description`.
- **Evidence kinds:** `published_schematic`, `published_experimental`, `weight_composition`, `causal_intervention`. `observed_attention_alignment` is a view, not an edge kind.
- **Validator rules:**
  - Endpoints exist. A head endpoint is a member of the edge's circuit. A role group belongs to the edge's circuit and is declared in it. `logits` is only a target, and `embed` is only a source. Source and target differ, and edge ids are unique.
  - `direct_logit` is used for exactly the edges into `logits`.
  - Published kinds belong to a `published` circuit, have status `published` and a non-empty `locator`. `magnitude` may be null, because numbers are never read off a figure.
  - Measured kinds (`weight_composition`, `causal_intervention`) never have status `published`, and need a `method_reference` and a `magnitude`. Weight composition has no `effect`. A causal edge must list its `conditions`.
- `Circuit.origin` gains `measured_reconstruction`. `Membership` gains `note`.

## Consequences

- 1.2.0 bundles no longer load. Both bundles were regenerated. They had no edges, so only `schema_version` and `note: null` on fixture memberships changed. Scores, masks and assets are unchanged.
- The rule "group relations are never expanded into head pairs" is a data convention checked in review and tests, because a validator cannot tell whether a head-pair edge had head-pair evidence. It can only require that measured kinds name their method.
- Cross-circuit edges (for example L10H7 in IOI and in copy suppression) are not edges. They show up as the same head with memberships in two circuits.
