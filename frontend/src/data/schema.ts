// Mirrors analysis/cirqloom_analysis/schema/bundle.py. Keep the two in sync:
// src/tests/schema.test.ts parses the Python-generated fixture to prove it.
import { z } from "zod";

export const EvidenceStatusSchema = z.enum([
  "published",
  "measured",
  "candidate",
  "experimental",
  "unavailable",
]);

export const EvidenceTypeSchema = z.enum([
  "heuristic_attention",
  "logit_attribution",
  "published_annotation",
  "structural",
  "causal_intervention",
]);

const SchemaVersionSchema = z.literal("1.3.0");

export const ModelInfoSchema = z.strictObject({
  id: z.string(),
  hf_id: z.string(),
  revision: z.string().nullable(),
  layers: z.int().positive(),
  heads_per_layer: z.int().positive(),
  tokenizer_id: z.string(),
});

/** Raw attention of one example prompt: little-endian float16, C order [layer, head, q, k]. */
export const AttentionAssetSchema = z.strictObject({
  uri: z.string(),
  dtype: z.literal("float16"),
  shape: z.array(z.int().positive()).length(4),
  prompt_index: z.int().nonnegative(),
});

/** One prompt suite at one sequence length, scored on every prompt. */
export const ExperimentSchema = z.strictObject({
  id: z.string(),
  prompt_suite: z.string(),
  kind: z.enum(["measurement", "control"]),
  description: z.string(),
  seeds: z.array(z.int()).min(1),
  sequence_length: z.int().positive(),
  tokens_uri: z.string(),
  attention: AttentionAssetSchema.nullable(),
});

/** tokens_uri contents: every prompt of one experiment, ids and display strings. */
export const TokensFileSchema = z.strictObject({
  experiment_id: z.string(),
  prompts: z.array(
    z.strictObject({
      id: z.string(),
      seed: z.int(),
      token_ids: z.array(z.int().nonnegative()),
      tokens: z.array(z.string()),
    }),
  ),
});

export const NodeSchema = z.strictObject({
  id: z.string(),
  type: z.literal("attention_head"),
  layer: z.int().nonnegative(),
  head: z.int().nonnegative(),
});

export const ObservationSchema = z.strictObject({
  node_id: z.string(),
  experiment_id: z.string(),
  detector_id: z.string(),
  detector_version: z.string(),
  evidence_type: EvidenceTypeSchema,
  /** Exact detector settings used, e.g. { burn_in: 2 }. */
  parameters: z.record(z.string(), z.int()),
  /** Mean over examples. */
  score: z.number().nullable(),
  /** Sample std over examples; null when fewer than 2. */
  score_std: z.number().nullable(),
  per_example_scores: z.array(z.number()),
  status: EvidenceStatusSchema,
  sample_count: z.int().nonnegative(),
  method_reference: z.string(),
});

/**
 * The attention cells one detector averaged in one experiment; the same for every head.
 * Built by the Python pipeline from the exact mask it scored with. The UI only draws it.
 */
export const DetectorMaskSchema = z.strictObject({
  experiment_id: z.string(),
  detector_id: z.string(),
  detector_version: z.string(),
  parameters: z.record(z.string(), z.int()),
  /** [q, k] = [destination, source], sorted and unique (checked by the Python validator). */
  cells_qk: z.array(z.tuple([z.int().nonnegative(), z.int().nonnegative()])).min(1),
});

export const RoleDefSchema = z.strictObject({
  id: z.string(),
  label: z.string(),
  description: z.string(),
});

export const CircuitSchema = z.strictObject({
  id: z.string(),
  name: z.string(),
  origin: z.enum(["synthetic_fixture", "published", "measured_reconstruction"]),
  description: z.string(),
  citation: z.string().nullable(),
  roles: z.array(RoleDefSchema),
});

export const MembershipSchema = z.strictObject({
  circuit_id: z.string(),
  node_id: z.string(),
  role: z.string(),
  source_reference: z.string(),
  evidence_status: EvidenceStatusSchema,
  /** e.g. "fuzzy member (Appendix C)" */
  note: z.string().nullable(),
});

/**
 * How an edge's evidence was obtained. A relation is stored at the granularity its evidence was
 * measured at (head → group, group → group, ...) and never expanded into head pairs.
 */
export const EdgeEvidenceKindSchema = z.enum([
  "published_schematic",
  "published_experimental",
  "weight_composition",
  "causal_intervention",
]);

/** Which input of the receiver the relation goes through, or a direct effect on the output. */
export const EdgeChannelSchema = z.enum(["q", "k", "v", "direct_logit"]);

export const EndpointSchema = z.discriminatedUnion("kind", [
  z.strictObject({ kind: z.literal("head"), node_id: z.string() }),
  z.strictObject({ kind: z.literal("role_group"), circuit_id: z.string(), role: z.string() }),
  z.strictObject({ kind: z.literal("special"), id: z.enum(["logits", "embed"]) }),
]);

export const EdgeSchema = z.strictObject({
  id: z.string(),
  circuit_id: z.string(),
  source: EndpointSchema,
  target: EndpointSchema,
  channel: EdgeChannelSchema,
  /** Token roles where the relation applies, e.g. sender at "S2", receiver at "END". */
  positions: z.strictObject({ source: z.string().nullable(), target: z.string().nullable() }).nullable(),
  evidence_kind: EdgeEvidenceKindSchema,
  evidence_status: EvidenceStatusSchema,
  /** Direction of the SOURCE's contribution to a metric (not the sign of a patching effect). */
  effect: z.strictObject({ metric: z.string(), direction: z.enum(["increase", "decrease"]) }).nullable(),
  /** null for published edges: never digitised from a figure. */
  magnitude: z.number().nullable(),
  conditions: z.array(z.string()),
  /** Where in the circuit's citation, e.g. "§3.1, Figure 3a, p. 5-6". */
  locator: z.string().nullable(),
  method_reference: z.string().nullable(),
  description: z.string(),
});

export const ProvenanceSchema = z.strictObject({
  generator: z.string(),
  generator_version: z.string(),
  notes: z.string(),
  environment: z.record(z.string(), z.string()),
});

export const ManifestSchema = z.strictObject({
  schema_version: SchemaVersionSchema,
  bundle_id: z.string(),
  synthetic_fixture: z.boolean(),
  display_warning: z.string().nullable(),
  model: ModelInfoSchema,
  experiments: z.array(ExperimentSchema),
  provenance: ProvenanceSchema,
  graph_uri: z.string(),
});

export const GraphSchema = z
  .strictObject({
    schema_version: SchemaVersionSchema,
    bundle_id: z.string(),
    nodes: z.array(NodeSchema),
    observations: z.array(ObservationSchema),
    masks: z.array(DetectorMaskSchema),
    circuits: z.array(CircuitSchema),
    memberships: z.array(MembershipSchema),
    edges: z.array(EdgeSchema),
  })
  .superRefine((graph, ctx) => {
    const seen = new Set<string>();
    graph.nodes.forEach((node, i) => {
      if (seen.has(node.id)) {
        ctx.addIssue({ code: "custom", message: `duplicate node id ${node.id}`, path: ["nodes", i] });
      }
      seen.add(node.id);
    });
    graph.observations.forEach((obs, i) => {
      if ((obs.score === null) !== (obs.status === "unavailable")) {
        ctx.addIssue({
          code: "custom",
          message: "score must be null if and only if status is 'unavailable'",
          path: ["observations", i, "score"],
        });
      }
    });
  });

export type EvidenceStatus = z.infer<typeof EvidenceStatusSchema>;
export type Manifest = z.infer<typeof ManifestSchema>;
export type Graph = z.infer<typeof GraphSchema>;
export type Experiment = z.infer<typeof ExperimentSchema>;
export type TokensFile = z.infer<typeof TokensFileSchema>;
export type Circuit = z.infer<typeof CircuitSchema>;
export type Membership = z.infer<typeof MembershipSchema>;
export type Observation = z.infer<typeof ObservationSchema>;
export type Edge = z.infer<typeof EdgeSchema>;
export type Endpoint = z.infer<typeof EndpointSchema>;
export type DetectorMask = z.infer<typeof DetectorMaskSchema>;

/** Checks that need both files at once. Returns problems; empty means OK. */
export function crossCheckBundle(manifest: Manifest, graph: Graph): string[] {
  const problems: string[] = [];
  if (manifest.bundle_id !== graph.bundle_id) {
    problems.push(`bundle_id mismatch: ${manifest.bundle_id} vs ${graph.bundle_id}`);
  }
  const expected = manifest.model.layers * manifest.model.heads_per_layer;
  if (graph.nodes.length !== expected) {
    problems.push(`expected ${expected} head nodes, found ${graph.nodes.length}`);
  }
  if (manifest.synthetic_fixture && !manifest.display_warning?.trim()) {
    problems.push("synthetic fixtures must carry a non-empty display_warning");
  }
  return problems;
}
