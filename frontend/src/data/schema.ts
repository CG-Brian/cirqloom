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

const SchemaVersionSchema = z.literal("1.0.0");

export const ModelInfoSchema = z.strictObject({
  id: z.string(),
  hf_id: z.string(),
  revision: z.string().nullable(),
  layers: z.int().positive(),
  heads_per_layer: z.int().positive(),
  tokenizer_id: z.string(),
});

export const ExperimentSchema = z.strictObject({
  id: z.string(),
  prompt_suite: z.string(),
  seed: z.int(),
  sequence_length: z.int().positive(),
  tokens_uri: z.string(),
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
  score: z.number().nullable(),
  status: EvidenceStatusSchema,
  sample_count: z.int().nonnegative(),
  method_reference: z.string(),
});

export const RoleDefSchema = z.strictObject({
  id: z.string(),
  label: z.string(),
  description: z.string(),
});

export const CircuitSchema = z.strictObject({
  id: z.string(),
  name: z.string(),
  origin: z.enum(["synthetic_fixture", "published"]),
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
});

export const EdgeSchema = z.strictObject({
  id: z.string(),
  source_node_id: z.string(),
  target_node_id: z.string(),
  edge_type: z.enum(["structural", "schematic_published", "experimentally_supported_in_paper"]),
  evidence_status: EvidenceStatusSchema,
  citation: z.string().nullable(),
  description: z.string(),
});

export const ProvenanceSchema = z.strictObject({
  generator: z.string(),
  generator_version: z.string(),
  notes: z.string(),
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
export type Circuit = z.infer<typeof CircuitSchema>;
export type Membership = z.infer<typeof MembershipSchema>;
export type Observation = z.infer<typeof ObservationSchema>;

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
