// @vitest-environment node
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { crossCheckBundle, GraphSchema, ManifestSchema, TokensFileSchema } from "../data/schema";

const FIXTURE_DIR = new URL("../../../data/fixtures/gpt2-small-demo/", import.meta.url);
const MEASURED_DIR = new URL("../../../data/bundles/gpt2-small-repeated-v1/", import.meta.url);

function loadJson(name: string, dir = FIXTURE_DIR): Record<string, unknown> {
  return JSON.parse(readFileSync(fileURLToPath(new URL(name, dir)), "utf8"));
}

const VALID_OBS = {
  node_id: "model:gpt2-small:head:L0H0",
  experiment_id: "exp-test",
  detector_id: "previous_token",
  detector_version: "0.0.0",
  evidence_type: "heuristic_attention",
  parameters: { burn_in: 2 },
  score: 0.5,
  score_std: 0.1,
  per_example_scores: [0.4, 0.5, 0.6],
  status: "measured",
  sample_count: 3,
  method_reference: "test",
};

describe("bundle contract shared with Python", () => {
  it("accepts the Python-generated fixture", () => {
    const manifest = ManifestSchema.parse(loadJson("manifest.json"));
    const graph = GraphSchema.parse(loadJson("graph.json"));
    expect(crossCheckBundle(manifest, graph)).toEqual([]);
    expect(graph.nodes).toHaveLength(144);
    expect(manifest.synthetic_fixture).toBe(true);
  });

  it("rejects unknown keys", () => {
    const manifest = { ...loadJson("manifest.json"), colour: "red" };
    expect(ManifestSchema.safeParse(manifest).success).toBe(false);
  });

  it("accepts a real score with status measured", () => {
    const graph = { ...loadJson("graph.json"), observations: [VALID_OBS] };
    expect(GraphSchema.safeParse(graph).success).toBe(true);
  });

  it("rejects a fake zero for an unavailable score", () => {
    const fake = { ...VALID_OBS, score: 0, status: "unavailable" };
    const graph = { ...loadJson("graph.json"), observations: [fake] };
    expect(GraphSchema.safeParse(graph).success).toBe(false);
  });

  it("flags a missing head", () => {
    const manifest = ManifestSchema.parse(loadJson("manifest.json"));
    const graph = GraphSchema.parse(loadJson("graph.json"));
    const short = { ...graph, nodes: graph.nodes.slice(1) };
    expect(crossCheckBundle(manifest, short)).toContain("expected 144 head nodes, found 143");
  });

  it("accepts the Python-generated measured GPT-2 bundle", () => {
    const manifest = ManifestSchema.parse(loadJson("manifest.json", MEASURED_DIR));
    const graph = GraphSchema.parse(loadJson("graph.json", MEASURED_DIR));
    expect(crossCheckBundle(manifest, graph)).toEqual([]);
    expect(manifest.synthetic_fixture).toBe(false);
    expect(manifest.model.revision).toMatch(/^[0-9a-f]{40}$/);
    expect(graph.circuits).toEqual([]);
    for (const experiment of manifest.experiments) {
      const tokens = TokensFileSchema.parse(loadJson(experiment.tokens_uri, MEASURED_DIR));
      expect(tokens.experiment_id).toBe(experiment.id);
      expect(tokens.prompts.map((p) => p.seed)).toEqual(experiment.seeds);
    }
    // Every observation is labelled experimental: nothing in the bundle claims more.
    expect(new Set(graph.observations.map((o) => o.status))).toEqual(new Set(["experimental"]));
    // One mask per (experiment, detector) that has observations.
    const pairs = new Set(graph.observations.map((o) => `${o.experiment_id}|${o.detector_id}`));
    expect(new Set(graph.masks.map((m) => `${m.experiment_id}|${m.detector_id}`))).toEqual(pairs);
  });

  it("parses edges with head, role-group and special endpoints, and rejects malformed ones", () => {
    const edge = {
      id: "e1",
      circuit_id: "c",
      source: { kind: "head", node_id: "model:gpt2-small:head:L7H3" },
      target: { kind: "role_group", circuit_id: "c", role: "name_mover" },
      channel: "q",
      positions: { source: "END", target: "END" },
      evidence_kind: "published_experimental",
      evidence_status: "published",
      effect: null,
      magnitude: null,
      conditions: [],
      locator: "§3.2, Figure 4b",
      method_reference: null,
      description: "test",
    };
    const withEdge = (e: object) => ({ ...loadJson("graph.json"), edges: [e] });
    const parsed = GraphSchema.parse(withEdge(edge));
    const target = parsed.edges[0]!.target;
    expect(target.kind === "role_group" && target.role).toBe("name_mover");
    expect(GraphSchema.safeParse(withEdge({ ...edge, target: { kind: "special", id: "logits" } })).success).toBe(true);
    // role_group without its circuit, an unknown endpoint kind, an unknown special id
    expect(GraphSchema.safeParse(withEdge({ ...edge, target: { kind: "role_group", role: "x" } })).success).toBe(false);
    expect(GraphSchema.safeParse(withEdge({ ...edge, target: { kind: "mlp", node_id: "x" } })).success).toBe(false);
    expect(GraphSchema.safeParse(withEdge({ ...edge, target: { kind: "special", id: "residual" } })).success).toBe(false);
    // the M0 edge shape is gone
    expect(GraphSchema.safeParse(withEdge({ ...edge, source_node_id: "x" })).success).toBe(false);
  });

  it("rejects a mask with no cells or a malformed cell", () => {
    const mask = {
      experiment_id: "exp-test",
      detector_id: "previous_token",
      detector_version: "0.0.0",
      parameters: { burn_in: 2 },
      cells_qk: [[2, 1]],
    };
    const withMask = (m: object) => ({ ...loadJson("graph.json"), masks: [m] });
    expect(GraphSchema.safeParse(withMask(mask)).success).toBe(true);
    expect(GraphSchema.safeParse(withMask({ ...mask, cells_qk: [] })).success).toBe(false);
    expect(GraphSchema.safeParse(withMask({ ...mask, cells_qk: [[2, 1, 0]] })).success).toBe(false);
  });
});
