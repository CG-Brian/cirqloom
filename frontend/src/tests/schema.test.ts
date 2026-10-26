// @vitest-environment node
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { crossCheckBundle, GraphSchema, ManifestSchema } from "../data/schema";

const FIXTURE_DIR = new URL("../../../data/fixtures/gpt2-small-demo/", import.meta.url);

function loadJson(name: string): Record<string, unknown> {
  return JSON.parse(readFileSync(fileURLToPath(new URL(name, FIXTURE_DIR)), "utf8"));
}

const VALID_OBS = {
  node_id: "model:gpt2-small:head:L0H0",
  experiment_id: "exp-test",
  detector_id: "previous_token",
  detector_version: "0.0.0",
  evidence_type: "heuristic_attention",
  score: 0.5,
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
});
