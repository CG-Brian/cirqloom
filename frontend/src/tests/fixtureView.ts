// Test helper: the real Python-generated fixture, parsed and indexed exactly as the app does.
import { readFileSync } from "node:fs";
import path from "node:path";
import { GraphSchema, ManifestSchema } from "../data/schema";
import { buildGraphView } from "../features/experiment/graphView";

// Vitest runs from frontend/. (Under jsdom, import.meta.url is not a file:// URL.)
const FIXTURE_DIR = path.resolve(process.cwd(), "../data/fixtures/gpt2-small-demo");

function readJson(name: string): unknown {
  return JSON.parse(readFileSync(path.join(FIXTURE_DIR, name), "utf8"));
}

export function loadFixture() {
  const manifest = ManifestSchema.parse(readJson("manifest.json"));
  const graph = GraphSchema.parse(readJson("graph.json"));
  return { manifest, graph, view: buildGraphView(manifest, graph) };
}
