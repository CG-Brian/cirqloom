import { describe, expect, it } from "vitest";
import type { Experiment, Manifest } from "../data/schema";
import { getActiveExperiment, observationsForExperiment } from "../features/experiment/activeExperiment";
import { loadFixture } from "./fixtureView";

const { manifest: fixtureManifest } = loadFixture();

const exp = (id: string, kind: Experiment["kind"]): Experiment => ({
  id,
  prompt_suite: "t",
  kind,
  description: "",
  seeds: [0],
  sequence_length: 4,
  tokens_uri: "t.json",
  attention: null,
});

const manifest = (experiments: Experiment[]): Manifest => ({ ...fixtureManifest, experiments });

describe("getActiveExperiment", () => {
  it("is null when the bundle has no experiments", () => {
    expect(getActiveExperiment(fixtureManifest, null)).toBeNull();
  });

  it("defaults to the first measurement, not a control listed before it", () => {
    const m = manifest([exp("c", "control"), exp("m1", "measurement"), exp("m2", "measurement")]);
    expect(getActiveExperiment(m, null)?.id).toBe("m1");
  });

  it("uses the user's pick when it exists and ignores a stale one", () => {
    const m = manifest([exp("m1", "measurement"), exp("c", "control")]);
    expect(getActiveExperiment(m, "c")?.id).toBe("c");
    expect(getActiveExperiment(m, "gone")?.id).toBe("m1");
  });

  it("falls back to a control when there is no measurement", () => {
    expect(getActiveExperiment(manifest([exp("c", "control")]), null)?.id).toBe("c");
  });
});

describe("observationsForExperiment", () => {
  it("keeps one experiment's observations and nothing without an experiment", () => {
    const base = {
      node_id: "n",
      detector_id: "d",
      detector_version: "1.0.0",
      evidence_type: "heuristic_attention" as const,
      parameters: {},
      score: 0.5,
      score_std: null,
      per_example_scores: [0.5],
      status: "experimental" as const,
      sample_count: 1,
      method_reference: "x",
    };
    const all = [{ ...base, experiment_id: "a" }, { ...base, experiment_id: "b" }];
    expect(observationsForExperiment(all, exp("a", "measurement")).map((o) => o.experiment_id)).toEqual(["a"]);
    expect(observationsForExperiment(all, null)).toEqual([]);
  });
});
