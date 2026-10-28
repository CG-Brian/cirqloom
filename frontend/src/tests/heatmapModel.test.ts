import { describe, expect, it } from "vitest";
import type { HeadAttention } from "../data/DataProvider";
import type { DetectorMask, Experiment, Manifest } from "../data/schema";
import {
  axisTicks,
  cellFromPoint,
  dimmed,
  displayToken,
  getDetectorMask,
  heatmapPixels,
  maskCellSet,
  measurementFor,
  readCell,
  weightColor,
} from "../features/attention/heatmapModel";

const mask = (experiment_id: string, detector_id: string): DetectorMask => ({
  experiment_id,
  detector_id,
  detector_version: "1.0.0",
  parameters: {},
  cells_qk: [[1, 0]],
});

describe("getDetectorMask", () => {
  const masks = [mask("a", "induction"), mask("b", "induction"), mask("a", "previous")];

  it("matches experiment and detector together", () => {
    expect(getDetectorMask(masks, "b", "induction")).toBe(masks[1]);
    expect(getDetectorMask(masks, "a", "previous")).toBe(masks[2]);
  });

  it("returns null, never an empty mask, when the pair is not recorded", () => {
    expect(getDetectorMask(masks, "b", "previous")).toBeNull();
    expect(getDetectorMask([], "a", "induction")).toBeNull();
  });
});

describe("measurementFor", () => {
  const exp = (id: string, kind: Experiment["kind"], seq: number): Experiment => ({
    id,
    prompt_suite: "t",
    kind,
    description: "",
    seeds: [0],
    sequence_length: seq,
    tokens_uri: "t.json",
    attention: kind === "measurement" ? { uri: "a", dtype: "float16", shape: [1, 1, seq, seq], prompt_index: 0 } : null,
  });
  const experiments = [exp("m21", "measurement", 21), exp("c21", "control", 21), exp("c51", "control", 51)];
  const manifest = { experiments } as unknown as Manifest;

  it("suggests the measurement run of the same length, or nothing", () => {
    expect(measurementFor(manifest, experiments[1]!)?.id).toBe("m21");
    expect(measurementFor(manifest, experiments[2]!)).toBeNull();
  });
});

describe("token display", () => {
  it("makes leading spaces and control whitespace visible", () => {
    expect(displayToken(" Mary")).toBe("␣Mary");
    expect(displayToken("\n\n")).toBe("↵↵");
    expect(displayToken("\t")).toBe("⇥");
    expect(displayToken("<|endoftext|>")).toBe("<|endoftext|>");
  });
});

describe("colours", () => {
  it("uses one fixed 0–1 scale that gets brighter with weight and clamps", () => {
    const sum = (c: number[]) => c.reduce((a, b) => a + b, 0);
    expect(sum(weightColor(0.2))).toBeLessThan(sum(weightColor(0.6)));
    expect(weightColor(-1)).toEqual(weightColor(0));
    expect(weightColor(2)).toEqual(weightColor(1));
  });

  it("dims a colour toward the background", () => {
    const [r] = dimmed(weightColor(1));
    expect(r).toBeLessThan(weightColor(1)[0]);
  });
});

describe("heatmapPixels", () => {
  const weights = [
    [1, 0, 0],
    [0.5, 0.5, 0],
    [0, 0, 1],
  ];

  it("leaves future cells transparent and paints the rest opaque", () => {
    const px = heatmapPixels(weights, null);
    const alpha = (q: number, k: number) => px[(q * 3 + k) * 4 + 3];
    expect(alpha(0, 1)).toBe(0);
    expect(alpha(1, 2)).toBe(0);
    expect(alpha(2, 0)).toBe(255);
    expect(alpha(1, 1)).toBe(255);
  });

  it("keeps mask cells at their true colour and dims the others", () => {
    const set = maskCellSet([[1, 0]], 3);
    const px = heatmapPixels(weights, set);
    const red = (q: number, k: number) => px[(q * 3 + k) * 4];
    expect(red(1, 0)).toBe(weightColor(0.5)[0]); // in mask
    expect(red(1, 1)).toBe(dimmed(weightColor(0.5))[0]); // same weight, outside the mask
  });
});

describe("geometry", () => {
  it("maps a point to row = destination q, column = source k", () => {
    expect(cellFromPoint(0, 0, 100, 10)).toEqual({ q: 0, k: 0 });
    expect(cellFromPoint(35, 92, 100, 10)).toEqual({ q: 9, k: 3 });
    expect(cellFromPoint(100, 5, 100, 10)).toBeNull();
    expect(cellFromPoint(-1, 5, 100, 10)).toBeNull();
  });

  it("puts about five ticks on an axis, starting at 0 and ending near the last index", () => {
    expect(axisTicks(21)).toEqual([0, 5, 10, 15, 20]);
    expect(axisTicks(101)).toEqual([0, 20, 40, 60, 80, 100]);
    expect(axisTicks(51)).toEqual([0, 10, 20, 30, 40, 50]);
    expect(axisTicks(4)).toEqual([0, 1, 2, 3]);
  });
});

describe("readCell", () => {
  const attention: HeadAttention = {
    experimentId: "e",
    nodeId: "n",
    promptId: "p",
    tokens: ["<|endoftext|>", " a", " b"],
    weights: [
      [1, 0, 0],
      [0.25, 0.75, 0],
      [0.1, 0.2, 0.7],
    ],
  };

  it("reports both tokens, the exact weight and mask membership", () => {
    const set = maskCellSet([[2, 1]], 3);
    expect(readCell(attention, set, 2, 1)).toEqual({
      q: 2,
      k: 1,
      queryToken: "␣b",
      sourceToken: "␣a",
      weight: 0.2,
      inMask: true,
    });
    expect(readCell(attention, set, 1, 0).inMask).toBe(false);
  });

  it("reports a future cell as not attendable (null), not as zero", () => {
    expect(readCell(attention, null, 0, 2).weight).toBeNull();
  });
});
