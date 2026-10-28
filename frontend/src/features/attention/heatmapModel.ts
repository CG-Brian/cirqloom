// Pure helpers behind the attention heatmap: mask lookup, token display, colours, geometry and
// cell readouts. No React, no canvas, so every rule here is unit-tested.
import type { DetectorMask, Experiment, Manifest } from "../../data/schema";
import type { HeadAttention } from "../../data/DataProvider";

/**
 * The cells `detectorId` averaged in `experimentId`, exactly as the Python pipeline exported
 * them. null when the bundle has no mask for that pair: never an empty or guessed mask.
 */
export function getDetectorMask(
  masks: DetectorMask[],
  experimentId: string,
  detectorId: string,
): DetectorMask | null {
  return masks.find((m) => m.experiment_id === experimentId && m.detector_id === detectorId) ?? null;
}

/** A measurement experiment with the same sequence length, to suggest from a control run. */
export function measurementFor(manifest: Manifest, experiment: Experiment): Experiment | null {
  return (
    manifest.experiments.find(
      (e) => e.kind === "measurement" && e.attention !== null && e.sequence_length === experiment.sequence_length,
    ) ?? null
  );
}

/** Makes whitespace visible: leading/trailing spaces as ␣, newlines as ↵, tabs as ⇥. */
export function displayToken(token: string): string {
  return token.replaceAll(" ", "␣").replaceAll("\n", "↵").replaceAll("\t", "⇥");
}

/** Mask cells as a Set of q * seq + k, for O(1) lookups while drawing. */
export function maskCellSet(cells: [number, number][], seq: number): Set<number> {
  return new Set(cells.map(([q, k]) => q * seq + k));
}

export type RGB = [number, number, number];

// Fixed 0–1 scale for every head: a head that barely attends anywhere looks faint, which is the
// honest picture. Rescaling per head would make weak heads look strong.
const RAMP: { at: number; rgb: RGB }[] = [
  { at: 0, rgb: [15, 23, 42] },
  { at: 0.5, rgb: [37, 117, 200] },
  { at: 1, rgb: [224, 242, 254] },
];

export function weightColor(weight: number): RGB {
  const w = Math.min(1, Math.max(0, weight));
  for (let i = 1; i < RAMP.length; i++) {
    const lo = RAMP[i - 1]!;
    const hi = RAMP[i]!;
    if (w <= hi.at) {
      const t = (w - lo.at) / (hi.at - lo.at);
      return [0, 1, 2].map((c) => Math.round(lo.rgb[c]! + t * (hi.rgb[c]! - lo.rgb[c]!))) as RGB;
    }
  }
  return RAMP[RAMP.length - 1]!.rgb;
}

/** Cells outside the shown mask are blended toward the background so the mask stands out. */
export const OUTSIDE_MASK_DIM = 0.35;
const BACKGROUND: RGB = [2, 6, 23];

export function dimmed([r, g, b]: RGB): RGB {
  const keep = OUTSIDE_MASK_DIM;
  return [
    Math.round(BACKGROUND[0] + keep * (r - BACKGROUND[0])),
    Math.round(BACKGROUND[1] + keep * (g - BACKGROUND[1])),
    Math.round(BACKGROUND[2] + keep * (b - BACKGROUND[2])),
  ];
}

/**
 * RGBA pixels at one pixel per cell, row = destination q, column = source k. Future cells
 * (k > q) are transparent so the hatch drawn underneath shows through.
 */
export function heatmapPixels(weights: number[][], mask: Set<number> | null): Uint8ClampedArray {
  const seq = weights.length;
  const pixels = new Uint8ClampedArray(seq * seq * 4);
  for (let q = 0; q < seq; q++) {
    for (let k = 0; k <= q; k++) {
      const base = weightColor(weights[q]![k]!);
      const [r, g, b] = mask && !mask.has(q * seq + k) ? dimmed(base) : base;
      const i = (q * seq + k) * 4;
      pixels[i] = r;
      pixels[i + 1] = g;
      pixels[i + 2] = b;
      pixels[i + 3] = 255;
    }
  }
  return pixels;
}

/** The cell under a point in a square drawing of `size` CSS pixels, or null outside it. */
export function cellFromPoint(x: number, y: number, size: number, seq: number): { q: number; k: number } | null {
  if (x < 0 || y < 0 || x >= size || y >= size) {
    return null;
  }
  return { q: Math.floor((y / size) * seq), k: Math.floor((x / size) * seq) };
}

/** Axis tick positions: 0, then a round step that gives about five ticks, and the last index. */
export function axisTicks(seq: number): number[] {
  const step = [1, 2, 5, 10, 20, 50, 100].find((s) => seq / s <= 6) ?? Math.ceil(seq / 5);
  const ticks: number[] = [];
  for (let t = 0; t < seq; t += step) {
    ticks.push(t);
  }
  if (ticks[ticks.length - 1] !== seq - 1 && seq - 1 - ticks[ticks.length - 1]! >= step / 2) {
    ticks.push(seq - 1);
  }
  return ticks;
}

export interface CellReadout {
  q: number;
  k: number;
  queryToken: string;
  sourceToken: string;
  /** null for a future cell: not attendable, which is not the same as weight 0. */
  weight: number | null;
  inMask: boolean;
}

export function readCell(attention: HeadAttention, mask: Set<number> | null, q: number, k: number): CellReadout {
  const seq = attention.tokens.length;
  return {
    q,
    k,
    queryToken: displayToken(attention.tokens[q] ?? ""),
    sourceToken: displayToken(attention.tokens[k] ?? ""),
    weight: k > q ? null : (attention.weights[q]?.[k] ?? null),
    inMask: mask?.has(q * seq + k) ?? false,
  };
}
