// Token-to-token attention inside ONE head, for one example prompt. Row = destination (query)
// position q, column = source (key) position k. Draws only what it is given: the weights, and
// optionally one detector mask exported by the Python pipeline. It never computes a score.
import { useEffect, useMemo, useRef, useState, type KeyboardEvent, type PointerEvent } from "react";
import type { HeadAttention } from "../data/DataProvider";
import {
  axisTicks,
  cellFromPoint,
  heatmapPixels,
  maskCellSet,
  readCell,
  weightColor,
} from "../features/attention/heatmapModel";

export interface HeatmapMask {
  /** e.g. "induction_attention v1.0.0" */
  label: string;
  /** [q, k] = [destination, source] */
  cellsQk: [number, number][];
}

interface AttentionHeatmapProps {
  attention: HeadAttention;
  mask: HeatmapMask | null;
  /** Seed of the drawn prompt and how many prompts the scores average. */
  seed: number;
  promptCount: number;
}

const GUTTER = 22; // px for the destination-axis tick labels
const FALLBACK_SIZE = 300;

function useWidth(ref: React.RefObject<HTMLElement | null>): number {
  const [width, setWidth] = useState(FALLBACK_SIZE + GUTTER);
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof ResizeObserver === "undefined") {
      return;
    }
    const observer = new ResizeObserver(([entry]) => {
      if (entry && entry.contentRect.width > 0) {
        setWidth(entry.contentRect.width);
      }
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, [ref]);
  return width;
}

/** A small diagonal-line tile: future cells are not attendable, which is not weight 0. */
function hatchPattern(ctx: CanvasRenderingContext2D, dpr: number): CanvasPattern | null {
  const tile = document.createElement("canvas");
  const n = Math.max(4, Math.round(5 * dpr));
  tile.width = n;
  tile.height = n;
  const t = tile.getContext("2d");
  if (!t) {
    return null;
  }
  t.strokeStyle = "rgba(100, 116, 139, 0.45)";
  t.lineWidth = Math.max(1, dpr * 0.75);
  t.beginPath();
  t.moveTo(0, n);
  t.lineTo(n, 0);
  t.stroke();
  return ctx.createPattern(tile, "repeat");
}

export function AttentionHeatmap({ attention, mask, seed, promptCount }: AttentionHeatmapProps) {
  const seq = attention.tokens.length;
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const width = useWidth(wrapRef);
  const size = Math.max(120, Math.floor(width - GUTTER));
  const [cursor, setCursor] = useState<{ q: number; k: number } | null>(null);

  const maskSet = useMemo(() => (mask ? maskCellSet(mask.cellsQk, seq) : null), [mask, seq]);
  const pixels = useMemo(() => heatmapPixels(attention.weights, maskSet), [attention, maskSet]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) {
      return; // no 2D canvas (e.g. jsdom): the text readout still works
    }
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(size * dpr);
    canvas.height = Math.round(size * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size, size);
    const cell = size / seq;

    // Hatch the upper triangle; the opaque k <= q cells drawn next cover all but k > q.
    const hatch = hatchPattern(ctx, dpr);
    if (hatch) {
      ctx.save();
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.fillStyle = hatch;
      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.lineTo(size * dpr, 0);
      ctx.lineTo(size * dpr, size * dpr);
      ctx.closePath();
      ctx.fill();
      ctx.restore();
    }

    // One pixel per cell, scaled up without smoothing so cell edges stay crisp.
    const image = document.createElement("canvas");
    image.width = seq;
    image.height = seq;
    const imageCtx = image.getContext("2d");
    if (imageCtx) {
      imageCtx.putImageData(new ImageData(new Uint8ClampedArray(pixels), seq, seq), 0, 0);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(image, 0, 0, size, size);
    }

    // Mask cells get an outline when cells are big enough to show one; otherwise the dimming
    // of everything outside the mask carries it.
    if (mask && cell >= 6) {
      ctx.strokeStyle = "rgba(251, 191, 36, 0.9)";
      ctx.lineWidth = 1;
      for (const [q, k] of mask.cellsQk) {
        ctx.strokeRect(k * cell + 0.5, q * cell + 0.5, cell - 1, cell - 1);
      }
    }

    if (cursor) {
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1.5;
      ctx.strokeRect(cursor.k * cell, cursor.q * cell, cell, cell);
    }
  }, [pixels, mask, size, seq, cursor]);

  const onPointerMove = (e: PointerEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    setCursor(cellFromPoint(e.clientX - rect.left, e.clientY - rect.top, rect.width, seq));
  };

  const onKeyDown = (e: KeyboardEvent<HTMLCanvasElement>) => {
    const moves: Record<string, [number, number]> = {
      ArrowUp: [-1, 0],
      ArrowDown: [1, 0],
      ArrowLeft: [0, -1],
      ArrowRight: [0, 1],
    };
    const move = moves[e.key];
    if (!move) {
      return;
    }
    e.preventDefault();
    e.stopPropagation();
    setCursor((c) => {
      const from = c ?? { q: 0, k: 0 };
      const clamp = (v: number) => Math.min(seq - 1, Math.max(0, v));
      return { q: clamp(from.q + move[0]), k: clamp(from.k + move[1]) };
    });
  };

  const readout = cursor ? readCell(attention, maskSet, cursor.q, cursor.k) : null;
  const ticks = axisTicks(seq);
  const pct = (i: number) => `${((i + 0.5) / seq) * 100}%`;
  const maskSummary = mask ? `; outlined: ${mask.cellsQk.length} cells averaged by ${mask.label}` : "";

  return (
    <div ref={wrapRef} className="w-full" data-testid="attention-heatmap">
      <div className="mb-1 flex justify-between text-[10px] text-slate-500" style={{ paddingLeft: GUTTER }}>
        <span>source position k →</span>
      </div>
      <div className="flex">
        <div className="relative shrink-0" style={{ width: GUTTER, height: size }} aria-hidden="true">
          {ticks.map((t) => (
            <span
              key={t}
              className="absolute right-1 -translate-y-1/2 font-mono text-[9px] text-slate-500 tabular-nums"
              style={{ top: pct(t) }}
            >
              {t}
            </span>
          ))}
        </div>
        <div className="relative" style={{ width: size }}>
          <div className="relative h-3" aria-hidden="true">
            {ticks.map((t) => (
              <span
                key={t}
                className="absolute top-0 -translate-x-1/2 font-mono text-[9px] leading-none text-slate-500 tabular-nums"
                style={{ left: pct(t) }}
              >
                {t}
              </span>
            ))}
          </div>
          <canvas
            ref={canvasRef}
            role="img"
            tabIndex={0}
            aria-label={`Attention of this head on prompt seed ${seed}: ${seq} by ${seq}, rows are destination positions, columns are source positions${maskSummary}. Use arrow keys to read cells.`}
            style={{ width: size, height: size }}
            className="block cursor-crosshair rounded-sm focus-visible:outline focus-visible:outline-2 focus-visible:outline-sky-400"
            onPointerMove={onPointerMove}
            onPointerLeave={() => setCursor(null)}
            onKeyDown={onKeyDown}
          />
        </div>
      </div>
      <div className="mt-0.5 text-[10px] text-slate-500" style={{ paddingLeft: GUTTER }}>
        ↑ rows: destination (query) position q
      </div>

      <div
        data-testid="heatmap-readout"
        className="mt-2 min-h-[2.75rem] rounded bg-slate-900 px-2 py-1.5 font-mono text-[11px] leading-snug text-slate-300"
      >
        {readout ? (
          <>
            <div>
              q {readout.q} <span className="text-slate-100">“{readout.queryToken}”</span> ← k {readout.k}{" "}
              <span className="text-slate-100">“{readout.sourceToken}”</span>
            </div>
            <div>
              {readout.weight === null ? (
                <span className="text-slate-400">future position: not attendable</span>
              ) : (
                <>
                  weight <span className="text-slate-50">{readout.weight.toPrecision(3)}</span>
                </>
              )}
              {readout.inMask && <span className="ml-2 text-amber-300">in mask</span>}
            </div>
          </>
        ) : (
          <span className="font-sans text-slate-500">Hover a cell, or focus the map and use arrow keys.</span>
        )}
      </div>

      <HeatmapLegend mask={mask} />

      <p className="mt-2 text-[10px] leading-snug text-slate-500">
        One example prompt (seed {seed}; 1 of {promptCount}). Detector scores average all {promptCount}{" "}
        prompts, so the cells of one prompt need not average to a score.
        Weights are float16 (about 3 significant digits). Each cell is attention between two token
        positions inside this head, not a connection between heads.
      </p>
    </div>
  );
}

function HeatmapLegend({ mask }: { mask: HeatmapMask | null }) {
  const stops = [0, 0.25, 0.5, 0.75, 1].map((w) => `rgb(${weightColor(w).join(",")}) ${w * 100}%`);
  return (
    <div className="mt-2 space-y-1 text-[10px] text-slate-400">
      <div className="flex items-center gap-2">
        <span className="font-mono tabular-nums">0</span>
        <span
          className="h-2 flex-1 rounded-sm"
          style={{ background: `linear-gradient(to right, ${stops.join(", ")})` }}
          aria-hidden="true"
        />
        <span className="font-mono tabular-nums">1</span>
        <span className="text-slate-500">weight, same scale for every head</span>
      </div>
      <div className="flex items-center gap-2">
        <span
          className="inline-block h-2.5 w-2.5 rounded-sm border border-slate-700"
          style={{
            backgroundImage:
              "repeating-linear-gradient(135deg, rgba(100,116,139,0.6) 0 1px, transparent 1px 4px)",
          }}
          aria-hidden="true"
        />
        <span>future positions (k &gt; q): not attendable, not zero</span>
      </div>
      <div className="flex items-center gap-2">
        <span className="inline-block h-2.5 w-2.5 rounded-sm border border-amber-400/90" aria-hidden="true" />
        {mask ? (
          <span>
            {mask.cellsQk.length} cells averaged by <span className="font-mono text-slate-300">{mask.label}</span>;
            other cells are dimmed
          </span>
        ) : (
          <span className="text-slate-500">No mask shown. Use “Show cells” on a detector above.</span>
        )}
      </div>
    </div>
  );
}
