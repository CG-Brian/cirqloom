// The inspector's attention section: decides which honest state to show (no experiments,
// control run, loading, failed load, heatmap) and hands the heatmap the mask the user picked.
import type { DataProvider } from "../data/DataProvider";
import type { DetectorMask, Experiment, Manifest } from "../data/schema";
import { getDetectorMask, measurementFor } from "../features/attention/heatmapModel";
import { useHeadAttention } from "../features/attention/useHeadAttention";
import { useExplorerStore } from "../features/selection/store";
import { AttentionHeatmap, type HeatmapMask } from "./AttentionHeatmap";
import { EmptyNote } from "./ui";

interface AttentionPanelProps {
  provider: DataProvider;
  manifest: Manifest;
  experiment: Experiment | null;
  nodeId: string;
  masks: DetectorMask[];
}

export function AttentionPanel({ provider, manifest, experiment, nodeId, masks }: AttentionPanelProps) {
  const maskDetectorId = useExplorerStore((s) => s.maskDetectorId);
  const setExperiment = useExplorerStore((s) => s.setExperiment);
  const stored = experiment !== null && experiment.attention !== null;
  const state = useHeadAttention(provider, stored ? experiment.id : null, nodeId);

  if (manifest.experiments.length === 0 || experiment === null) {
    return <EmptyNote>No attention data: this bundle contains no measured runs, so there is no heatmap to show.</EmptyNote>;
  }
  if (!stored || experiment.attention === null) {
    const suggestion = measurementFor(manifest, experiment);
    return (
      <EmptyNote>
        No attention matrix is stored for {experiment.id}; this run keeps scores only. Attention is
        stored for one prompt of each measurement run.
        {suggestion && (
          <>
            {" "}
            <button
              type="button"
              onClick={() => setExperiment(suggestion.id)}
              className="text-sky-300 underline-offset-2 hover:underline"
            >
              Show {suggestion.id}
            </button>
          </>
        )}
      </EmptyNote>
    );
  }
  if (state.status === "loading" || state.status === "idle") {
    return <p className="text-xs text-slate-400">Loading attention…</p>;
  }
  if (state.status === "error") {
    return (
      <div role="alert" className="rounded border border-red-400/30 px-2 py-1.5 text-xs text-red-200/90">
        Could not load attention for this head: {state.error.message}
      </div>
    );
  }

  const record = maskDetectorId === null ? null : getDetectorMask(masks, experiment.id, maskDetectorId);
  const mask: HeatmapMask | null = record && {
    label: `${record.detector_id} v${record.detector_version}`,
    cellsQk: record.cells_qk,
  };
  return (
    <>
      {maskDetectorId !== null && record === null && (
        <p className="mb-2 text-[11px] text-slate-400">
          Cells not recorded for <span className="font-mono">{maskDetectorId}</span> in this experiment; no mask is drawn.
        </p>
      )}
      <AttentionHeatmap
        attention={state.attention}
        mask={mask}
        seed={experiment.seeds[experiment.attention.prompt_index] ?? experiment.attention.prompt_index}
        promptCount={experiment.seeds.length}
      />
    </>
  );
}
