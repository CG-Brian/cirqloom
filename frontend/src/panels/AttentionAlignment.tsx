// Relation inspector section: observed attention of the relation's heads at named positions on
// one prompt of the selected experiment (measured here, experimental). Kept visibly separate from
// the published claim above it, and never presented as information passing between heads.
import { useEffect, useMemo, useState } from "react";
import type { DataProvider, HeadAttention } from "../data/DataProvider";
import type { Experiment, Manifest } from "../data/schema";
import { alignmentPlan, alignmentRows, type AlignmentPlan } from "../features/attention/alignment";
import { weightColor } from "../features/attention/heatmapModel";
import { edgeGroupHeads, roleLabel, type VisualEdgeGroup } from "../features/circuits/edgeGroups";
import type { GraphView } from "../features/experiment/graphView";
import { useExplorerStore } from "../features/selection/store";
import { EmptyNote } from "./ui";

type LoadState =
  | { status: "loading" }
  | { status: "ready"; attentions: Map<string, HeadAttention> }
  | { status: "error"; error: Error };

/** Loads every head of the plan for one experiment; drops results for a stale request. */
function useAttentions(provider: DataProvider, experimentId: string | null, plan: AlignmentPlan): LoadState | null {
  const ids = useMemo(() => [...plan.senders, ...plan.receivers.filter((id) => !plan.senders.includes(id))], [plan]);
  const key = experimentId === null || ids.length === 0 ? null : `${experimentId}|${ids.join(",")}`;
  const [settled, setSettled] = useState<{ key: string; state: LoadState } | null>(null);
  useEffect(() => {
    if (key === null || experimentId === null) {
      return;
    }
    let ignore = false;
    Promise.all(ids.map((id) => provider.loadHeadAttention(experimentId, id)))
      .then((list) => {
        if (!ignore) {
          setSettled({ key, state: { status: "ready", attentions: new Map(list.map((a) => [a.nodeId, a])) } });
        }
      })
      .catch((err: unknown) => {
        if (!ignore) {
          setSettled({ key, state: { status: "error", error: err instanceof Error ? err : new Error(String(err)) } });
        }
      });
    return () => {
      ignore = true;
    };
  }, [provider, experimentId, ids, key]);
  if (key === null) {
    return null;
  }
  return settled?.key === key ? settled.state : { status: "loading" };
}

/** Measurement experiments whose stored prompt has all the given named positions. */
function useExperimentsWithPositions(provider: DataProvider, manifest: Manifest, labels: string[]): Experiment[] {
  const [found, setFound] = useState<Experiment[]>([]);
  const wanted = labels.join(",");
  useEffect(() => {
    let ignore = false;
    const candidates = manifest.experiments.filter((e) => e.attention !== null);
    Promise.all(candidates.map((e) => provider.loadExperimentPositions(e.id).catch(() => null)))
      .then((all) => {
        if (!ignore) {
          setFound(candidates.filter((_, i) => wanted.split(",").every((l) => all[i]?.[l] !== undefined)));
        }
      });
    return () => {
      ignore = true;
    };
  }, [provider, manifest, wanted]);
  return found;
}

interface AttentionAlignmentProps {
  provider: DataProvider;
  view: GraphView;
  manifest: Manifest;
  experiment: Experiment | null;
  group: VisualEdgeGroup;
}

export function AttentionAlignment({ provider, view, manifest, experiment, group }: AttentionAlignmentProps) {
  const setExperiment = useExplorerStore((s) => s.setExperiment);
  const select = useExplorerStore((s) => s.select);
  const plan = useMemo(() => alignmentPlan(group, edgeGroupHeads(view, group)), [view, group]);
  const labels = [plan.senderQuery, plan.receiverQuery].filter((l): l is string => l !== null);
  const stored = experiment?.attention ? experiment : null;
  const state = useAttentions(provider, stored?.id ?? null, plan);
  const suggestions = useExperimentsWithPositions(provider, manifest, labels);

  if (labels.length === 0) {
    return <EmptyNote>This relation names no token positions, so there is no row to align.</EmptyNote>;
  }
  const switcher = suggestions.length > 0 && (
    <span className="mt-1 flex flex-wrap gap-1">
      {suggestions.map((e) => (
        <button
          key={e.id}
          type="button"
          onClick={() => setExperiment(e.id)}
          className="rounded border border-slate-700 px-1.5 py-0.5 font-mono text-[10px] text-sky-300 hover:border-slate-500"
        >
          {e.id}
        </button>
      ))}
    </span>
  );
  if (stored === null) {
    return (
      <EmptyNote>
        The selected experiment stores no attention. Show one whose prompts have positions {labels.join(", ")}:
        {switcher}
      </EmptyNote>
    );
  }
  if (state === null || state.status === "loading") {
    return <p className="text-xs text-slate-400">Loading attention…</p>;
  }
  if (state.status === "error") {
    return (
      <div role="alert" className="rounded border border-red-400/30 px-2 py-1.5 text-xs text-red-200/90">
        Could not load attention: {state.error.message}
      </div>
    );
  }
  const rows = alignmentRows(plan, state.attentions);
  if (rows.length === 0) {
    return (
      <EmptyNote>
        Prompts of {stored.id} have no positions {labels.join(", ")}. Show an experiment that does:
        {switcher}
      </EmptyNote>
    );
  }
  const first = state.attentions.values().next().value as HeadAttention;
  const seed = stored.seeds[stored.attention?.prompt_index ?? 0];
  const columns = rows[0]!.cells.map((c) => c.label);
  const roleOf = (nodeId: string) =>
    (view.membershipsByNode.get(nodeId) ?? [])
      .filter((m) => m.circuit_id === group.circuitId)
      .map((m) => roleLabel(view, group.circuitId, m.role))
      .join(", ");

  return (
    <div data-testid="attention-alignment">
      {first.text && (
        <p className="mb-1.5 text-[11px] leading-snug text-slate-300">
          <span className="text-slate-500">Prompt (seed {seed}, {stored.id}): </span>“{first.text}”
        </p>
      )}
      <table className="w-full border-collapse text-[11px]">
        <caption className="sr-only">
          Attention from each head&apos;s query position to the named positions, on one prompt
        </caption>
        <thead>
          <tr className="text-slate-500">
            <th scope="col" className="py-0.5 pr-1 text-left font-normal">head @ query</th>
            {columns.map((label) => (
              <th key={label} scope="col" className="px-0.5 py-0.5 text-center font-mono font-normal text-sky-300">
                {label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={`${r.side}:${r.nodeId}`} className="border-t border-slate-800/80">
              <th scope="row" className="py-0.5 pr-1 text-left font-normal">
                <button
                  type="button"
                  onClick={() => select(r.nodeId)}
                  className="font-mono text-slate-200 hover:underline"
                  title={roleOf(r.nodeId)}
                >
                  {view.nodeById.get(r.nodeId)?.label ?? r.nodeId}
                </button>
                <span className="text-slate-500"> {r.side} @{r.queryLabel}</span>
              </th>
              {r.cells.map((c) => (
                <td
                  key={c.label}
                  className="px-0.5 py-0.5 text-center font-mono tabular-nums"
                  style={
                    c.weight === null
                      ? undefined
                      : { backgroundColor: `rgb(${weightColor(c.weight).join(",")})`, color: c.weight > 0.55 ? "#0f172a" : "#e2e8f0" }
                  }
                  aria-label={c.weight === null ? `${c.label}: after the query, not attendable` : `${c.label}: ${c.weight.toPrecision(3)}`}
                >
                  {c.weight === null ? <span className="text-slate-600">—</span> : c.weight.toFixed(2)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-2 text-[10px] leading-snug text-slate-500">
        Measured here on one prompt (float16, same 0–1 colour scale as the heatmap); experimental.
        Each row is one head&apos;s attention from the query position the relation names for its side.
        Rows are separate observations: lining them up does not show information passing from the
        senders to the receivers, and attention alone does not verify a head&apos;s published role.
      </p>
    </div>
  );
}
