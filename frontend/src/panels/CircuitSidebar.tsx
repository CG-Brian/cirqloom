// Left panel: circuit overlays, the role legend for the active one, detectors with their top
// heads on the selected experiment, and the keyboard head list. Reads and writes the shared store; holds no selection state of its own.
import type { Experiment } from "../data/schema";
import { circuitFocusBounds, type CircuitLayout } from "../features/circuits/circuitLayout";
import { edgeGroupLabel, EVIDENCE_KIND_LABEL, getEdgeDisplayStyle, type VisualEdgeGroup } from "../features/circuits/edgeGroups";
import type { GraphView } from "../features/experiment/graphView";
import { getRoleLegend, getTopHeads, type CircuitOverlay } from "../features/overlays/selectors";
import { useExplorerStore } from "../features/selection/store";
import { HeadList } from "./HeadList";
import { CollapseButton, EmptyNote, EvidenceBadge, OriginBadge, PanelSection, RoleMarker } from "./ui";

interface CircuitSidebarProps {
  view: GraphView;
  /** The experiment the inspector shows; detector top heads are ranked on it. */
  experiment: Experiment | null;
  overlay: CircuitOverlay | null;
  edgeGroups: VisualEdgeGroup[];
  circuitLayout: CircuitLayout | null;
  collapsed: boolean;
  onToggleCollapsed: () => void;
}

const LINE_DASH = { solid: undefined, dashed: "5 3", dotted: "1.5 3" } as const;

/** A detector's highest-scoring heads on one experiment, as buttons that select the head. */
function TopHeads({ view, experimentId, detectorId }: { view: GraphView; experimentId: string; detectorId: string }) {
  const select = useExplorerStore((s) => s.select);
  const top = getTopHeads(view, experimentId, detectorId);
  if (top.length === 0) {
    return <div className="mt-0.5 text-[10px] text-slate-500 italic">Not run on this experiment.</div>;
  }
  return (
    <div className="mt-1 flex flex-wrap items-center gap-1" aria-label={`Top heads for ${detectorId}`} role="group">
      <span className="text-[10px] text-slate-500">Top:</span>
      {top.map((t) => (
        <button
          key={t.nodeId}
          type="button"
          onClick={() => select(t.nodeId)}
          aria-label={`Inspect ${t.label} (${detectorId} ${t.score.toPrecision(3)})`}
          className="rounded border border-slate-700 px-1.5 py-px font-mono text-[10px] text-slate-300 tabular-nums hover:border-slate-500 hover:text-slate-100 focus-visible:outline focus-visible:outline-2 focus-visible:outline-sky-400"
        >
          {t.label} {t.score.toFixed(2)}
        </button>
      ))}
    </div>
  );
}

/** Keyboard-accessible list of the shown circuit's relations, parallel to the 3D ribbons. */
function RelationList({ view, groups }: { view: GraphView; groups: VisualEdgeGroup[] }) {
  const selectedEdgeGroupId = useExplorerStore((s) => s.selectedEdgeGroupId);
  const selectEdgeGroup = useExplorerStore((s) => s.selectEdgeGroup);
  const hoverEdgeGroup = useExplorerStore((s) => s.hoverEdgeGroup);
  const unhoverEdgeGroup = useExplorerStore((s) => s.unhoverEdgeGroup);
  const records = groups.reduce((n, g) => n + g.edges.length, 0);
  return (
    <div className="mt-3">
      <h4 className="mb-1.5 text-[10px] font-medium text-slate-500 uppercase">
        Relations ({groups.length} · {records} evidence records)
      </h4>
      <ul className="space-y-1" aria-label="Circuit relations">
        {groups.map((g) => {
          const selected = g.id === selectedEdgeGroupId;
          const style = getEdgeDisplayStyle(g, "idle");
          return (
            <li key={g.id}>
              <button
                type="button"
                aria-pressed={selected}
                onClick={() => selectEdgeGroup(selected ? null : g.id)}
                onMouseEnter={() => hoverEdgeGroup(g.id)}
                onMouseLeave={() => unhoverEdgeGroup(g.id)}
                onFocus={() => hoverEdgeGroup(g.id)}
                onBlur={() => unhoverEdgeGroup(g.id)}
                className={[
                  "flex w-full items-start gap-2 rounded border px-2 py-1 text-left text-[11px] transition-colors",
                  "focus-visible:outline focus-visible:outline-2 focus-visible:outline-sky-400",
                  selected ? "border-slate-300/60 bg-slate-800 text-slate-100" : "border-slate-800 text-slate-300 hover:border-slate-700",
                ].join(" ")}
              >
                <svg width="18" height="8" viewBox="0 0 18 8" className="mt-1 shrink-0" aria-hidden="true">
                  <line x1="1" y1="4" x2="17" y2="4" stroke="#cbd5e1" strokeWidth="1.5" strokeDasharray={LINE_DASH[style.pattern]} />
                </svg>
                <span className="flex-1">
                  {edgeGroupLabel(view, g)}
                  <span className="block text-[10px] text-slate-500">
                    {g.edges.length} record{g.edges.length === 1 ? "" : "s"} · {EVIDENCE_KIND_LABEL[g.evidenceKind]}
                    {g.conditions.length > 0 ? " · conditional" : ""}
                  </span>
                </span>
              </button>
            </li>
          );
        })}
      </ul>
      <p className="mt-2 text-[10px] leading-snug text-slate-500">
        Solid: published experiment. Dotted: holds only under a stated condition. Ribbons join
        group anchors; they are not wires between heads.
      </p>
    </div>
  );
}

export function CircuitSidebar({ view, experiment, overlay, edgeGroups, circuitLayout, collapsed, onToggleCollapsed }: CircuitSidebarProps) {
  const activeCircuitId = useExplorerStore((s) => s.activeCircuitId);
  const focusCamera = useExplorerStore((s) => s.focusCamera);
  const toggleCircuit = useExplorerStore((s) => s.toggleCircuit);
  // Detectors scored on the shown experiment; the rest are summarised in one line.
  const shownDetectors = experiment
    ? view.detectors.filter((d) => getTopHeads(view, experiment.id, d.id, 1).length > 0)
    : view.detectors;
  const hiddenDetectors = view.detectors.length - shownDetectors.length;

  if (collapsed) {
    return (
      <aside
        aria-label="Circuits and heads"
        className="flex shrink-0 items-start border-slate-800 bg-slate-950/70 p-1 max-lg:border-b lg:w-9 lg:flex-col lg:border-r"
      >
        <CollapseButton label="circuit sidebar" expanded={false} onClick={onToggleCollapsed} direction="left" />
        <span className="px-1 text-[11px] text-slate-500 lg:hidden">Circuits &amp; heads</span>
      </aside>
    );
  }

  return (
    <aside
      aria-label="Circuits and heads"
      className="flex shrink-0 flex-col border-slate-800 bg-slate-950/70 max-lg:border-b lg:w-64 lg:border-r"
    >
      <div className="flex items-center justify-between border-b border-slate-800/80 px-3 py-2">
        <h2 className="text-xs font-semibold text-slate-200">Circuits &amp; heads</h2>
        <CollapseButton label="circuit sidebar" expanded onClick={onToggleCollapsed} direction="left" />
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto">
        <PanelSection title="Circuit overlays">
          {view.circuits.length === 0 ? (
            <EmptyNote>No circuits in this bundle.</EmptyNote>
          ) : (
            <ul className="space-y-1.5">
              {view.circuits.map((circuit) => {
                const active = circuit.id === activeCircuitId;
                const members = view.nodes.filter((n) =>
                  view.membershipsByNode.get(n.id)?.some((m) => m.circuit_id === circuit.id),
                ).length;
                return (
                  <li key={circuit.id}>
                    <button
                      type="button"
                      aria-pressed={active}
                      onClick={() => toggleCircuit(circuit.id)}
                      className={[
                        "w-full rounded border px-2.5 py-2 text-left transition-colors",
                        "focus-visible:outline focus-visible:outline-2 focus-visible:outline-sky-400",
                        active
                          ? "border-indigo-300/50 bg-indigo-300/10"
                          : "border-slate-800 hover:border-slate-700 hover:bg-slate-900",
                      ].join(" ")}
                    >
                      <span className="flex items-center justify-between gap-2">
                        <span className="text-sm text-slate-100">{circuit.name}</span>
                        <span className="text-[10px] text-slate-500 tabular-nums">{members} heads</span>
                      </span>
                      <span className="mt-1 flex items-center gap-2">
                        <OriginBadge origin={circuit.origin} />
                        <span className="text-[10px] text-slate-500">{active ? "Shown" : "Hidden"}</span>
                      </span>
                      {circuit.description && (
                        <span className="mt-1.5 block text-[11px] leading-snug text-slate-400">
                          {circuit.description}
                        </span>
                      )}
                    </button>
                  </li>
                );
              })}
            </ul>
          )}

          {overlay && (
            <div className="mt-3">
              {circuitLayout && (
                <button
                  type="button"
                  onClick={() => focusCamera(circuitFocusBounds(view, circuitLayout))}
                  className="mb-2 w-full rounded border border-slate-700 px-2 py-1 text-[11px] text-slate-200 hover:border-slate-500 focus-visible:outline focus-visible:outline-2 focus-visible:outline-sky-400"
                >
                  Focus camera on {overlay.circuit.name}
                </button>
              )}
              <h4 className="mb-1.5 text-[10px] font-medium text-slate-500 uppercase">
                Roles in {overlay.circuit.name}
              </h4>
              <ul className="space-y-1">
                {getRoleLegend(overlay).map((entry) => (
                  <li key={entry.roleId} className="flex items-center gap-2 text-xs">
                    <RoleMarker style={entry.style} />
                    <span className="flex-1 text-slate-200">{entry.label}</span>
                    <span className="text-[10px] text-slate-500 tabular-nums">{entry.memberCount}</span>
                  </li>
                ))}
              </ul>
              <p className="mt-2 text-[10px] leading-snug text-slate-500">
                A ring marks membership in this circuit. It is not a measured score.
              </p>
              {edgeGroups.length > 0 && <RelationList view={view} groups={edgeGroups} />}
            </div>
          )}
        </PanelSection>

        <PanelSection title="Detectors in this bundle">
          {view.detectors.length === 0 ? (
            <EmptyNote>No functional detectors were run for this bundle.</EmptyNote>
          ) : (
            <>
              <ul className="space-y-2">
                {shownDetectors.map((d) => (
                  <li key={d.id} className="text-xs">
                    <div className="flex flex-wrap items-center justify-between gap-x-2 gap-y-0.5">
                      <span className="font-mono text-slate-200">
                        {d.id} <span className="text-slate-500">v{d.version}</span>
                      </span>
                      <span className="flex gap-1">
                        {d.statuses.map((status) => (
                          <EvidenceBadge key={status} status={status} />
                        ))}
                      </span>
                    </div>
                    {experiment && <TopHeads view={view} experimentId={experiment.id} detectorId={d.id} />}
                  </li>
                ))}
              </ul>
              {hiddenDetectors > 0 && (
                <p className="mt-2 text-[10px] text-slate-500 italic">
                  {hiddenDetectors} more detector{hiddenDetectors === 1 ? " was" : "s were"} not run on this
                  experiment; pick another experiment to see {hiddenDetectors === 1 ? "it" : "them"}.
                </p>
              )}
              <p className="mt-2 text-[10px] leading-snug text-slate-500">
                Top heads on the selected experiment, ranked on each detector&apos;s own scale: scores
                of different detectors are not comparable. Click a head to inspect it. Score overlays
                and thresholds on the 3D scene are not built.
              </p>
            </>
          )}
        </PanelSection>

        <PanelSection title="Heads">
          <HeadList view={view} overlay={overlay} />
        </PanelSection>
      </div>
    </aside>
  );
}
