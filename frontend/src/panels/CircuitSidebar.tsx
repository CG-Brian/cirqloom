// Left panel: circuit overlays, the role legend for the active one, detector overlays, and the
// keyboard head list. Reads and writes the shared store; holds no selection state of its own.
import type { GraphView } from "../features/experiment/graphView";
import { getRoleLegend, type CircuitOverlay } from "../features/overlays/selectors";
import { useExplorerStore } from "../features/selection/store";
import { HeadList } from "./HeadList";
import { CollapseButton, EmptyNote, OriginBadge, PanelSection, RoleMarker } from "./ui";

interface CircuitSidebarProps {
  view: GraphView;
  overlay: CircuitOverlay | null;
  collapsed: boolean;
  onToggleCollapsed: () => void;
}

export function CircuitSidebar({ view, overlay, collapsed, onToggleCollapsed }: CircuitSidebarProps) {
  const activeCircuitId = useExplorerStore((s) => s.activeCircuitId);
  const toggleCircuit = useExplorerStore((s) => s.toggleCircuit);

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
            </div>
          )}
        </PanelSection>

        <PanelSection title="Detector overlays">
          <EmptyNote>No functional detectors were run for this bundle.</EmptyNote>
        </PanelSection>

        <PanelSection title="Heads">
          <HeadList view={view} overlay={overlay} />
        </PanelSection>
      </div>
    </aside>
  );
}
