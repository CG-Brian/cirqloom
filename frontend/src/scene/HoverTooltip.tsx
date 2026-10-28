// DOM tooltip for the hovered head, drawn over the canvas. Lives outside <Canvas> so a hover
// re-renders only this component, and it is positioned by the parent without React state.
import { forwardRef } from "react";
import { EVIDENCE_KIND_LABEL, edgeGroupLabel, type VisualEdgeGroup } from "../features/circuits/edgeGroups";
import type { GraphView } from "../features/experiment/graphView";
import type { CircuitOverlay } from "../features/overlays/selectors";
import { useExplorerStore } from "../features/selection/store";

interface HoverTooltipProps {
  view: GraphView;
  overlay: CircuitOverlay | null;
  edgeGroups: VisualEdgeGroup[];
}

export const HoverTooltip = forwardRef<HTMLDivElement, HoverTooltipProps>(function HoverTooltip(
  { view, overlay, edgeGroups },
  ref,
) {
  const hoveredNodeId = useExplorerStore((s) => s.hoveredNodeId);
  const hoveredEdgeGroupId = useExplorerStore((s) => s.hoveredEdgeGroupId);
  const group = hoveredNodeId === null ? edgeGroups.find((g) => g.id === hoveredEdgeGroupId) : undefined;
  const node = hoveredNodeId === null ? undefined : view.nodeById.get(hoveredNodeId);
  const roleIds = node ? (overlay?.rolesByNode.get(node.id) ?? []) : [];
  const roles = roleIds.map((id) => overlay?.circuit.roles.find((r) => r.id === id)?.label ?? id);

  return (
    <div
      ref={ref}
      role="tooltip"
      hidden={!node && !group}
      className="pointer-events-none absolute top-0 left-0 z-10 whitespace-nowrap rounded bg-slate-900/90 px-2 py-0.5 text-xs text-slate-100 shadow"
    >
      {node && <span className="font-mono">{node.label}</span>}
      {roles.length > 0 && <span className="ml-2 text-slate-400">{roles.join(", ")}</span>}
      {group && (
        <>
          <span>{edgeGroupLabel(view, group)}</span>
          <span className="ml-2 text-slate-400">
            {group.edges.length} record{group.edges.length === 1 ? "" : "s"} ·{" "}
            {EVIDENCE_KIND_LABEL[group.evidenceKind]} · click for evidence
          </span>
        </>
      )}
    </div>
  );
});
