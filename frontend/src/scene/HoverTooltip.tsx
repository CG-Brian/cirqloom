// DOM tooltip for the hovered head, drawn over the canvas. Lives outside <Canvas> so a hover
// re-renders only this component, and it is positioned by the parent without React state.
import { forwardRef } from "react";
import type { GraphView } from "../features/experiment/graphView";
import type { CircuitOverlay } from "../features/overlays/selectors";
import { useExplorerStore } from "../features/selection/store";

interface HoverTooltipProps {
  view: GraphView;
  overlay: CircuitOverlay | null;
}

export const HoverTooltip = forwardRef<HTMLDivElement, HoverTooltipProps>(function HoverTooltip(
  { view, overlay },
  ref,
) {
  const hoveredNodeId = useExplorerStore((s) => s.hoveredNodeId);
  const node = hoveredNodeId === null ? undefined : view.nodeById.get(hoveredNodeId);
  const roleIds = node ? (overlay?.rolesByNode.get(node.id) ?? []) : [];
  const roles = roleIds.map((id) => overlay?.circuit.roles.find((r) => r.id === id)?.label ?? id);

  return (
    <div
      ref={ref}
      role="tooltip"
      hidden={!node}
      className="pointer-events-none absolute top-0 left-0 z-10 whitespace-nowrap rounded bg-slate-900/90 px-2 py-0.5 text-xs text-slate-100 shadow"
    >
      {node && <span className="font-mono">{node.label}</span>}
      {roles.length > 0 && <span className="ml-2 text-slate-400">{roles.join(", ")}</span>}
    </div>
  );
});
