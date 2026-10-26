// Keyboard-accessible parallel to the 3D scene (AGENTS.md §9): every head is a real button.
// Selection lives in the shared store, so picking here and clicking in 3D stay in sync.
//
// Keys: one Tab stop for the whole grid (roving tabindex). Arrow keys move and select
// (Left/Right = head, Up/Down = layer), Home/End jump within a layer, Escape clears (App).
import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import type { GraphView, HeadView } from "../features/experiment/graphView";
import {
  getHeadEvidence,
  getRoleStyle,
  type CircuitOverlay,
} from "../features/overlays/selectors";
import { useExplorerStore } from "../features/selection/store";

interface HeadListProps {
  view: GraphView;
  overlay: CircuitOverlay | null;
}

function neighbour(view: GraphView, from: HeadView, key: string): HeadView | undefined {
  const inLayer = (layer: number, head: number) =>
    view.nodesByLayer[layer]?.find((n) => n.head === head);
  const layerNodes = view.nodesByLayer[from.layer] ?? [];
  switch (key) {
    case "ArrowRight":
      return inLayer(from.layer, from.head + 1);
    case "ArrowLeft":
      return inLayer(from.layer, from.head - 1);
    case "ArrowDown":
      return inLayer(from.layer + 1, from.head);
    case "ArrowUp":
      return inLayer(from.layer - 1, from.head);
    case "Home":
      return layerNodes[0];
    case "End":
      return layerNodes[layerNodes.length - 1];
    default:
      return undefined;
  }
}

function accessibleName(view: GraphView, node: HeadView): string {
  const roles = getHeadEvidence(view, node.id).memberships.map(
    (m) => `${m.roleLabel} in ${m.circuit.name}`,
  );
  return roles.length > 0 ? `${node.label}, ${roles.join("; ")}` : node.label;
}

export function HeadList({ view, overlay }: HeadListProps) {
  const selectedNodeId = useExplorerStore((s) => s.selectedNodeId);
  const select = useExplorerStore((s) => s.select);
  const buttons = useRef(new Map<string, HTMLButtonElement>());
  const [focusedId, setFocusedId] = useState<string | null>(null);

  const tabStopId = focusedId ?? selectedNodeId ?? view.nodes[0]?.id ?? null;

  // A selection made in the 3D scene scrolls its button into view.
  useEffect(() => {
    if (selectedNodeId) {
      buttons.current.get(selectedNodeId)?.scrollIntoView?.({ block: "nearest" });
    }
  }, [selectedNodeId]);

  const onKeyDown = (node: HeadView) => (e: KeyboardEvent<HTMLButtonElement>) => {
    const next = neighbour(view, node, e.key);
    if (!next) {
      return;
    }
    e.preventDefault();
    select(next.id);
    buttons.current.get(next.id)?.focus();
  };

  return (
    <div>
      <div role="group" aria-label="Attention heads by layer" className="space-y-0.5">
        {view.nodesByLayer.map((nodes, layer) => (
          <div key={layer} role="group" aria-label={`Layer ${layer}`} className="flex items-center gap-1.5">
            <span aria-hidden="true" className="w-6 shrink-0 text-right font-mono text-[10px] text-slate-500">
              L{layer}
            </span>
            <div className="grid flex-1 grid-cols-12 gap-0.5">
              {nodes.map((node) => {
                const selected = node.id === selectedNodeId;
                const role = overlay?.rolesByNode.get(node.id)?.[0];
                const ringColor = overlay && role ? getRoleStyle(overlay.circuit, role).color : null;
                const receded = overlay !== null && ringColor === null && !selected;
                return (
                  <button
                    key={node.id}
                    ref={(el) => {
                      if (el) {
                        buttons.current.set(node.id, el);
                      } else {
                        buttons.current.delete(node.id);
                      }
                    }}
                    type="button"
                    tabIndex={node.id === tabStopId ? 0 : -1}
                    aria-pressed={selected}
                    aria-label={accessibleName(view, node)}
                    title={node.label}
                    onClick={() => select(node.id)}
                    onKeyDown={onKeyDown(node)}
                    onFocus={() => setFocusedId(node.id)}
                    onBlur={() => setFocusedId(null)}
                    style={ringColor ? { boxShadow: `inset 0 0 0 1px ${ringColor}` } : undefined}
                    className={[
                      "h-5 rounded-sm font-mono text-[10px] tabular-nums transition-colors",
                      "focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-sky-400",
                      selected
                        ? "bg-slate-100 text-slate-900"
                        : "bg-slate-800/70 text-slate-400 hover:bg-slate-700 hover:text-slate-100",
                      receded ? "opacity-45" : "",
                    ].join(" ")}
                  >
                    {node.head}
                  </button>
                );
              })}
            </div>
          </div>
        ))}
      </div>
      <p className="mt-2 text-[10px] text-slate-500">
        Arrow keys move · Home/End jump · Esc clears
      </p>
    </div>
  );
}
