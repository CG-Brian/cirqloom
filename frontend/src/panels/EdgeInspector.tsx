// Right panel for one circuit relation (a visual edge group). Shows what the evidence says, at
// the granularity it was measured at, and lists every original evidence record unchanged.
import type { DataProvider } from "../data/DataProvider";
import type { Edge, Experiment, Manifest } from "../data/schema";
import {
  CHANNEL_LABEL,
  edgeGroupLabel,
  endpointLabel,
  EVIDENCE_KIND_LABEL,
  roleLabel,
  senderRoles,
  type VisualEdgeGroup,
} from "../features/circuits/edgeGroups";
import { relationFocusBounds, type CircuitLayout } from "../features/circuits/circuitLayout";
import type { GraphView } from "../features/experiment/graphView";
import { useExplorerStore } from "../features/selection/store";
import { AttentionAlignment } from "./AttentionAlignment";
import { CollapseButton, EvidenceBadge, HeaderButton, OriginBadge, PanelSection } from "./ui";

interface EdgeInspectorProps {
  provider: DataProvider;
  view: GraphView;
  manifest: Manifest;
  /** Observed attention is shown for this experiment's stored prompt. */
  experiment: Experiment | null;
  group: VisualEdgeGroup;
  /** Where the relation is drawn, for "Focus"; null when no circuit is shown. */
  circuitLayout: CircuitLayout | null;
  collapsed: boolean;
  onToggleCollapsed: () => void;
}

/** "after_knockout:name_mover" → "Only after the Name Mover heads are knocked out." */
export function describeCondition(view: GraphView, circuitId: string, condition: string): string {
  const [kind, role] = condition.split(":");
  if (kind === "after_knockout" && role) {
    return `Only after the ${roleLabel(view, circuitId, role)} heads are knocked out.`;
  }
  return condition;
}

function isPublished(group: VisualEdgeGroup): boolean {
  return group.evidenceKind === "published_schematic" || group.evidenceKind === "published_experimental";
}

function receiverText(view: GraphView, group: VisualEdgeGroup): string {
  if (group.target.kind === "special") {
    return group.target.id;
  }
  const name = endpointLabel(view, group.target);
  return group.target.kind === "role_group" ? `the whole ${name} group` : name;
}

function granularityText(view: GraphView, group: VisualEdgeGroup): string {
  // Published relations were measured by the cited paper, not by Cirqloom: say who measured.
  const measured = isPublished(group) ? "Measured by the source" : "Measured";
  const into = group.target.kind === "special"
    ? `into the ${group.target.id}`
    : `into ${receiverText(view, group)} (${CHANNEL_LABEL[group.channel].long})`;
  if (group.sourceKind === "role_group") {
    const role = senderRoles(view, group)[0];
    return `${measured} with all ${role ? roleLabel(view, group.circuitId, role) : ""} heads changed together ${into}. One record.`;
  }
  const n = group.edges.length;
  const pairs = group.target.kind === "role_group"
    ? " The source reports no result for any single sender–receiver head pair, so none is drawn."
    : "";
  return `${measured} one sender head at a time ${into}. ${n} record${n === 1 ? "" : "s"}, one per sender head.${pairs}`;
}

function senderText(view: GraphView, edge: Edge): string {
  const source = edge.source;
  if (source.kind !== "head") {
    return `${endpointLabel(view, source)} ${source.kind === "role_group" ? "group" : ""}`.trim();
  }
  const roles = (view.membershipsByNode.get(source.node_id) ?? [])
    .filter((m) => m.circuit_id === edge.circuit_id)
    .map((m) => roleLabel(view, edge.circuit_id, m.role));
  return `${endpointLabel(view, source)}${roles.length ? ` (${roles.join(", ")})` : ""}`;
}

export function EdgeInspector({
  provider,
  view,
  manifest,
  experiment,
  group,
  circuitLayout,
  collapsed,
  onToggleCollapsed,
}: EdgeInspectorProps) {
  const focusCamera = useExplorerStore((s) => s.focusCamera);
  const selectEdgeGroup = useExplorerStore((s) => s.selectEdgeGroup);
  const select = useExplorerStore((s) => s.select);
  const circuit = view.circuitById.get(group.circuitId);
  const published = isPublished(group);
  const first = group.edges[0]!;

  if (collapsed) {
    return (
      <aside
        aria-label="Relation inspector"
        className="flex shrink-0 items-start border-slate-800 bg-slate-950/70 p-1 max-lg:border-t lg:w-9 lg:flex-col lg:border-l"
      >
        <CollapseButton label="relation inspector" expanded={false} onClick={onToggleCollapsed} direction="right" />
      </aside>
    );
  }

  return (
    <aside
      aria-label="Relation inspector"
      className="flex shrink-0 flex-col border-slate-800 bg-slate-950/70 max-lg:border-t lg:w-96 lg:border-l"
    >
      <div className="flex items-center justify-between border-b border-slate-800/80 px-3 py-2">
        <h2 className="text-xs font-semibold text-slate-200">Relation inspector</h2>
        <CollapseButton label="relation inspector" expanded onClick={onToggleCollapsed} direction="right" />
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto">
        <PanelSection
          title="Relation"
          aside={
            <span className="flex gap-1">
              {circuitLayout && (
                <HeaderButton
                  label="Focus"
                  ariaLabel="Focus the camera on this relation"
                  onClick={() => focusCamera(relationFocusBounds(view, circuitLayout, group))}
                />
              )}
              <HeaderButton label="Clear" onClick={() => selectEdgeGroup(null)} />
            </span>
          }
        >
          <p className="text-sm text-slate-50">{edgeGroupLabel(view, group)}</p>
          {circuit && <p className="mt-0.5 text-[11px] text-slate-400">in {circuit.name}</p>}
          <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
            <span className="inline-flex rounded border border-slate-600 px-1.5 py-px text-[10px] leading-4 text-slate-200">
              {EVIDENCE_KIND_LABEL[group.evidenceKind]}
            </span>
            <EvidenceBadge status={group.evidenceStatus} />
            {circuit && <OriginBadge origin={circuit.origin} />}
          </div>
        </PanelSection>

        <PanelSection title="What the evidence says">
          <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1.5 text-xs">
            <dt className="text-slate-500">Sender</dt>
            <dd className="text-slate-200">
              {senderRoles(view, group).map((r) => roleLabel(view, group.circuitId, r)).join(" + ")}{" "}
              {group.sourceKind === "role_group" ? "group" : `heads (${group.edges.length})`}
            </dd>
            <dt className="text-slate-500">Receiver</dt>
            <dd className="text-slate-200">{receiverText(view, group)}</dd>
            <dt className="text-slate-500">Channel</dt>
            <dd className="text-slate-200">
              {CHANNEL_LABEL[group.channel].long}
              {group.channel !== "direct_logit" && <span className="text-slate-500"> ({CHANNEL_LABEL[group.channel].short})</span>}
            </dd>
            {group.positions && (
              <>
                <dt className="text-slate-500">Positions</dt>
                <dd className="text-slate-200">
                  sender at {group.positions.source ?? "—"}
                  {group.positions.target && <> → receiver at {group.positions.target}</>}
                </dd>
              </>
            )}
            <dt className="text-slate-500">Effect</dt>
            <dd className="text-slate-200">
              {group.effect
                ? `The sender's own contribution ${group.effect.direction === "increase" ? "increases" : "decreases"} ${group.effect.metric.replaceAll("_", " ")}.`
                : <span className="italic text-slate-400">Not stated as a direction.</span>}
            </dd>
            <dt className="text-slate-500">Conditions</dt>
            <dd className="text-slate-200">
              {group.conditions.length === 0
                ? published
                  ? "None beyond the source's own setup (its task prompts, unmodified model)."
                  : "None: holds in the unmodified model on the stated prompts."
                : group.conditions.map((c) => <div key={c}>{describeCondition(view, group.circuitId, c)}</div>)}
            </dd>
            <dt className="text-slate-500">Magnitude</dt>
            <dd className="text-slate-200">
              {first.magnitude === null
                ? <span className="italic text-slate-400">Not recorded{published ? " (never read off a figure)" : ""}.</span>
                : first.magnitude}
            </dd>
          </dl>
          <p data-testid="relation-granularity" className="mt-2 rounded bg-slate-900 px-2 py-1.5 text-[11px] leading-snug text-slate-300">
            {granularityText(view, group)}
          </p>
        </PanelSection>

        <PanelSection title="Source">
          {published ? (
            <div className="text-xs text-slate-300">
              <div>{circuit?.citation ?? "No citation"}</div>
              <div className="mt-1 text-slate-400">Where: {group.locator}</div>
            </div>
          ) : (
            <div className="text-xs text-slate-300">Method: {first.method_reference}</div>
          )}
        </PanelSection>

        <PanelSection
          title="Observed attention (measured here)"
          aside={<EvidenceBadge status="experimental" />}
        >
          <AttentionAlignment provider={provider} view={view} manifest={manifest} experiment={experiment} group={group} />
        </PanelSection>

        <PanelSection title={`Evidence records (${group.edges.length})`}>
          <ul className="space-y-1.5" aria-label="Evidence records">
            {group.edges.map((edge) => (
              <li key={edge.id} className="rounded border border-slate-800 px-2.5 py-2">
                <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
                  <span className="text-slate-100">
                    {senderText(view, edge)} → {endpointLabel(view, edge.target)}
                  </span>
                  {edge.source.kind === "head" && (
                    <button
                      type="button"
                      onClick={() => edge.source.kind === "head" && select(edge.source.node_id)}
                      className="rounded border border-slate-700 px-1.5 py-0.5 text-[10px] text-slate-300 hover:border-slate-500"
                    >
                      Inspect head
                    </button>
                  )}
                </div>
                <div className="mt-1 text-[11px] leading-snug text-slate-400">{edge.description}</div>
                <div className="mt-1 font-mono text-[9px] break-all text-slate-600">{edge.id}</div>
              </li>
            ))}
          </ul>
        </PanelSection>

        <PanelSection title="What this does not show">
          <p className="text-[11px] leading-relaxed text-slate-500">
            The ribbon is drawn between group anchors for readability. It is not a physical wire
            between heads; its width and colour are not an amount of information; and it does not
            claim that every member of a group takes part. Published relations hold for the task
            and prompts the source studied.
          </p>
        </PanelSection>
      </div>
    </aside>
  );
}
