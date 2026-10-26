// Right panel: everything known about the selected head, kept apart by evidence kind:
// circuit membership, published annotations, measured observations, and attention data.
// Missing data is stated as missing; nothing is shown as a zero or a placeholder chart.
import type { Manifest, Observation } from "../data/schema";
import type { GraphView } from "../features/experiment/graphView";
import { getHeadEvidence, type HeadMembership } from "../features/overlays/selectors";
import { useExplorerStore } from "../features/selection/store";
import {
  CollapseButton,
  EmptyNote,
  EvidenceBadge,
  OriginBadge,
  PanelSection,
  RoleMarker,
} from "./ui";

interface HeadInspectorProps {
  view: GraphView;
  manifest: Manifest;
  collapsed: boolean;
  onToggleCollapsed: () => void;
}

function ExternalRef({ value }: { value: string }) {
  return /^https?:\/\//.test(value) ? (
    <a href={value} target="_blank" rel="noreferrer" className="break-all text-sky-300 hover:underline">
      {value}
    </a>
  ) : (
    <span className="break-words">{value}</span>
  );
}

function MembershipItem({ item }: { item: HeadMembership }) {
  const { membership, circuit, roleLabel, style } = item;
  return (
    <li className="rounded border border-slate-800 px-2.5 py-2">
      <div className="flex items-center gap-2">
        <RoleMarker style={style} />
        <span className="text-sm text-slate-100">{roleLabel}</span>
      </div>
      <div className="mt-1 text-[11px] text-slate-400">in {circuit.name}</div>
      <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
        <OriginBadge origin={circuit.origin} />
        <EvidenceBadge status={membership.evidence_status} />
      </div>
      <div className="mt-1.5 text-[10px] text-slate-500">
        Source: <ExternalRef value={membership.source_reference} />
      </div>
    </li>
  );
}

function ObservationRow({ obs }: { obs: Observation }) {
  return (
    <li className="rounded border border-slate-800 px-2.5 py-2">
      <div className="flex items-center justify-between gap-2">
        <span className="font-mono text-xs text-slate-200">
          {obs.detector_id} <span className="text-slate-500">v{obs.detector_version}</span>
        </span>
        <EvidenceBadge status={obs.status} />
      </div>
      <div className="mt-1 flex items-baseline gap-2">
        {obs.score === null ? (
          <span className="text-xs text-slate-400 italic">Not measured</span>
        ) : (
          <span className="font-mono text-sm text-slate-100 tabular-nums">{obs.score.toPrecision(3)}</span>
        )}
        <span className="text-[10px] text-slate-500">
          n = {obs.sample_count} · {obs.evidence_type.replaceAll("_", " ")} · {obs.experiment_id}
        </span>
      </div>
      <div className="mt-1 text-[10px] text-slate-500">
        Method: <ExternalRef value={obs.method_reference} />
      </div>
    </li>
  );
}

export function HeadInspector({ view, manifest, collapsed, onToggleCollapsed }: HeadInspectorProps) {
  const selectedNodeId = useExplorerStore((s) => s.selectedNodeId);
  const select = useExplorerStore((s) => s.select);
  const node = selectedNodeId === null ? undefined : view.nodeById.get(selectedNodeId);

  if (collapsed) {
    return (
      <aside
        aria-label="Head inspector"
        className="flex shrink-0 items-start border-slate-800 bg-slate-950/70 p-1 max-lg:border-t lg:w-9 lg:flex-col lg:border-l"
      >
        <CollapseButton label="head inspector" expanded={false} onClick={onToggleCollapsed} direction="right" />
        <span className="px-1 text-[11px] text-slate-500 lg:hidden">
          Inspector{node ? ` · ${node.label}` : ""}
        </span>
      </aside>
    );
  }

  const evidence = node ? getHeadEvidence(view, node.id) : null;
  const noExperiments = manifest.experiments.length === 0;

  return (
    <aside
      aria-label="Head inspector"
      className="flex shrink-0 flex-col border-slate-800 bg-slate-950/70 max-lg:border-t lg:w-72 lg:border-l"
    >
      <div className="flex items-center justify-between border-b border-slate-800/80 px-3 py-2">
        <h2 className="text-xs font-semibold text-slate-200">Head inspector</h2>
        <CollapseButton label="head inspector" expanded onClick={onToggleCollapsed} direction="right" />
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto">
        {!node || !evidence ? (
          <div className="p-3">
            <EmptyNote>Select a head in the scene or the head list to inspect it.</EmptyNote>
          </div>
        ) : (
          <>
            <PanelSection
              title="Identity"
              aside={
                <button
                  type="button"
                  onClick={() => select(null)}
                  className="rounded px-1.5 text-[11px] text-slate-400 hover:bg-slate-800 hover:text-slate-100"
                >
                  Clear
                </button>
              }
            >
              <p className="font-mono text-lg text-slate-50">{node.label}</p>
              <dl className="mt-2 grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-xs">
                <dt className="text-slate-500">Stable ID</dt>
                <dd className="font-mono text-[11px] break-all text-slate-300">{node.id}</dd>
                <dt className="text-slate-500">Model</dt>
                <dd className="text-slate-300">
                  {manifest.model.id} <span className="text-slate-500">({manifest.model.hf_id})</span>
                </dd>
                <dt className="text-slate-500">Revision</dt>
                <dd className="text-slate-300">
                  {manifest.model.revision ?? <span className="italic text-slate-400">none — no weights loaded</span>}
                </dd>
                <dt className="text-slate-500">Layer · head</dt>
                <dd className="text-slate-300">
                  {node.layer} · {node.head} <span className="text-slate-500">(zero-based)</span>
                </dd>
              </dl>
            </PanelSection>

            <PanelSection title="Circuit membership">
              {evidence.memberships.length === 0 ? (
                <EmptyNote>Not a member of any circuit in this bundle.</EmptyNote>
              ) : (
                <ul className="space-y-1.5">
                  {evidence.memberships.map((item) => (
                    <MembershipItem
                      key={`${item.circuit.id}:${item.membership.role}`}
                      item={item}
                    />
                  ))}
                </ul>
              )}
            </PanelSection>

            <PanelSection title="Published annotations">
              {evidence.published.length === 0 ? (
                <EmptyNote>
                  No published annotations for this head
                  {manifest.synthetic_fixture ? " — this bundle is a synthetic fixture." : " in this bundle."}
                </EmptyNote>
              ) : (
                <ul className="space-y-1.5 text-xs">
                  {evidence.published.map(({ circuit, roleLabel, membership }) => (
                    <li key={`${circuit.id}:${membership.role}`} className="rounded border border-sky-400/20 px-2.5 py-2">
                      <div className="text-slate-100">
                        {roleLabel} <span className="text-slate-400">in {circuit.name}</span>
                      </div>
                      {circuit.citation && (
                        <div className="mt-1 text-[11px] text-slate-400">
                          <ExternalRef value={circuit.citation} />
                        </div>
                      )}
                    </li>
                  ))}
                </ul>
              )}
            </PanelSection>

            <PanelSection
              title="Measured observations"
              aside={
                <span className="text-[10px] text-slate-500">
                  {manifest.experiments.length} experiment{manifest.experiments.length === 1 ? "" : "s"}
                </span>
              }
            >
              {evidence.observations.length === 0 ? (
                <EmptyNote>
                  No detector observations for this head
                  {noExperiments ? " — this bundle contains no experiments." : " in this bundle."}
                </EmptyNote>
              ) : (
                <ul className="space-y-1.5">
                  {evidence.observations.map((obs) => (
                    <ObservationRow key={`${obs.experiment_id}:${obs.detector_id}`} obs={obs} />
                  ))}
                </ul>
              )}
            </PanelSection>

            <PanelSection title="Attention">
              <EmptyNote>
                {noExperiments
                  ? "No attention data: this bundle contains no measured runs, so there is no heatmap to show."
                  : "Attention heatmaps are not available in this version of the explorer."}
              </EmptyNote>
            </PanelSection>

            <PanelSection title="What this does not show">
              <p className="text-[11px] leading-relaxed text-slate-500">
                Rings and roles describe annotations in this bundle; they do not establish a causal
                role. Attention weights are token-to-token inside one head and are not connections
                between heads.
              </p>
            </PanelSection>
          </>
        )}
      </div>
    </aside>
  );
}
