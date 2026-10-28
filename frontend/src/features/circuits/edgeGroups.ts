// Evidence → visualisation, without changing the evidence (AGENTS.md §8, §9, ADR 0004).
//
// The bundle stores each relation at the granularity it was measured at (e.g. one sender head
// into a whole role group). For drawing, edges that say the same thing about the same
// experiment are bundled into one VisualEdgeGroup. A group keeps the original Edge records,
// unmodified, and is never turned into head-pair edges. Nothing here imports React or three.
import type { Edge, EdgeEvidenceKind, Endpoint } from "../../data/schema";
import type { GraphView } from "../experiment/graphView";

export interface VisualEdgeGroup {
  /** Stable: derived from the first record's id. */
  id: string;
  circuitId: string;
  /** The original evidence records, in bundle order. Never modified. */
  edges: Edge[];
  /** Shared by every record (part of the grouping key). */
  sourceKind: Endpoint["kind"];
  target: Endpoint;
  channel: Edge["channel"];
  evidenceKind: EdgeEvidenceKind;
  evidenceStatus: Edge["evidence_status"];
  effect: Edge["effect"];
  conditions: string[];
  positions: Edge["positions"];
  locator: string | null;
}

export function endpointKey(end: Endpoint): string {
  switch (end.kind) {
    case "head":
      return `head:${end.node_id}`;
    case "role_group":
      return `role:${end.circuit_id}:${end.role}`;
    case "special":
      return `special:${end.id}`;
  }
}

/** Everything that must match for two records to be drawn as one ribbon. */
export function edgeGroupKey(edge: Edge): string {
  return JSON.stringify([
    edge.circuit_id,
    edge.source.kind,
    endpointKey(edge.target),
    edge.channel,
    edge.evidence_kind,
    edge.evidence_status,
    edge.locator,
    edge.effect,
    [...edge.conditions].sort(),
    edge.positions,
  ]);
}

/** The edges of one circuit, bundled by meaning. Every edge appears in exactly one group. */
export function getVisualEdgeGroups(view: GraphView, circuitId: string | null): VisualEdgeGroup[] {
  if (circuitId === null) {
    return [];
  }
  const groups = new Map<string, VisualEdgeGroup>();
  for (const edge of view.edges) {
    if (edge.circuit_id !== circuitId) {
      continue;
    }
    const key = edgeGroupKey(edge);
    const group = groups.get(key);
    if (group) {
      group.edges.push(edge);
    } else {
      groups.set(key, {
        id: `vg:${edge.id}`,
        circuitId,
        edges: [edge],
        sourceKind: edge.source.kind,
        target: edge.target,
        channel: edge.channel,
        evidenceKind: edge.evidence_kind,
        evidenceStatus: edge.evidence_status,
        effect: edge.effect,
        conditions: [...edge.conditions],
        positions: edge.positions,
        locator: edge.locator,
      });
    }
  }
  return [...groups.values()];
}

/** Roles of a head in one circuit, in declared order. */
function rolesOf(view: GraphView, circuitId: string, nodeId: string): string[] {
  const circuit = view.circuitById.get(circuitId);
  const order = new Map(circuit?.roles.map((r, i) => [r.id, i]));
  return (view.membershipsByNode.get(nodeId) ?? [])
    .filter((m) => m.circuit_id === circuitId)
    .map((m) => m.role)
    .sort((a, b) => (order.get(a) ?? 0) - (order.get(b) ?? 0));
}

export function roleLabel(view: GraphView, circuitId: string, roleId: string): string {
  return view.circuitById.get(circuitId)?.roles.find((r) => r.id === roleId)?.label ?? roleId;
}

/** Member heads of a role group, in node order. */
export function roleMembers(view: GraphView, circuitId: string, roleId: string): string[] {
  return view.nodes
    .filter((n) => (view.membershipsByNode.get(n.id) ?? []).some(
      (m) => m.circuit_id === circuitId && m.role === roleId,
    ))
    .map((n) => n.id);
}

/** Roles the senders of a group play (one for a group source; the union for head senders). */
export function senderRoles(view: GraphView, group: VisualEdgeGroup): string[] {
  const roles: string[] = [];
  for (const edge of group.edges) {
    const found = edge.source.kind === "head"
      ? rolesOf(view, group.circuitId, edge.source.node_id)
      : edge.source.kind === "role_group" ? [edge.source.role] : [];
    for (const role of found) {
      if (!roles.includes(role)) {
        roles.push(role);
      }
    }
  }
  const order = new Map(view.circuitById.get(group.circuitId)?.roles.map((r, i) => [r.id, i]));
  return roles.sort((a, b) => (order.get(a) ?? 0) - (order.get(b) ?? 0));
}

export const CHANNEL_LABEL: Record<Edge["channel"], { short: string; long: string }> = {
  q: { short: "Q", long: "queries" },
  k: { short: "K", long: "keys" },
  v: { short: "V", long: "values" },
  direct_logit: { short: "logits", long: "direct effect on the logits" },
};

export const EVIDENCE_KIND_LABEL: Record<EdgeEvidenceKind, string> = {
  published_schematic: "Published schematic",
  published_experimental: "Published experiment",
  weight_composition: "Weight composition (ours)",
  causal_intervention: "Causal intervention (ours)",
};

export function endpointLabel(view: GraphView, end: Endpoint): string {
  switch (end.kind) {
    case "head":
      return view.nodeById.get(end.node_id)?.label ?? end.node_id;
    case "role_group":
      return roleLabel(view, end.circuit_id, end.role);
    case "special":
      return end.id;
  }
}

/** e.g. "S-Inhibition heads → Name Mover queries", "S-Inhibition group → Name Mover queries". */
export function edgeGroupLabel(view: GraphView, group: VisualEdgeGroup): string {
  const senders = senderRoles(view, group).map((r) => roleLabel(view, group.circuitId, r)).join(" + ");
  const from = `${senders || "?"} ${group.sourceKind === "role_group" ? "group" : "heads"}`;
  if (group.target.kind === "special") {
    return `${from} → ${group.target.id}`;
  }
  return `${from} → ${endpointLabel(view, group.target)} ${CHANNEL_LABEL[group.channel].long}`;
}

/**
 * Heads a group is about: its sender heads (or the members of a sender group) and the members
 * of a receiving group. Used to emphasise them on hover; it adds no relation between them.
 */
export function edgeGroupHeads(view: GraphView, group: VisualEdgeGroup): {
  senders: string[];
  receivers: string[];
} {
  const senders: string[] = [];
  for (const edge of group.edges) {
    const ids = edge.source.kind === "head"
      ? [edge.source.node_id]
      : edge.source.kind === "role_group"
        ? roleMembers(view, edge.source.circuit_id, edge.source.role)
        : [];
    for (const id of ids) {
      if (!senders.includes(id)) {
        senders.push(id);
      }
    }
  }
  const receivers = group.target.kind === "head"
    ? [group.target.node_id]
    : group.target.kind === "role_group"
      ? roleMembers(view, group.target.circuit_id, group.target.role)
      : [];
  return { senders, receivers };
}

/** Groups whose evidence involves this head (as a sender, or as a member of the receiver). */
export function edgeGroupsForHead(
  view: GraphView,
  groups: VisualEdgeGroup[],
  nodeId: string,
): VisualEdgeGroup[] {
  return groups.filter((g) => {
    const { senders, receivers } = edgeGroupHeads(view, g);
    return senders.includes(nodeId) || receivers.includes(nodeId);
  });
}

export type EdgeLinePattern = "solid" | "dashed" | "dotted";

export interface EdgeDisplayStyle {
  color: string;
  opacity: number;
  /** Screen pixels. */
  lineWidth: number;
  pattern: EdgeLinePattern;
}

export type EdgeFocusState = "idle" | "focused" | "dimmed";

/**
 * Line pattern encodes the evidence kind (and conditional evidence), so it never depends on
 * colour alone. Colour stays neutral: ribbons are relations, not role identities.
 */
export function getEdgeDisplayStyle(group: VisualEdgeGroup, state: EdgeFocusState): EdgeDisplayStyle {
  const base: Record<EdgeEvidenceKind, { lineWidth: number; pattern: EdgeLinePattern }> = {
    published_schematic: { lineWidth: 1.5, pattern: "dashed" },
    published_experimental: { lineWidth: 2.5, pattern: "solid" },
    weight_composition: { lineWidth: 1.25, pattern: "solid" },
    causal_intervention: { lineWidth: 3.5, pattern: "solid" },
  };
  const kind = base[group.evidenceKind];
  // Evidence that holds only under a condition (e.g. after a knockout) is dotted.
  const pattern: EdgeLinePattern = group.conditions.length > 0 ? "dotted" : kind.pattern;
  const opacity = state === "focused" ? 0.95 : state === "dimmed" ? 0.15 : 0.55;
  return {
    color: state === "focused" ? "#f8fafc" : "#cbd5e1",
    opacity,
    lineWidth: state === "focused" ? kind.lineWidth + 1 : kind.lineWidth,
    pattern,
  };
}
