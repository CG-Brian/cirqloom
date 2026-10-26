// Pure highlight logic (AGENTS.md §9, §10). Inputs are the GraphView and plain flags from the
// store; outputs are plain style descriptions. Nothing here imports three.js or React.
//
// Emphasis rules:
// - A head's base colour is always neutral. Membership never recolours the whole node.
// - The circuit is the visual focus: its layers get an outline accent, non-members recede.
// - Members get a thin role-coloured ring and a faint halo, only while that overlay is active.
// - Selection adds a neutral bright ring on top, so it reads the same under any overlay.
// - No edges are derived here. Members are never connected by guessed lines.
import type { Circuit, Membership, Observation } from "../../data/schema";
import { boundsOf, type Bounds, type Vec3 } from "../../scene/layout";
import type { GraphView } from "../experiment/graphView";

export type LinePattern = "solid" | "dashed" | "dotted" | "dash-dot";
export type MarkerShape = "circle" | "square" | "triangle" | "diamond";

export interface RoleStyle {
  color: string;
  pattern: LinePattern;
  marker: MarkerShape;
}

export const PALETTE = {
  neutralNode: "#64748b",
  selectionRing: "#f8fafc",
  layerFrame: "#334155",
  circuitOutline: "#a5b4fc",
} as const;

export const EMPHASIS = {
  memberRingOpacity: 0.8,
  memberHaloOpacity: 0.12,
  subduedNodeOpacity: 0.35,
  hoverScale: 1.15,
  selectedScale: 1.08,
} as const;

/**
 * Okabe–Ito colours (colour-blind safe). Each also has a distinct pattern + marker pair,
 * so roles stay distinguishable without colour. Assigned by the role's index in its circuit.
 */
export const ROLE_STYLES: readonly RoleStyle[] = [
  { color: "#56B4E9", pattern: "solid", marker: "circle" },
  { color: "#E69F00", pattern: "dashed", marker: "square" },
  { color: "#009E73", pattern: "dotted", marker: "triangle" },
  { color: "#CC79A7", pattern: "dash-dot", marker: "diamond" },
  { color: "#0072B2", pattern: "dashed", marker: "circle" },
  { color: "#D55E00", pattern: "dotted", marker: "square" },
  { color: "#F0E442", pattern: "dash-dot", marker: "triangle" },
];

const UNKNOWN_ROLE_STYLE: RoleStyle = {
  color: PALETTE.neutralNode,
  pattern: "dotted",
  marker: "circle",
};

export function getRoleStyle(circuit: Circuit, roleId: string): RoleStyle {
  const index = circuit.roles.findIndex((r) => r.id === roleId);
  if (index < 0) {
    return UNKNOWN_ROLE_STYLE;
  }
  return ROLE_STYLES[index % ROLE_STYLES.length] ?? UNKNOWN_ROLE_STYLE;
}

/** Everything the scene needs to draw one active circuit. */
export interface CircuitOverlay {
  circuit: Circuit;
  /** Member node id -> its roles in this circuit, in the circuit's declared role order. */
  rolesByNode: Map<string, string[]>;
  /** Layers containing at least one member. */
  layers: Set<number>;
  /** Bounds of the member heads, for framing the circuit. */
  bounds: Bounds;
}

export function getCircuitOverlay(view: GraphView, circuitId: string | null): CircuitOverlay | null {
  if (circuitId === null) {
    return null;
  }
  const circuit = view.circuitById.get(circuitId);
  if (!circuit) {
    return null;
  }
  const roleOrder = new Map(circuit.roles.map((r, i) => [r.id, i]));
  const rank = (roleId: string) => roleOrder.get(roleId) ?? circuit.roles.length;

  const rolesByNode = new Map<string, string[]>();
  const layers = new Set<number>();
  const positions: Vec3[] = [];
  for (const node of view.nodes) {
    const roles = (view.membershipsByNode.get(node.id) ?? [])
      .filter((m) => m.circuit_id === circuitId)
      .map((m) => m.role);
    if (roles.length === 0) {
      continue;
    }
    rolesByNode.set(node.id, [...new Set(roles)].sort((a, b) => rank(a) - rank(b)));
    layers.add(node.layer);
    positions.push(node.position);
  }
  return { circuit, rolesByNode, layers, bounds: boundsOf(positions) };
}

export function getCircuitMembers(view: GraphView, circuitId: string): Set<string> {
  return new Set(getCircuitOverlay(view, circuitId)?.rolesByNode.keys());
}

/** Bounds of any set of heads (e.g. a selection), or null if none of the ids are known. */
export function getBoundsForSelection(view: GraphView, nodeIds: Iterable<string>): Bounds | null {
  const positions: Vec3[] = [];
  for (const id of nodeIds) {
    const node = view.nodeById.get(id);
    if (node) {
      positions.push(node.position);
    }
  }
  return positions.length > 0 ? boundsOf(positions) : null;
}

export interface NodeStyleInput {
  nodeId: string;
  isSelected: boolean;
  isHovered: boolean;
  overlay: CircuitOverlay | null;
}

export interface RoleRing extends RoleStyle {
  ringOpacity: number;
  haloOpacity: number;
  /** Further roles this head has in the same circuit (shown in the tooltip/inspector). */
  extraRoles: number;
}

export interface NodeDisplayStyle {
  color: string;
  opacity: number;
  scale: number;
  roleRing: RoleRing | null;
  selectionRing: { color: string; opacity: number } | null;
}

export function getNodeDisplayStyle(input: NodeStyleInput): NodeDisplayStyle {
  const { nodeId, isSelected, isHovered, overlay } = input;
  const roles = overlay?.rolesByNode.get(nodeId) ?? [];
  const primaryRole = roles[0];

  const roleRing: RoleRing | null =
    overlay && primaryRole !== undefined
      ? {
          ...getRoleStyle(overlay.circuit, primaryRole),
          ringOpacity: EMPHASIS.memberRingOpacity,
          haloOpacity: EMPHASIS.memberHaloOpacity,
          extraRoles: roles.length - 1,
        }
      : null;

  // Non-members recede while a circuit is shown, but never what the user is pointing at.
  const subdued = overlay !== null && roleRing === null && !isSelected && !isHovered;

  return {
    color: PALETTE.neutralNode,
    opacity: subdued ? EMPHASIS.subduedNodeOpacity : 1,
    scale: isHovered ? EMPHASIS.hoverScale : isSelected ? EMPHASIS.selectedScale : 1,
    roleRing,
    selectionRing: isSelected ? { color: PALETTE.selectionRing, opacity: 1 } : null,
  };
}

export interface LayerFrameStyle {
  edgeColor: string;
  edgeOpacity: number;
  /** The fill is always neutral; only the outline carries the circuit accent. */
  fillColor: string;
  fillOpacity: number;
}

/** Circuit-level emphasis: layers the active circuit spans get an outline accent. */
export function getLayerFrameStyle(layer: number, overlay: CircuitOverlay | null): LayerFrameStyle {
  const fillColor = PALETTE.layerFrame;
  if (overlay === null) {
    return { edgeColor: PALETTE.layerFrame, edgeOpacity: 0.6, fillColor, fillOpacity: 0.12 };
  }
  if (overlay.layers.has(layer)) {
    return { edgeColor: PALETTE.circuitOutline, edgeOpacity: 0.45, fillColor, fillOpacity: 0.12 };
  }
  return { edgeColor: PALETTE.layerFrame, edgeOpacity: 0.3, fillColor, fillOpacity: 0.06 };
}

export interface HeadMembership {
  membership: Membership;
  circuit: Circuit;
  roleLabel: string;
  style: RoleStyle;
}

export interface HeadEvidence {
  /** Every circuit membership of this head, across all circuits, in circuit order. */
  memberships: HeadMembership[];
  /** The subset that comes from published research (origin "published"). */
  published: HeadMembership[];
  /** Detector results on this head. Empty means nothing was measured, not a score of zero. */
  observations: Observation[];
}

/** Everything the inspector and list need to describe one head, kept apart by evidence kind. */
export function getHeadEvidence(view: GraphView, nodeId: string): HeadEvidence {
  const circuitOrder = new Map(view.circuits.map((c, i) => [c.id, i]));
  const memberships: HeadMembership[] = [];
  for (const membership of view.membershipsByNode.get(nodeId) ?? []) {
    const circuit = view.circuitById.get(membership.circuit_id);
    if (!circuit) {
      continue;
    }
    const role = circuit.roles.find((r) => r.id === membership.role);
    memberships.push({
      membership,
      circuit,
      roleLabel: role?.label ?? membership.role,
      style: getRoleStyle(circuit, membership.role),
    });
  }
  memberships.sort(
    (a, b) =>
      (circuitOrder.get(a.circuit.id) ?? 0) - (circuitOrder.get(b.circuit.id) ?? 0) ||
      a.circuit.roles.findIndex((r) => r.id === a.membership.role) -
        b.circuit.roles.findIndex((r) => r.id === b.membership.role),
  );
  return {
    memberships,
    published: memberships.filter((m) => m.circuit.origin === "published"),
    observations: view.observationsByNode.get(nodeId) ?? [],
  };
}

export interface RoleLegendEntry {
  roleId: string;
  label: string;
  description: string;
  style: RoleStyle;
  memberCount: number;
}

/** Legend rows in declared role order. Roles with no members are listed with a count of 0. */
export function getRoleLegend(overlay: CircuitOverlay): RoleLegendEntry[] {
  const counts = new Map<string, number>();
  for (const roles of overlay.rolesByNode.values()) {
    for (const role of roles) {
      counts.set(role, (counts.get(role) ?? 0) + 1);
    }
  }
  return overlay.circuit.roles.map((role) => ({
    roleId: role.id,
    label: role.label,
    description: role.description,
    style: getRoleStyle(overlay.circuit, role.id),
    memberCount: counts.get(role.id) ?? 0,
  }));
}
