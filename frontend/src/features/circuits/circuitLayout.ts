// Where a circuit's graph sits in the 3D scene. Pure geometry from the GraphView and the
// edge groups; the renderer draws exactly these points.
//
// - Role-group hubs float in a band above the layer frames, over the centroid of their
//   members, so ribbons run between groups in the air instead of between individual heads.
// - A group whose senders are a subset of a role (e.g. IOI's "fuzzy" heads) gets its own
//   anchor over those heads, with spokes down to each of them: the per-head evidence stays
//   visible without inventing head-to-head lines.
// - The special `logits` endpoint sits after the last layer.
import type { Endpoint } from "../../data/schema";
import { CIRCUIT_BAND_HEADROOM, circuitBandY, layerX, LAYER_SPACING, type Vec3 } from "../../scene/layout";
import type { GraphView } from "../experiment/graphView";
import { getRoleStyle, type RoleStyle } from "../overlays/selectors";
import { edgeGroupHeads, roleMembers, type VisualEdgeGroup } from "./edgeGroups";

export interface Anchor {
  /** "role:<circuit>:<role>", "special:logits" or "subset:<group id>". */
  key: string;
  kind: "role_group" | "special" | "subset";
  position: Vec3;
  label: string;
  /** Role style for role hubs, null otherwise. */
  style: RoleStyle | null;
  /** Heads this anchor stands for; a spoke is drawn to each. */
  nodeIds: string[];
}

export interface EdgeGroupGeometry {
  groupId: string;
  from: string;
  to: string;
  /** Sampled curve points, source → target. */
  points: Vec3[];
  /** Where the arrow tip sits and the direction it points. */
  arrow: { position: Vec3; direction: Vec3 };
  /** Where the channel label goes. */
  mid: Vec3;
}

export interface CircuitLayout {
  anchors: Map<string, Anchor>;
  edges: EdgeGroupGeometry[];
}

const CURVE_SAMPLES = 32;
/** Extra lift for each further group between the same two anchors, so they never overlap. */
const PARALLEL_LIFT = 0.45;
/** Stop the arrow this far before the target anchor so it is not hidden inside the hub. */
const ARROW_GAP = 0.2;

const mean = (points: Vec3[]): Vec3 => {
  const n = Math.max(points.length, 1);
  return [0, 1, 2].map((i) => points.reduce((sum, p) => sum + p[i]!, 0) / n) as Vec3;
};

export function logitsPosition(view: GraphView): Vec3 {
  return [layerX(view.layers - 1, view.layers) + LAYER_SPACING, circuitBandY(view.headsPerLayer), 0];
}

function hubAbove(view: GraphView, nodeIds: string[]): Vec3 {
  const [x, , z] = mean(nodeIds.map((id) => view.nodeById.get(id)?.position ?? [0, 0, 0]));
  return [x, circuitBandY(view.headsPerLayer), z];
}

function roleAnchor(view: GraphView, circuitId: string, role: string): Anchor {
  const circuit = view.circuitById.get(circuitId);
  const nodeIds = roleMembers(view, circuitId, role);
  return {
    key: `role:${circuitId}:${role}`,
    kind: "role_group",
    position: hubAbove(view, nodeIds),
    label: circuit?.roles.find((r) => r.id === role)?.label ?? role,
    style: circuit ? getRoleStyle(circuit, role) : null,
    nodeIds,
  };
}

function endpointAnchor(view: GraphView, end: Endpoint): Anchor | null {
  if (end.kind === "role_group") {
    return roleAnchor(view, end.circuit_id, end.role);
  }
  if (end.kind === "special") {
    const position: Vec3 = end.id === "logits"
      ? logitsPosition(view)
      : [layerX(0, view.layers) - LAYER_SPACING, circuitBandY(view.headsPerLayer), 0];
    return { key: `special:${end.id}`, kind: "special", position, label: end.id, style: null, nodeIds: [] };
  }
  return null;
}

/**
 * The anchor a group's ribbon starts from: the sender role's hub when the senders are exactly
 * that role's members (or the group itself is the sender), otherwise an anchor over the
 * sender heads.
 */
function sourceAnchor(view: GraphView, group: VisualEdgeGroup): Anchor {
  const first = group.edges[0]!.source;
  if (group.sourceKind === "role_group" && first.kind === "role_group") {
    return roleAnchor(view, first.circuit_id, first.role);
  }
  const { senders } = edgeGroupHeads(view, group);
  const roles = new Set(
    senders.flatMap((id) => (view.membershipsByNode.get(id) ?? [])
      .filter((m) => m.circuit_id === group.circuitId)
      .map((m) => m.role)),
  );
  if (roles.size === 1) {
    const [role] = roles;
    const members = roleMembers(view, group.circuitId, role!);
    if (members.length === senders.length && members.every((id) => senders.includes(id))) {
      return roleAnchor(view, group.circuitId, role!);
    }
  }
  return {
    key: `subset:${group.id}`,
    kind: "subset",
    position: hubAbove(view, senders),
    label: `${senders.length} heads`,
    style: null,
    nodeIds: senders,
  };
}

function quadratic(a: Vec3, c: Vec3, b: Vec3, t: number): Vec3 {
  const u = 1 - t;
  return [0, 1, 2].map((i) => u * u * a[i]! + 2 * u * t * c[i]! + t * t * b[i]!) as Vec3;
}

/**
 * Peak height above the band: grows a little with distance, then each further relation between
 * the same anchors goes higher. Clamped to the headroom the camera frames (sceneBounds).
 */
function lift(a: Vec3, b: Vec3, parallelIndex: number): number {
  const distance = Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
  const base = Math.min(0.45 + 0.06 * distance, 1.0);
  return Math.min(base + parallelIndex * PARALLEL_LIFT, CIRCUIT_BAND_HEADROOM);
}

export function getCircuitLayout(
  view: GraphView,
  circuitId: string,
  groups: VisualEdgeGroup[],
): CircuitLayout {
  const anchors = new Map<string, Anchor>();
  const add = (anchor: Anchor) => {
    if (!anchors.has(anchor.key)) {
      anchors.set(anchor.key, anchor);
    }
    return anchor.key;
  };
  // Every role of the circuit gets a hub, even one with no relations, so the groups are visible.
  for (const role of view.circuitById.get(circuitId)?.roles ?? []) {
    add(roleAnchor(view, circuitId, role.id));
  }

  const pairCount = new Map<string, number>();
  const edges: EdgeGroupGeometry[] = [];
  for (const group of groups) {
    if (group.circuitId !== circuitId) {
      continue;
    }
    const target = endpointAnchor(view, group.target);
    if (!target) {
      continue; // head receivers have no hub; not used by any current circuit
    }
    const from = add(sourceAnchor(view, group));
    const to = add(target);
    const pair = `${from}→${to}`;
    const index = pairCount.get(pair) ?? 0;
    pairCount.set(pair, index + 1);

    const a = anchors.get(from)!.position;
    const b = anchors.get(to)!.position;
    const mid = mean([a, b]);
    const control: Vec3 = [mid[0], mid[1] + 2 * lift(a, b, index), mid[2]];
    const points = Array.from({ length: CURVE_SAMPLES + 1 }, (_, i) =>
      quadratic(a, control, b, i / CURVE_SAMPLES),
    );
    const tipT = 1 - ARROW_GAP / Math.max(Math.hypot(b[0] - a[0], b[2] - a[2]), ARROW_GAP * 2);
    const tip = quadratic(a, control, b, tipT);
    const before = quadratic(a, control, b, Math.max(tipT - 0.02, 0));
    edges.push({
      groupId: group.id,
      from,
      to,
      points,
      arrow: { position: tip, direction: [tip[0] - before[0], tip[1] - before[1], tip[2] - before[2]] },
      mid: quadratic(a, control, b, 0.5),
    });
  }
  return { anchors, edges };
}

/** Anchor keys of a group's two ends (for highlighting hubs on hover). */
export function groupAnchorKeys(layout: CircuitLayout, groupId: string): Set<string> {
  const geometry = layout.edges.find((e) => e.groupId === groupId);
  return new Set(geometry ? [geometry.from, geometry.to] : []);
}

