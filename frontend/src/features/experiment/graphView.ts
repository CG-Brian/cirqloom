// Turns raw bundle JSON into indexed, render-ready data. Scene components read only this.
import type { Circuit, Graph, Manifest, Membership, Observation } from "../../data/schema";
import { boundsOf, headPosition, type Bounds, type Vec3 } from "../../scene/layout";

export interface HeadView {
  id: string;
  layer: number;
  head: number;
  label: string;
  position: Vec3;
}

export interface GraphView {
  modelId: string;
  layers: number;
  headsPerLayer: number;
  nodes: HeadView[];
  nodeById: Map<string, HeadView>;
  nodesByLayer: HeadView[][];
  circuits: Circuit[];
  circuitById: Map<string, Circuit>;
  /** Many-to-many: one head can belong to several circuits/roles. */
  membershipsByNode: Map<string, Membership[]>;
  /** Detector results per head. A head with no entry has no measurements in this bundle. */
  observationsByNode: Map<string, Observation[]>;
  bounds: Bounds;
}

export function headLabel(layer: number, head: number): string {
  return `L${layer}H${head}`;
}

export function buildGraphView(manifest: Manifest, graph: Graph): GraphView {
  const { layers, heads_per_layer: headsPerLayer } = manifest.model;

  const nodes: HeadView[] = graph.nodes
    .map((n) => ({
      id: n.id,
      layer: n.layer,
      head: n.head,
      label: headLabel(n.layer, n.head),
      position: headPosition(n.layer, n.head, layers, headsPerLayer),
    }))
    .sort((a, b) => a.layer - b.layer || a.head - b.head);

  const nodesByLayer: HeadView[][] = Array.from({ length: layers }, () => []);
  for (const node of nodes) {
    nodesByLayer[node.layer]?.push(node);
  }

  const membershipsByNode = groupByNode(graph.memberships);
  const observationsByNode = groupByNode(graph.observations);

  return {
    modelId: manifest.model.id,
    layers,
    headsPerLayer,
    nodes,
    nodeById: new Map(nodes.map((n) => [n.id, n])),
    nodesByLayer,
    circuits: graph.circuits,
    circuitById: new Map(graph.circuits.map((c) => [c.id, c])),
    membershipsByNode,
    observationsByNode,
    bounds: boundsOf(nodes.map((n) => n.position)),
  };
}

function groupByNode<T extends { node_id: string }>(items: T[]): Map<string, T[]> {
  const groups = new Map<string, T[]>();
  for (const item of items) {
    const list = groups.get(item.node_id);
    if (list) {
      list.push(item);
    } else {
      groups.set(item.node_id, [item]);
    }
  }
  return groups;
}
