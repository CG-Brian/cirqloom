import { describe, expect, it } from "vitest";
import type { Graph, Manifest } from "../data/schema";
import { buildGraphView } from "../features/experiment/graphView";
import {
  EMPHASIS,
  getBoundsForSelection,
  getCircuitMembers,
  getCircuitOverlay,
  getLayerFrameStyle,
  getNodeDisplayStyle,
  getRoleLegend,
  PALETTE,
  ROLE_STYLES,
} from "../features/overlays/selectors";

// A tiny 2-layer x 2-head model. L0H0 belongs to circuit A twice (roles r2 then r1, listed
// out of declared order on purpose) and also to circuit B. L1H1 belongs to A only.
const id = (layer: number, head: number) => `model:toy:head:L${layer}H${head}`;

const manifest: Manifest = {
  schema_version: "1.1.0",
  bundle_id: "toy",
  synthetic_fixture: true,
  display_warning: "toy",
  model: { id: "toy", hf_id: "toy", revision: null, layers: 2, heads_per_layer: 2, tokenizer_id: "toy" },
  experiments: [],
  provenance: { generator: "test", generator_version: "0", notes: "", environment: {} },
  graph_uri: "graph.json",
};

const membership = (circuit_id: string, node_id: string, role: string) => ({
  circuit_id,
  node_id,
  role,
  source_reference: "test",
  evidence_status: "experimental" as const,
});

const graph: Graph = {
  schema_version: "1.1.0",
  bundle_id: "toy",
  nodes: [0, 1].flatMap((layer) =>
    [0, 1].map((head) => ({ id: id(layer, head), type: "attention_head" as const, layer, head })),
  ),
  observations: [],
  circuits: [
    {
      id: "A",
      name: "Circuit A",
      origin: "synthetic_fixture",
      description: "",
      citation: null,
      roles: [
        { id: "r1", label: "Role 1", description: "" },
        { id: "r2", label: "Role 2", description: "" },
        { id: "r3", label: "Role 3", description: "" },
      ],
    },
    {
      id: "B",
      name: "Circuit B",
      origin: "synthetic_fixture",
      description: "",
      citation: null,
      roles: [{ id: "q1", label: "Q1", description: "" }],
    },
  ],
  memberships: [
    membership("A", id(0, 0), "r2"),
    membership("A", id(0, 0), "r1"),
    membership("B", id(0, 0), "q1"),
    membership("A", id(1, 1), "r2"),
  ],
  edges: [],
};

const view = buildGraphView(manifest, graph);
const overlayA = getCircuitOverlay(view, "A");
const overlayB = getCircuitOverlay(view, "B");

const style = (nodeId: string, overlay = overlayA, isSelected = false, isHovered = false) =>
  getNodeDisplayStyle({ nodeId, overlay, isSelected, isHovered });

describe("circuit overlay", () => {
  it("is null when no circuit or an unknown circuit is active", () => {
    expect(getCircuitOverlay(view, null)).toBeNull();
    expect(getCircuitOverlay(view, "missing")).toBeNull();
  });

  it("keeps a head in several circuits (many-to-many)", () => {
    expect(view.membershipsByNode.get(id(0, 0))).toHaveLength(3);
    expect(getCircuitMembers(view, "A")).toEqual(new Set([id(0, 0), id(1, 1)]));
    expect(getCircuitMembers(view, "B")).toEqual(new Set([id(0, 0)]));
  });

  it("orders a head's roles by the circuit's declared role order", () => {
    expect(overlayA?.rolesByNode.get(id(0, 0))).toEqual(["r1", "r2"]);
  });

  it("records the layers the circuit spans", () => {
    expect(overlayA?.layers).toEqual(new Set([0, 1]));
    expect(overlayB?.layers).toEqual(new Set([0]));
  });

  it("frames member heads only", () => {
    const members = getBoundsForSelection(view, [id(0, 0), id(1, 1)]);
    expect(overlayA?.bounds).toEqual(members);
    expect(getBoundsForSelection(view, ["missing"])).toBeNull();
  });

  it("builds a legend in role order, including roles with no members", () => {
    const legend = getRoleLegend(overlayA!);
    expect(legend.map((e) => [e.roleId, e.memberCount])).toEqual([
      ["r1", 1],
      ["r2", 2],
      ["r3", 0],
    ]);
  });
});

describe("node display style", () => {
  it("never changes the base colour", () => {
    for (const nodeId of view.nodes.map((n) => n.id)) {
      for (const overlay of [null, overlayA, overlayB]) {
        for (const isSelected of [false, true]) {
          for (const isHovered of [false, true]) {
            expect(style(nodeId, overlay, isSelected, isHovered).color).toBe(PALETTE.neutralNode);
          }
        }
      }
    }
  });

  it("shows no ring, halo or dimming when the overlay is off", () => {
    const s = style(id(0, 0), null);
    expect(s.roleRing).toBeNull();
    expect(s.opacity).toBe(1);
  });

  it("gives members a subtle ring in their first role's style", () => {
    const ring = style(id(0, 0)).roleRing;
    expect(ring).toMatchObject({ ...ROLE_STYLES[0], extraRoles: 1 });
    expect(ring!.ringOpacity).toBeLessThan(1);
    expect(ring!.haloOpacity).toBeLessThanOrEqual(0.2);
  });

  it("uses the role style of the circuit that is active", () => {
    expect(style(id(1, 1), overlayA).roleRing?.color).toBe(ROLE_STYLES[1]?.color);
    expect(style(id(1, 1), overlayB).roleRing).toBeNull();
  });

  it("subdues non-members while a circuit is shown", () => {
    expect(style(id(0, 1)).opacity).toBe(EMPHASIS.subduedNodeOpacity);
    expect(style(id(0, 1)).opacity).toBeGreaterThan(0);
  });

  it("adds a neutral selection ring on top of the role ring", () => {
    const s = style(id(0, 0), overlayA, true);
    expect(s.selectionRing?.color).toBe(PALETTE.selectionRing);
    expect(s.roleRing).not.toBeNull();
    expect(style(id(1, 1), overlayB, true).selectionRing?.color).toBe(PALETTE.selectionRing);
  });

  it("does not subdue a selected or hovered non-member", () => {
    expect(style(id(0, 1), overlayA, true).opacity).toBe(1);
    expect(style(id(0, 1), overlayA, false, true).opacity).toBe(1);
  });

  it("scales hover above selection, and both only slightly", () => {
    const hovered = style(id(0, 1), null, false, true).scale;
    const selected = style(id(0, 1), null, true).scale;
    expect(hovered).toBeGreaterThan(selected);
    expect(selected).toBeGreaterThan(1);
    expect(hovered).toBeLessThan(1.25);
  });
});

describe("encoding and layer frames", () => {
  it("pairs every role colour with a unique pattern + marker", () => {
    expect(new Set(ROLE_STYLES.map((s) => s.color)).size).toBe(ROLE_STYLES.length);
    expect(new Set(ROLE_STYLES.map((s) => `${s.pattern}/${s.marker}`)).size).toBe(ROLE_STYLES.length);
  });

  it("accents layers the circuit spans and lets others recede", () => {
    const none = getLayerFrameStyle(0, null);
    const spanned = getLayerFrameStyle(0, overlayB);
    const other = getLayerFrameStyle(1, overlayB);
    expect(spanned.edgeColor).toBe(PALETTE.circuitOutline);
    expect(other.edgeOpacity).toBeLessThan(none.edgeOpacity);
    expect(other.fillOpacity).toBeLessThan(spanned.fillOpacity);
  });

  it("never fills a layer with the accent colour", () => {
    for (const overlay of [null, overlayA, overlayB]) {
      for (const layer of [0, 1]) {
        expect(getLayerFrameStyle(layer, overlay).fillColor).toBe(PALETTE.layerFrame);
      }
    }
  });
});
