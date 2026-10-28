// Evidence → visual groups on the real GPT-2 bundle: grouping must not lose, duplicate or
// change any of the 27 published IOI records, and must not mix records that mean different things.
import { describe, expect, it } from "vitest";
import {
  edgeGroupHeads,
  edgeGroupKey,
  edgeGroupLabel,
  edgeGroupsForHead,
  getEdgeDisplayStyle,
  getVisualEdgeGroups,
} from "../features/circuits/edgeGroups";
import { getCircuitLayout, logitsPosition } from "../features/circuits/circuitLayout";
import { circuitBandY } from "../scene/layout";
import { loadMeasured } from "./fixtureView";

const IOI = "circuit:gpt2-small:ioi:wang-2022";
const { graph, view } = loadMeasured();
const groups = getVisualEdgeGroups(view, IOI);
const head = (layer: number, h: number) => `model:gpt2-small:head:L${layer}H${h}`;

describe("getVisualEdgeGroups", () => {
  it("keeps every one of the 27 records exactly once, unmodified", () => {
    const flat = groups.flatMap((g) => g.edges);
    expect(flat).toHaveLength(27);
    expect(new Set(flat.map((e) => e.id)).size).toBe(27);
    // Same objects as the bundle (no copies, no edits), in bundle order within each group.
    for (const edge of graph.edges) {
      expect(flat).toContain(edge);
    }
  });

  it("bundles the 27 records into the 8 relations of the source record", () => {
    const summary = groups.map((g) => [edgeGroupLabel(view, g), g.edges.length]);
    expect(summary).toEqual([
      ["Name Mover heads → logits", 3],
      ["Negative Name Mover heads → logits", 2],
      ["S-Inhibition heads → Name Mover queries", 4],
      ["S-Inhibition group → Name Mover queries", 1],
      ["Duplicate Token + Induction heads → S-Inhibition values", 4],
      ["Duplicate Token + Induction heads → S-Inhibition keys", 3],
      ["Previous Token heads → Induction keys", 2],
      ["Backup Name Mover heads → logits", 8],
    ]);
  });

  it("only groups records that agree on every field that carries meaning", () => {
    for (const g of groups) {
      const keys = new Set(g.edges.map(edgeGroupKey));
      expect(keys.size).toBe(1);
      for (const e of g.edges) {
        expect(e.channel).toBe(g.channel);
        expect(e.locator).toBe(g.locator);
        expect(e.conditions).toEqual(g.conditions);
        expect(e.effect).toEqual(g.effect);
        expect(e.source.kind).toBe(g.sourceKind);
      }
    }
  });

  it("never merges the per-head experiment (4b) with the group experiment (4c)", () => {
    const toNameMoverQ = groups.filter((g) => g.channel === "q");
    expect(toNameMoverQ.map((g) => [g.sourceKind, g.locator])).toEqual([
      ["head", "§3.2, Figure 4b (p. 7)"],
      ["role_group", "§3.2, Figure 4c (p. 7)"],
    ]);
  });

  it("returns nothing without a circuit, and nothing for a circuit without relations", () => {
    expect(getVisualEdgeGroups(view, null)).toEqual([]);
    expect(getVisualEdgeGroups(view, "circuit:none")).toEqual([]);
  });
});

describe("heads a relation is about (for emphasis only)", () => {
  it("are the sender heads and the receiving group's members", () => {
    const fuzzy = groups.find((g) => g.channel === "k" && g.target.kind === "role_group" && g.target.role === "s_inhibition")!;
    const { senders, receivers } = edgeGroupHeads(view, fuzzy);
    expect(senders).toEqual([head(0, 10), head(5, 8), head(5, 9)]);
    expect(new Set(receivers)).toEqual(new Set([head(7, 3), head(7, 9), head(8, 6), head(8, 10)]));
  });

  it("finds a head's relations as sender or receiver member", () => {
    const labels = edgeGroupsForHead(view, groups, head(9, 9)).map((g) => edgeGroupLabel(view, g));
    expect(labels).toEqual([
      "Name Mover heads → logits",
      "S-Inhibition heads → Name Mover queries",
      "S-Inhibition group → Name Mover queries",
    ]);
  });
});

describe("edge display style", () => {
  it("encodes evidence kind and conditions by line pattern, not colour alone", () => {
    const backup = groups.find((g) => g.conditions.length > 0)!;
    const plain = groups[0]!;
    expect(getEdgeDisplayStyle(plain, "idle").pattern).toBe("solid");
    expect(getEdgeDisplayStyle(backup, "idle").pattern).toBe("dotted");
    expect(getEdgeDisplayStyle({ ...plain, evidenceKind: "published_schematic" }, "idle").pattern).toBe("dashed");
    expect(getEdgeDisplayStyle(plain, "dimmed").opacity).toBeLessThan(getEdgeDisplayStyle(plain, "idle").opacity);
  });
});

describe("getCircuitLayout", () => {
  const layout = getCircuitLayout(view, IOI, groups);

  it("has a hub for each of the 7 roles over its members, above the layers", () => {
    const hubs = [...layout.anchors.values()].filter((a) => a.kind === "role_group");
    expect(hubs).toHaveLength(7);
    for (const hub of hubs) {
      const xs = hub.nodeIds.map((id) => view.nodeById.get(id)!.position[0]);
      expect(hub.position[0]).toBeCloseTo(xs.reduce((a, b) => a + b, 0) / xs.length);
      expect(hub.position[1]).toBe(circuitBandY(view.headsPerLayer));
    }
  });

  it("anchors subsets of a role on their own heads, not on the whole role", () => {
    const subsets = [...layout.anchors.values()].filter((a) => a.kind === "subset");
    expect(subsets.map((a) => a.nodeIds)).toEqual([
      [head(0, 1), head(3, 0), head(5, 5), head(6, 9)],
      [head(0, 10), head(5, 8), head(5, 9)],
    ]);
  });

  it("draws one curve per relation, ending at logits or a role hub, never at a single head", () => {
    expect(layout.edges).toHaveLength(groups.length);
    for (const e of layout.edges) {
      expect(e.to.startsWith("role:") || e.to === "special:logits").toBe(true);
      // Curves stay above the layer frames.
      expect(Math.min(...e.points.map((p) => p[1]))).toBeGreaterThanOrEqual(circuitBandY(view.headsPerLayer) - 1e-9);
    }
    expect(layout.anchors.get("special:logits")!.position).toEqual(logitsPosition(view));
  });

  it("lifts relations between the same two anchors apart so they never overlap", () => {
    const [fig4b, fig4c] = layout.edges.filter((e) => e.from === `role:${IOI}:s_inhibition` && e.to === `role:${IOI}:name_mover`);
    const peak = (pts: [number, number, number][]) => Math.max(...pts.map((p) => p[1]));
    expect(peak(fig4c!.points)).toBeGreaterThan(peak(fig4b!.points) + 0.2);
  });

  it("is within the scene bounds the camera frames", () => {
    for (const e of layout.edges) {
      for (const p of e.points) {
        expect(p[1]).toBeLessThanOrEqual(view.bounds.max[1] + 1e-9);
        expect(p[0]).toBeLessThanOrEqual(view.bounds.max[0] + 1e-9);
      }
    }
  });
});
