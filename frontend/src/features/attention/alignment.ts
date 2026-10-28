// Observed attention alignment for one circuit relation (AGENTS.md §2, §9).
//
// On one prompt with named positions (IOI: IO, S1, S1+1, S2, END), each head of a relation gets
// one row: its attention from the query position the relation names for its side (the sender's
// position, or the receiver's) to every named position. Rows are independent observations of
// separate heads. Lining them up does not show information passing from one head to another.
import type { HeadAttention } from "../../data/DataProvider";
import type { VisualEdgeGroup } from "../circuits/edgeGroups";
import { namedPositions } from "./heatmapModel";

export interface AlignmentCell {
  label: string;
  index: number;
  /** null when the position is after the query (not attendable), never 0. */
  weight: number | null;
}

export interface AlignmentRow {
  nodeId: string;
  side: "sender" | "receiver";
  queryLabel: string;
  queryIndex: number;
  cells: AlignmentCell[];
}

export interface AlignmentPlan {
  /** Heads to load, sender heads first. */
  senders: string[];
  receivers: string[];
  senderQuery: string | null;
  receiverQuery: string | null;
}

/** Which heads to show and at which named query position, from the relation's evidence. */
export function alignmentPlan(
  group: VisualEdgeGroup,
  heads: { senders: string[]; receivers: string[] },
): AlignmentPlan {
  const senderQuery = group.positions?.source ?? null;
  // A relation into the logits has no receiving head, so no receiver rows.
  const receiverQuery = group.target.kind === "special" ? null : (group.positions?.target ?? null);
  return {
    senders: senderQuery ? heads.senders : [],
    receivers: receiverQuery ? heads.receivers : [],
    senderQuery,
    receiverQuery,
  };
}

function row(
  nodeId: string,
  side: AlignmentRow["side"],
  queryLabel: string,
  attention: HeadAttention,
): AlignmentRow | null {
  const queryIndex = attention.positions?.[queryLabel];
  if (queryIndex === undefined) {
    return null;
  }
  return {
    nodeId,
    side,
    queryLabel,
    queryIndex,
    cells: namedPositions(attention.positions).map(([label, index]) => ({
      label,
      index,
      weight: index > queryIndex ? null : (attention.weights[queryIndex]?.[index] ?? null),
    })),
  };
}

/** Rows for every loaded head of the plan; heads without the query position are left out. */
export function alignmentRows(plan: AlignmentPlan, attentions: Map<string, HeadAttention>): AlignmentRow[] {
  const rows: AlignmentRow[] = [];
  for (const [side, ids, query] of [
    ["sender", plan.senders, plan.senderQuery],
    ["receiver", plan.receivers, plan.receiverQuery],
  ] as const) {
    if (!query) {
      continue;
    }
    for (const id of ids) {
      const attention = attentions.get(id);
      const r = attention ? row(id, side, query, attention) : null;
      if (r) {
        rows.push(r);
      }
    }
  }
  return rows;
}
