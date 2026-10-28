import { act, renderHook } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { DataProvider, HeadAttention } from "../data/DataProvider";
import { useHeadAttention } from "../features/attention/useHeadAttention";

function deferredProvider() {
  const pending = new Map<string, (a: HeadAttention) => void>();
  const provider = {
    loadHeadAttention: (_experimentId: string, nodeId: string) =>
      new Promise<HeadAttention>((resolve) => pending.set(nodeId, resolve)),
  } as unknown as DataProvider;
  const resolve = async (nodeId: string) => {
    await act(async () => {
      pending.get(nodeId)!({ experimentId: "e", nodeId, promptId: "p", tokens: ["x"], weights: [[1]] });
    });
  };
  return { provider, resolve };
}

describe("useHeadAttention", () => {
  it("is idle without an experiment or a head", () => {
    const { provider } = deferredProvider();
    const { result } = renderHook(() => useHeadAttention(provider, null, "n"));
    expect(result.current.status).toBe("idle");
  });

  it("drops a slow answer for a head that is no longer selected", async () => {
    const { provider, resolve } = deferredProvider();
    const { result, rerender } = renderHook(({ node }) => useHeadAttention(provider, "e", node), {
      initialProps: { node: "L5H5" },
    });
    rerender({ node: "L6H9" });
    await resolve("L5H5"); // arrives late, after the selection moved on
    expect(result.current.status).toBe("loading");
    await resolve("L6H9");
    expect(result.current).toMatchObject({ status: "ready", attention: { nodeId: "L6H9" } });
  });
});
