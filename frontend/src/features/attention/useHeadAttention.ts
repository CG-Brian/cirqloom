import { useEffect, useState } from "react";
import type { DataProvider, HeadAttention } from "../../data/DataProvider";

export type HeadAttentionState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "ready"; attention: HeadAttention }
  | { status: "error"; error: Error };

type Settled = Extract<HeadAttentionState, { status: "ready" | "error" }>;

/**
 * Loads one head's attention for one experiment. A result that arrives after the selection
 * moved on is dropped, so a slow L5H5 load can never overwrite L6H9. No automatic retry: a
 * static asset that fails once (missing file, wrong size) fails the same way again.
 */
export function useHeadAttention(
  provider: DataProvider,
  experimentId: string | null,
  nodeId: string | null,
): HeadAttentionState {
  const key = experimentId !== null && nodeId !== null ? `${experimentId}|${nodeId}` : null;
  const [settled, setSettled] = useState<{ key: string; state: Settled } | null>(null);

  useEffect(() => {
    if (experimentId === null || nodeId === null) {
      return;
    }
    const requestKey = `${experimentId}|${nodeId}`;
    let ignore = false;
    provider
      .loadHeadAttention(experimentId, nodeId)
      .then((attention) => {
        if (!ignore) {
          setSettled({ key: requestKey, state: { status: "ready", attention } });
        }
      })
      .catch((err: unknown) => {
        if (!ignore) {
          const error = err instanceof Error ? err : new Error(String(err));
          setSettled({ key: requestKey, state: { status: "error", error } });
        }
      });
    return () => {
      ignore = true;
    };
  }, [provider, experimentId, nodeId]);

  if (key === null) {
    return { status: "idle" };
  }
  // A settled result for another head or experiment is stale: keep showing loading.
  return settled !== null && settled.key === key ? settled.state : { status: "loading" };
}
