import { useEffect, useState } from "react";
import type { DataProvider } from "../../data/DataProvider";
import type { Manifest } from "../../data/schema";
import { buildGraphView, type GraphView } from "./graphView";

export type BundleState =
  | { status: "loading" }
  | { status: "ready"; manifest: Manifest; view: GraphView }
  | { status: "error"; error: Error };

type Settled = Exclude<BundleState, { status: "loading" }>;

/** `provider` must be a stable object (create it once at module scope). */
export function useBundle(provider: DataProvider, modelId: string): BundleState {
  const [settled, setSettled] = useState<{ modelId: string; state: Settled } | null>(null);

  useEffect(() => {
    let ignore = false;
    Promise.all([provider.loadManifest(modelId), provider.loadGraph(modelId, null)])
      .then(([manifest, graph]) => {
        if (!ignore) {
          const view = buildGraphView(manifest, graph);
          setSettled({ modelId, state: { status: "ready", manifest, view } });
        }
      })
      .catch((err: unknown) => {
        if (!ignore) {
          const error = err instanceof Error ? err : new Error(String(err));
          setSettled({ modelId, state: { status: "error", error } });
        }
      });
    return () => {
      ignore = true;
    };
  }, [provider, modelId]);

  // A result for a different model is stale: show loading until the new one arrives.
  if (settled === null || settled.modelId !== modelId) {
    return { status: "loading" };
  }
  return settled.state;
}
