// App shell (AGENTS.md §9 screen anatomy): top bar, circuit sidebar, 3D scene, head inspector,
// status bar. Domain state (selection, active circuit) lives in the store; panel collapse is
// local layout state.
import { useMemo, useState } from "react";
import { StaticBundleProvider } from "../data/StaticBundleProvider";
import type { Manifest } from "../data/schema";
import { getActiveExperiment } from "../features/experiment/activeExperiment";
import type { GraphView } from "../features/experiment/graphView";
import { useBundle } from "../features/experiment/useBundle";
import { getCircuitOverlay } from "../features/overlays/selectors";
import { useExplorerStore } from "../features/selection/store";
import { useEscapeToDeselect } from "../features/selection/useEscapeToDeselect";
import { CircuitSidebar } from "../panels/CircuitSidebar";
import { HeadInspector } from "../panels/HeadInspector";
import { ModelScene } from "../scene/ModelScene";
import { StatusBar } from "./StatusBar";
import { TopBar } from "./TopBar";

// Served from ../data (vite publicDir). The measured bundle is the default; ?bundle=fixture
// opens the synthetic demo fixture, which the browser smoke tests use for its circuit.
const BUNDLE_PATHS = {
  measured: "/bundles/gpt2-small-repeated-v1",
  fixture: "/fixtures/gpt2-small-demo",
} as const;

function bundleFromUrl(): keyof typeof BUNDLE_PATHS {
  return new URLSearchParams(window.location.search).get("bundle") === "fixture" ? "fixture" : "measured";
}

// Created once at module scope: useBundle needs a stable provider.
export const provider = new StaticBundleProvider({ "gpt2-small": BUNDLE_PATHS[bundleFromUrl()] });

export function App() {
  const bundle = useBundle(provider, "gpt2-small");
  useEscapeToDeselect();

  if (bundle.status === "loading") {
    return <p className="p-4 text-sm text-slate-400">Loading bundle…</p>;
  }
  if (bundle.status === "error") {
    return (
      <div role="alert" className="p-4">
        <h1 className="text-sm font-semibold text-red-300">Could not load the bundle</h1>
        <pre className="mt-2 text-xs whitespace-pre-wrap text-red-200/80">{bundle.error.message}</pre>
      </div>
    );
  }
  return <Explorer manifest={bundle.manifest} view={bundle.view} />;
}

function Explorer({ manifest, view }: { manifest: Manifest; view: GraphView }) {
  const activeCircuitId = useExplorerStore((s) => s.activeCircuitId);
  const experimentId = useExplorerStore((s) => s.experimentId);
  const experiment = getActiveExperiment(manifest, experimentId);
  const overlay = useMemo(() => getCircuitOverlay(view, activeCircuitId), [view, activeCircuitId]);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [inspectorCollapsed, setInspectorCollapsed] = useState(false);

  return (
    <div className="flex h-full flex-col">
      <TopBar manifest={manifest} experiment={experiment} />
      <div className="flex min-h-0 flex-1 flex-col overflow-y-auto lg:flex-row lg:overflow-hidden">
        <main className="relative order-1 h-[55vh] shrink-0 lg:order-2 lg:h-auto lg:min-w-0 lg:flex-1">
          <ModelScene view={view} overlay={overlay} />
        </main>
        <div className="order-2 lg:order-1 lg:flex">
          <CircuitSidebar
            view={view}
            overlay={overlay}
            collapsed={sidebarCollapsed}
            onToggleCollapsed={() => setSidebarCollapsed((c) => !c)}
          />
        </div>
        <div className="order-3 lg:flex">
          <HeadInspector
            provider={provider}
            view={view}
            manifest={manifest}
            experiment={experiment}
            collapsed={inspectorCollapsed}
            onToggleCollapsed={() => setInspectorCollapsed((c) => !c)}
          />
        </div>
      </div>
      <StatusBar manifest={manifest} overlay={overlay} />
    </div>
  );
}
