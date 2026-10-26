// Minimal shell so the Phase 5 scene can be run. Phase 6 replaces the temporary header with
// TopBar / CircuitSidebar / HeadInspector / StatusBar and adds Escape-to-deselect.
import { StaticBundleProvider } from "../data/StaticBundleProvider";
import { useBundle } from "../features/experiment/useBundle";
import { useExplorerStore } from "../features/selection/store";
import { ModelScene } from "../scene/ModelScene";

// Created once at module scope: useBundle needs a stable provider.
const provider = new StaticBundleProvider({ "gpt2-small": "/fixtures/gpt2-small-demo" });

export function App() {
  const bundle = useBundle(provider, "gpt2-small");
  const activeCircuitId = useExplorerStore((s) => s.activeCircuitId);
  const toggleCircuit = useExplorerStore((s) => s.toggleCircuit);
  const resetCamera = useExplorerStore((s) => s.resetCamera);

  if (bundle.status === "loading") {
    return <p className="p-4 text-slate-400">Loading bundle…</p>;
  }
  if (bundle.status === "error") {
    return (
      <pre role="alert" className="p-4 whitespace-pre-wrap text-red-300">
        {bundle.error.message}
      </pre>
    );
  }

  const { manifest, view } = bundle;
  return (
    <div className="flex h-full flex-col">
      <header className="flex flex-wrap items-center gap-4 border-b border-slate-800 px-4 py-2 text-sm">
        <span className="font-semibold">Cirqloom</span>
        <span className="text-slate-400">{manifest.model.id}</span>
        {view.circuits.map((c) => (
          <label key={c.id} className="flex items-center gap-1.5">
            <input
              type="checkbox"
              checked={activeCircuitId === c.id}
              onChange={() => toggleCircuit(c.id)}
            />
            {c.name}
          </label>
        ))}
        <button
          type="button"
          onClick={resetCamera}
          className="rounded border border-slate-700 px-2 py-0.5 hover:bg-slate-800"
        >
          Reset view
        </button>
        {manifest.synthetic_fixture && (
          <span className="rounded bg-amber-500/15 px-2 py-0.5 text-amber-300">
            {manifest.display_warning}
          </span>
        )}
      </header>
      <main className="min-h-0 flex-1">
        <ModelScene view={view} />
      </main>
    </div>
  );
}
