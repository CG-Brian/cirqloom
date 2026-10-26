import type { Manifest } from "../data/schema";
import { useExplorerStore } from "../features/selection/store";

export function TopBar({ manifest }: { manifest: Manifest }) {
  const resetCamera = useExplorerStore((s) => s.resetCamera);
  const experiments = manifest.experiments;

  return (
    <header className="flex flex-wrap items-center gap-x-4 gap-y-2 border-b border-slate-800 bg-slate-950 px-4 py-2">
      <h1 className="text-sm font-semibold tracking-tight text-slate-50">Cirqloom</h1>
      <span className="rounded bg-slate-800 px-2 py-0.5 font-mono text-[11px] text-slate-300">
        {manifest.model.id}
      </span>
      <label className="flex items-center gap-2 text-xs text-slate-400">
        Experiment
        <select
          disabled={experiments.length === 0}
          className="rounded border border-slate-700 bg-slate-900 px-2 py-1 text-xs text-slate-200 disabled:text-slate-500"
        >
          {experiments.length === 0 ? (
            <option>No experiments in bundle</option>
          ) : (
            experiments.map((e) => (
              <option key={e.id} value={e.id}>
                {e.id}
              </option>
            ))
          )}
        </select>
      </label>
      <div className="ml-auto">
        <button
          type="button"
          onClick={resetCamera}
          className="rounded border border-slate-700 px-2.5 py-1 text-xs text-slate-200 hover:bg-slate-800 focus-visible:outline focus-visible:outline-2 focus-visible:outline-sky-400"
        >
          Reset view
        </button>
      </div>
    </header>
  );
}
