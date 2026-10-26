import type { Manifest } from "../data/schema";
import type { CircuitOverlay } from "../features/overlays/selectors";

interface StatusBarProps {
  manifest: Manifest;
  overlay: CircuitOverlay | null;
}

export function StatusBar({ manifest, overlay }: StatusBarProps) {
  return (
    <footer className="flex flex-wrap items-center gap-x-4 gap-y-1 border-t border-slate-800 bg-slate-950 px-4 py-1.5 text-[11px] text-slate-400">
      {manifest.synthetic_fixture && (
        <span role="status" className="rounded bg-amber-500/15 px-2 py-0.5 font-medium text-amber-300">
          {manifest.display_warning}
        </span>
      )}
      <span>
        Overlay: <span className="text-slate-200">{overlay ? overlay.circuit.name : "none"}</span>
      </span>
      <span className="flex items-center gap-1.5">
        <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">
          <circle cx="6" cy="6" r="4.5" fill="none" stroke="#f8fafc" strokeWidth="1.5" />
        </svg>
        selected head
      </span>
      <span className="ml-auto font-mono text-slate-500">
        {manifest.bundle_id} · schema {manifest.schema_version}
      </span>
    </footer>
  );
}
