// First-time visitor guide, shown in the inspector while nothing is selected. Three short steps
// through the app's central interaction, plus what the two most common evidence badges mean.
import type { Manifest } from "../data/schema";

interface VisitorGuideProps {
  manifest: Manifest;
  /** Name of a circuit with relations in this bundle, if any (e.g. the IOI circuit). */
  relationCircuitName: string | null;
}

export function VisitorGuide({ manifest, relationCircuitName }: VisitorGuideProps) {
  const hasExperiments = manifest.experiments.length > 0;
  return (
    <section aria-labelledby="visitor-guide-title" className="mt-3 rounded border border-slate-800 px-3 py-2.5">
      <h3 id="visitor-guide-title" className="text-xs font-semibold text-slate-200">
        New here? Three things to try
      </h3>
      <ol className="mt-2 list-decimal space-y-2 pl-4 text-[11px] leading-snug text-slate-300">
        <li>
          <span className="text-slate-100">Explore an attention head.</span> Click a head in the 3D view or
          in the Heads list (arrow keys work there).
          {hasExperiments
            ? " This panel then shows its measured scores and an attention heatmap: rows are destination tokens, columns are source tokens. “Show cells” outlines the cells a score averaged."
            : " This panel then shows its identity and circuit membership."}
        </li>
        {hasExperiments && (
          <li>
            <span className="text-slate-100">Pick an experiment.</span> The Experiment menu at the top switches
            prompt suites: repeated random tokens, IOI sentences, and a paired control for each. Compare a
            head on a measurement run and on its control.
          </li>
        )}
        <li>
          <span className="text-slate-100">Inspect a circuit relation.</span>{" "}
          {relationCircuitName
            ? `Turn on ${relationCircuitName} in the left panel, then open a relation from its list or click a ribbon. The relation inspector keeps the paper's claim apart from attention measured here.`
            : "Turn on a circuit in the left panel to see its role groups and member heads."}
        </li>
      </ol>
      <p className="mt-2 text-[10px] leading-snug text-slate-500">
        Badges: <span className="text-slate-300">Published</span> comes from a cited paper;{" "}
        <span className="text-slate-300">Experimental</span> is measured here with a heuristic and is not causal
        evidence.
      </p>
    </section>
  );
}
