// Small presentational pieces shared by the sidebar, inspector and status bar.
// Every badge carries a text label, so meaning never depends on colour alone.
import type { ReactNode } from "react";
import type { Circuit, EvidenceStatus } from "../data/schema";
import type { LinePattern, MarkerShape, RoleStyle } from "../features/overlays/selectors";

const EVIDENCE_LABEL: Record<EvidenceStatus, string> = {
  published: "Published",
  measured: "Measured",
  candidate: "Candidate",
  experimental: "Experimental",
  unavailable: "Unavailable",
};

const EVIDENCE_CLASS: Record<EvidenceStatus, string> = {
  published: "border-sky-400/50 text-sky-200",
  measured: "border-emerald-400/50 text-emerald-200",
  candidate: "border-teal-400/50 border-dashed text-teal-200",
  experimental: "border-amber-400/50 border-dashed text-amber-200",
  unavailable: "border-slate-500/60 border-dotted text-slate-400 italic",
};

export function EvidenceBadge({ status }: { status: EvidenceStatus }) {
  return (
    <span
      className={`inline-flex items-center rounded border px-1.5 py-px text-[10px] leading-4 font-medium uppercase tracking-wide ${EVIDENCE_CLASS[status]}`}
    >
      {EVIDENCE_LABEL[status]}
    </span>
  );
}

export function OriginBadge({ origin }: { origin: Circuit["origin"] }) {
  return origin === "published" ? (
    <span className="inline-flex rounded bg-sky-500/15 px-1.5 py-px text-[10px] leading-4 font-medium text-sky-200">
      Published annotation
    </span>
  ) : (
    <span className="inline-flex rounded bg-amber-500/15 px-1.5 py-px text-[10px] leading-4 font-medium text-amber-200">
      Synthetic fixture
    </span>
  );
}

const DASH: Record<LinePattern, string | undefined> = {
  solid: undefined,
  dashed: "4 3",
  dotted: "1 3",
  "dash-dot": "5 2 1 2",
};

function MarkerPath({ shape, color }: { shape: MarkerShape; color: string }) {
  const common = { fill: "none", stroke: color, strokeWidth: 1.5 };
  switch (shape) {
    case "circle":
      return <circle cx="6" cy="6" r="4" {...common} />;
    case "square":
      return <rect x="2.5" y="2.5" width="7" height="7" {...common} />;
    case "triangle":
      return <path d="M6 2 L10 10 L2 10 Z" {...common} />;
    case "diamond":
      return <path d="M6 1.5 L10.5 6 L6 10.5 L1.5 6 Z" {...common} />;
  }
}

/** Marker shape + line sample for a role, matching the ring drawn in the 3D scene. */
export function RoleMarker({ style }: { style: RoleStyle }) {
  return (
    <svg width="32" height="12" viewBox="0 0 32 12" aria-hidden="true" className="shrink-0">
      <MarkerPath shape={style.marker} color={style.color} />
      <line
        x1="14"
        y1="6"
        x2="31"
        y2="6"
        stroke={style.color}
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeDasharray={DASH[style.pattern]}
      />
    </svg>
  );
}

export function PanelSection({
  title,
  children,
  aside,
}: {
  title: string;
  children: ReactNode;
  aside?: ReactNode;
}) {
  return (
    <section className="border-t border-slate-800/80 px-3 py-3 first:border-t-0">
      <div className="mb-2 flex items-center justify-between gap-2">
        <h3 className="text-[11px] font-semibold tracking-wider text-slate-400 uppercase">{title}</h3>
        {aside}
      </div>
      {children}
    </section>
  );
}

/** Honest empty/unavailable state: visibly different from a measured value. */
export function EmptyNote({ children }: { children: ReactNode }) {
  return (
    <p className="rounded border border-dashed border-slate-700 px-2 py-1.5 text-xs text-slate-400">
      {children}
    </p>
  );
}

export function CollapseButton({
  label,
  expanded,
  onClick,
  direction,
}: {
  label: string;
  expanded: boolean;
  onClick: () => void;
  direction: "left" | "right";
}) {
  // Chevron points the way the panel will move.
  const pointsLeft = (direction === "left") === expanded;
  return (
    <button
      type="button"
      onClick={onClick}
      aria-expanded={expanded}
      aria-label={`${expanded ? "Collapse" : "Expand"} ${label}`}
      title={`${expanded ? "Collapse" : "Expand"} ${label}`}
      className="rounded p-1 text-slate-400 hover:bg-slate-800 hover:text-slate-100 focus-visible:outline focus-visible:outline-sky-400"
    >
      <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true">
        <path
          d={pointsLeft ? "M9 3 L5 7 L9 11" : "M5 3 L9 7 L5 11"}
          fill="none"
          stroke="currentColor"
          strokeWidth="1.6"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    </button>
  );
}
