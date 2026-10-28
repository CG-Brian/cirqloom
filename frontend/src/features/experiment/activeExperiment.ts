// Which experiment the UI shows. Pure, so the default rule is testable and shared.
import type { Experiment, Manifest, Observation } from "../../data/schema";

/**
 * The user's pick if it exists in this bundle; otherwise the first measurement experiment,
 * otherwise the first experiment; null when the bundle has none.
 */
export function getActiveExperiment(manifest: Manifest, experimentId: string | null): Experiment | null {
  const { experiments } = manifest;
  return (
    experiments.find((e) => e.id === experimentId) ??
    experiments.find((e) => e.kind === "measurement") ??
    experiments[0] ??
    null
  );
}

/** Observations from one experiment only: scores from different suites are not comparable. */
export function observationsForExperiment(
  observations: Observation[],
  experiment: Experiment | null,
): Observation[] {
  return experiment ? observations.filter((o) => o.experiment_id === experiment.id) : [];
}
