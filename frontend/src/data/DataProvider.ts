import type { Experiment, Graph, Manifest } from "./schema";

export interface ModelSummary {
  id: string;
}

/** Token-to-token attention inside ONE head. Not a connection between heads. */
export interface HeadAttention {
  experimentId: string;
  nodeId: string;
  /** The one example prompt whose attention is stored for this experiment. */
  promptId: string;
  /** Display string per position, leading spaces kept. */
  tokens: string[];
  /** weights[destination][source], decoded from float16 (about 3 significant digits). */
  weights: number[][];
}

/** The only way UI code gets data. Swap implementations without touching components. */
export interface DataProvider {
  listModels(): Promise<ModelSummary[]>;
  listExperiments(modelId: string): Promise<Experiment[]>;
  loadManifest(modelId: string): Promise<Manifest>;
  loadGraph(modelId: string, experimentId: string | null): Promise<Graph>;
  loadHeadAttention(experimentId: string, nodeId: string): Promise<HeadAttention>;
}

export class BundleLoadError extends Error {
  readonly url: string;
  readonly status: number | null;

  constructor(url: string, status: number | null, detail: string) {
    super(`Failed to load ${url}: ${detail}`);
    this.name = "BundleLoadError";
    this.url = url;
    this.status = status;
  }
}

export class BundleSchemaError extends Error {
  readonly url: string;
  readonly problems: string[];

  constructor(url: string, problems: string[]) {
    super(`${url} does not match the bundle schema:\n- ${problems.join("\n- ")}`);
    this.name = "BundleSchemaError";
    this.url = url;
    this.problems = problems;
  }
}

export class AttentionUnavailableError extends Error {
  constructor(experimentId: string, nodeId: string) {
    super(`No attention data for ${nodeId} in experiment ${experimentId}`);
    this.name = "AttentionUnavailableError";
  }
}
