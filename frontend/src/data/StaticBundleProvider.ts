import type { z } from "zod";
import {
  AttentionUnavailableError,
  BundleLoadError,
  BundleSchemaError,
  type DataProvider,
  type HeadAttention,
  type ModelSummary,
} from "./DataProvider";
import {
  crossCheckBundle,
  GraphSchema,
  ManifestSchema,
  type Experiment,
  type Graph,
  type Manifest,
} from "./schema";

async function fetchJson<T>(url: string, schema: z.ZodType<T>): Promise<T> {
  let res: Response;
  try {
    res = await fetch(url);
  } catch (err) {
    throw new BundleLoadError(url, null, String(err));
  }
  if (!res.ok) {
    throw new BundleLoadError(url, res.status, `HTTP ${res.status}`);
  }
  let body: unknown;
  try {
    body = await res.json();
  } catch {
    throw new BundleLoadError(url, res.status, "response is not valid JSON");
  }
  const parsed = schema.safeParse(body);
  if (!parsed.success) {
    const problems = parsed.error.issues.map(
      (issue) => `${issue.path.map(String).join(".") || "(root)"}: ${issue.message}`,
    );
    throw new BundleSchemaError(url, problems);
  }
  return parsed.data;
}

/** Reads precomputed bundles served as static files. */
export class StaticBundleProvider implements DataProvider {
  private readonly bundlePaths: Record<string, string>;
  private readonly manifests = new Map<string, Promise<Manifest>>();

  /** bundlePaths maps a model id to its bundle folder, e.g. { "gpt2-small": "/fixtures/gpt2-small-demo" }. */
  constructor(bundlePaths: Record<string, string>) {
    this.bundlePaths = bundlePaths;
  }

  async listModels(): Promise<ModelSummary[]> {
    return Object.keys(this.bundlePaths).map((id) => ({ id }));
  }

  async listExperiments(modelId: string): Promise<Experiment[]> {
    return (await this.loadManifest(modelId)).experiments;
  }

  loadManifest(modelId: string): Promise<Manifest> {
    let pending = this.manifests.get(modelId);
    if (!pending) {
      pending = fetchJson(`${this.basePath(modelId)}/manifest.json`, ManifestSchema);
      // Cache successes only, so a failed load can be retried.
      pending.catch(() => this.manifests.delete(modelId));
      this.manifests.set(modelId, pending);
    }
    return pending;
  }

  // M0 bundles have one graph per model, so the experiment id is not needed yet.
  async loadGraph(modelId: string): Promise<Graph> {
    const manifest = await this.loadManifest(modelId);
    const url = `${this.basePath(modelId)}/${manifest.graph_uri}`;
    const graph = await fetchJson(url, GraphSchema);
    const problems = crossCheckBundle(manifest, graph);
    if (problems.length > 0) {
      throw new BundleSchemaError(url, problems);
    }
    return graph;
  }

  async loadHeadAttention(experimentId: string, nodeId: string): Promise<HeadAttention> {
    // M0 bundles declare no attention assets. Say so instead of returning an empty matrix.
    throw new AttentionUnavailableError(experimentId, nodeId);
  }

  private basePath(modelId: string): string {
    const path = this.bundlePaths[modelId];
    if (path === undefined) {
      throw new Error(`No bundle registered for model "${modelId}"`);
    }
    return path;
  }
}
