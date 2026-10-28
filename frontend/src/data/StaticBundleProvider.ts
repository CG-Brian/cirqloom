import type { z } from "zod";
import {
  AttentionUnavailableError,
  BundleLoadError,
  BundleSchemaError,
  type DataProvider,
  type HeadAttention,
  type ModelSummary,
} from "./DataProvider";
import { decodeFloat16 } from "./float16";
import {
  crossCheckBundle,
  GraphSchema,
  ManifestSchema,
  TokensFileSchema,
  type Experiment,
  type Graph,
  type Manifest,
  type TokensFile,
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

async function fetchBinary(url: string): Promise<ArrayBuffer> {
  let res: Response;
  try {
    res = await fetch(url);
  } catch (err) {
    throw new BundleLoadError(url, null, String(err));
  }
  if (!res.ok) {
    throw new BundleLoadError(url, res.status, `HTTP ${res.status}`);
  }
  return res.arrayBuffer();
}

const HEAD_ID = /:head:L(\d+)H(\d+)$/;

/** Memoises successful loads by key; failed loads are dropped so they can be retried. */
function memo<T>(cache: Map<string, Promise<T>>, key: string, load: () => Promise<T>): Promise<T> {
  let pending = cache.get(key);
  if (!pending) {
    pending = load();
    pending.catch(() => cache.delete(key));
    cache.set(key, pending);
  }
  return pending;
}

/** Reads precomputed bundles served as static files. */
export class StaticBundleProvider implements DataProvider {
  private readonly bundlePaths: Record<string, string>;
  private readonly manifests = new Map<string, Promise<Manifest>>();
  private readonly binaries = new Map<string, Promise<ArrayBuffer>>();
  private readonly tokenFiles = new Map<string, Promise<TokensFile>>();

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
    return memo(this.manifests, modelId, () =>
      fetchJson(`${this.basePath(modelId)}/manifest.json`, ManifestSchema),
    );
  }

  // One graph per bundle holds every experiment's observations; filter by experiment in the UI.
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

  /**
   * Attention of one head on the experiment's stored example prompt. Fetches the experiment's
   * float16 asset once (lazily, on first selection) and decodes only this head's slice.
   */
  async loadHeadAttention(experimentId: string, nodeId: string): Promise<HeadAttention> {
    const found = await this.findExperiment(experimentId);
    const asset = found?.experiment.attention;
    if (!found || !asset) {
      throw new AttentionUnavailableError(experimentId, nodeId);
    }
    const { modelId, experiment } = found;
    const match = HEAD_ID.exec(nodeId);
    const [layers, heads, rows, cols] = asset.shape as [number, number, number, number];
    const layer = match ? Number(match[1]) : -1;
    const head = match ? Number(match[2]) : -1;
    if (layer < 0 || layer >= layers || head < 0 || head >= heads) {
      throw new AttentionUnavailableError(experimentId, nodeId);
    }

    const base = this.basePath(modelId);
    const binUrl = `${base}/${asset.uri}`;
    const tokensUrl = `${base}/${experiment.tokens_uri}`;
    const [buffer, tokensFile] = await Promise.all([
      memo(this.binaries, binUrl, () => fetchBinary(binUrl)),
      memo(this.tokenFiles, tokensUrl, () => fetchJson(tokensUrl, TokensFileSchema)),
    ]);

    const expectedBytes = layers * heads * rows * cols * 2;
    if (buffer.byteLength !== expectedBytes) {
      throw new BundleSchemaError(binUrl, [
        `${buffer.byteLength} bytes, expected ${expectedBytes} for shape [${asset.shape.join(", ")}]`,
      ]);
    }
    const prompt = tokensFile.prompts[asset.prompt_index];
    if (!prompt || prompt.tokens.length !== rows || rows !== experiment.sequence_length) {
      throw new BundleSchemaError(tokensUrl, [
        `prompt ${asset.prompt_index} does not match the attention asset of ${experiment.id}`,
      ]);
    }

    const flat = decodeFloat16(buffer, (layer * heads + head) * rows * cols, rows * cols);
    const weights = Array.from({ length: rows }, (_, q) =>
      Array.from(flat.subarray(q * cols, (q + 1) * cols)),
    );
    return { experimentId, nodeId, promptId: prompt.id, tokens: prompt.tokens, weights };
  }

  private async findExperiment(
    experimentId: string,
  ): Promise<{ modelId: string; experiment: Experiment } | null> {
    for (const modelId of Object.keys(this.bundlePaths)) {
      const manifest = await this.loadManifest(modelId);
      const experiment = manifest.experiments.find((e) => e.id === experimentId);
      if (experiment) {
        return { modelId, experiment };
      }
    }
    return null;
  }

  private basePath(modelId: string): string {
    const path = this.bundlePaths[modelId];
    if (path === undefined) {
      throw new Error(`No bundle registered for model "${modelId}"`);
    }
    return path;
  }
}
