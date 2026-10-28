// @vitest-environment node
// Lazy attention loading from the committed measured GPT-2 bundle, with fetch served from disk.
import { readFileSync } from "node:fs";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AttentionUnavailableError } from "../data/DataProvider";
import { decodeFloat16, halfToNumber } from "../data/float16";
import { StaticBundleProvider } from "../data/StaticBundleProvider";

const DATA_DIR = path.resolve(process.cwd(), "../data");
const BUNDLE = "/bundles/gpt2-small-repeated-v1";
const EXPERIMENT = "repeated_tokens_v1:L10";

describe("float16 decoding", () => {
  it("decodes known bit patterns", () => {
    expect(halfToNumber(0x0000)).toBe(0);
    expect(halfToNumber(0x3c00)).toBe(1);
    expect(halfToNumber(0x3800)).toBe(0.5);
    expect(halfToNumber(0xc000)).toBe(-2);
    expect(halfToNumber(0x0001)).toBeCloseTo(5.96e-8, 10); // smallest subnormal
    expect(halfToNumber(0x7c00)).toBe(Infinity);
    expect(halfToNumber(0x7e00)).toBeNaN();
  });

  it("reads little-endian values at an element offset", () => {
    const buffer = new Uint8Array([0x00, 0x3c, 0x00, 0x38, 0x00, 0xc0]).buffer;
    expect(Array.from(decodeFloat16(buffer, 1, 2))).toEqual([0.5, -2]);
  });
});

describe("StaticBundleProvider.loadHeadAttention", () => {
  const requested: string[] = [];

  beforeEach(() => {
    requested.length = 0;
    vi.stubGlobal("fetch", async (url: string) => {
      requested.push(url);
      try {
        const body = readFileSync(path.join(DATA_DIR, url));
        return new Response(body, { status: 200 });
      } catch {
        return new Response("missing", { status: 404 });
      }
    });
  });

  afterEach(() => vi.unstubAllGlobals());

  it("returns real tokens and a causal, row-normalised matrix for one head", async () => {
    const provider = new StaticBundleProvider({ "gpt2-small": BUNDLE });
    const att = await provider.loadHeadAttention(EXPERIMENT, "model:gpt2-small:head:L4H11");
    expect(att.promptId).toBe("repeated_tokens_v1:L10:seed0");
    expect(att.tokens).toHaveLength(21);
    expect(att.tokens[0]).toBe("<|endoftext|>");
    expect(att.tokens.slice(1, 11)).toEqual(att.tokens.slice(11)); // the repeated half
    expect(att.weights).toHaveLength(21);
    att.weights.forEach((row, q) => {
      expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 2); // float16: ~3 digits
      expect(row.slice(q + 1).every((w) => w === 0)).toBe(true); // no future cells
    });
  });

  it("fetches the experiment asset once and reuses it for other heads", async () => {
    const provider = new StaticBundleProvider({ "gpt2-small": BUNDLE });
    await provider.loadHeadAttention(EXPERIMENT, "model:gpt2-small:head:L0H0");
    await provider.loadHeadAttention(EXPERIMENT, "model:gpt2-small:head:L5H5");
    expect(requested.filter((u) => u.endsWith(".attention.f16"))).toHaveLength(1);
  });

  it("refuses experiments without attention and unknown heads", async () => {
    const provider = new StaticBundleProvider({ "gpt2-small": BUNDLE });
    await expect(
      provider.loadHeadAttention("nonrepeated_tokens_v1:L10", "model:gpt2-small:head:L0H0"),
    ).rejects.toBeInstanceOf(AttentionUnavailableError);
    await expect(
      provider.loadHeadAttention(EXPERIMENT, "model:gpt2-small:head:L12H0"),
    ).rejects.toBeInstanceOf(AttentionUnavailableError);
  });
});
