// Browser smoke tests (AGENTS.md §12): the real app, real WebGL canvas, real bundles.
// Most tests use the synthetic fixture (?bundle=fixture) for its demo circuit; the last one
// opens the default measured GPT-2 bundle.
import { expect, test, type Page } from "@playwright/test";
import { PerspectiveCamera, Vector3 } from "three";
import { readFileSync } from "node:fs";
import path from "node:path";
import { GraphSchema, ManifestSchema } from "../src/data/schema";
import { getCircuitLayout } from "../src/features/circuits/circuitLayout";
import { getVisualEdgeGroups } from "../src/features/circuits/edgeGroups";
import { buildGraphView } from "../src/features/experiment/graphView";
import { CAMERA_FOV, canonicalCameraPose, headPosition, sceneBounds, type Vec3 } from "../src/scene/layout";

// The fixture's shape (GPT-2 Small).
const LAYERS = 12;
const HEADS = 12;

/** Collects console errors and uncaught exceptions for the whole test. */
function trackErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on("console", (msg) => {
    if (msg.type() === "error") {
      errors.push(msg.text());
    }
  });
  page.on("pageerror", (err) => errors.push(err.message));
  return errors;
}

const headButton = (page: Page, label: string) =>
  page.getByRole("button", { name: new RegExp(`^${label}\\b`) });

test.beforeEach(async ({ page }, testInfo) => {
  if (!testInfo.title.startsWith("measured bundle")) {
    await page.goto("/?bundle=fixture");
  }
});

test("loads the scene and labels the fixture", async ({ page }) => {
  await expect(page.getByTestId("model-scene").locator("canvas")).toBeVisible();
  await expect(page.getByRole("status")).toHaveText("Demo fixture — not measured model data.");
  await expect(page.getByRole("group", { name: "Attention heads by layer" }).getByRole("button")).toHaveCount(144);
});

test("selecting a head in the list opens it in the inspector", async ({ page }) => {
  const inspector = page.getByRole("complementary", { name: "Head inspector" });
  await headButton(page, "L0H0").click();
  await expect(headButton(page, "L0H0")).toHaveAttribute("aria-pressed", "true");
  await expect(inspector.getByText("model:gpt2-small:head:L0H0")).toBeVisible();
  await expect(inspector.getByText(/no experiments/i)).toBeVisible();

  await page.keyboard.press("Escape");
  await expect(headButton(page, "L0H0")).toHaveAttribute("aria-pressed", "false");
  await expect(inspector.getByText(/select a head/i)).toBeVisible();
});

/** Where a head appears on screen, using the app's own layout and canonical camera. */
function headScreenPoint(
  layer: number,
  head: number,
  box: { x: number; y: number; width: number; height: number },
) {
  const positions = Array.from({ length: LAYERS * HEADS }, (_, i) =>
    headPosition(Math.floor(i / HEADS), i % HEADS, LAYERS, HEADS),
  );
  const aspect = box.width / box.height;
  const pose = canonicalCameraPose(sceneBounds(positions, LAYERS, HEADS), aspect);
  const camera = new PerspectiveCamera(CAMERA_FOV, aspect, 0.1, 500);
  camera.position.set(...pose.position);
  camera.lookAt(...pose.target);
  camera.updateMatrixWorld();
  const ndc = new Vector3(...headPosition(layer, head, LAYERS, HEADS)).project(camera);
  return { x: box.x + ((ndc.x + 1) / 2) * box.width, y: box.y + ((1 - ndc.y) / 2) * box.height };
}

test("clicking a head in the 3D scene selects it in the list and inspector", async ({ page }) => {
  const canvas = page.getByTestId("model-scene").locator("canvas");
  await expect(canvas).toBeVisible();
  const inspector = page.getByRole("complementary", { name: "Head inspector" });
  let box = await canvas.boundingBox();

  // Under load the first frames may render before the layout settles and the camera has
  // fitted the canvas, so retry briefly, re-measuring the canvas on every attempt.
  await expect(async () => {
    box = await canvas.boundingBox();
    if (!box) {
      throw new Error("canvas has no bounding box");
    }
    const point = headScreenPoint(5, 5, box);
    await page.mouse.click(point.x, point.y);
    await expect(inspector.getByText("model:gpt2-small:head:L5H5")).toBeVisible({ timeout: 1_000 });
  }).toPass({ timeout: 15_000 });
  await expect(headButton(page, "L5H5")).toHaveAttribute("aria-pressed", "true");

  // A click on empty space (top-left corner of the canvas) clears the selection.
  await page.mouse.click(box!.x + 8, box!.y + 8);
  await expect(headButton(page, "L5H5")).toHaveAttribute("aria-pressed", "false");
});

test("circuit overlay toggles and shows its role legend", async ({ page }) => {
  const circuit = page.getByRole("button", { name: /Demo fixture circuit/ }).first();
  await circuit.click();
  await expect(circuit).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByText("Roles in Demo fixture circuit")).toBeVisible();
  await expect(page.getByRole("contentinfo")).toContainText("Overlay: Demo fixture circuit");

  await circuit.click();
  await expect(circuit).toHaveAttribute("aria-pressed", "false");
  await expect(page.getByText("Roles in Demo fixture circuit")).toBeHidden();
});

test("reset view and panel collapse run without console errors", async ({ page }) => {
  const errors = trackErrors(page);
  await page.reload();
  await expect(page.getByTestId("model-scene").locator("canvas")).toBeVisible();

  const canvas = page.getByTestId("model-scene").locator("canvas");
  const box = await canvas.boundingBox();
  if (box) {
    await page.mouse.move(box.x + box.width * 0.3, box.y + box.height * 0.8);
    await page.mouse.down();
    await page.mouse.move(box.x + box.width * 0.6, box.y + box.height * 0.7, { steps: 8 });
    await page.mouse.up();
  }
  await page.getByRole("button", { name: "Reset view" }).click();
  await page.getByRole("button", { name: "Collapse circuit sidebar" }).click();
  await page.getByRole("button", { name: "Expand circuit sidebar" }).click();
  await page.getByRole("button", { name: "Collapse head inspector" }).click();
  await page.waitForTimeout(500);

  expect(errors).toEqual([]);
});

test("measured bundle: real scores per experiment, control labelled, no fixture warning", async ({ page }) => {
  const errors = trackErrors(page);
  await page.goto("/");
  await expect(page.getByRole("status")).toContainText("Measured on openai-community/gpt2");
  await expect(page.getByText("Demo fixture — not measured model data.")).toHaveCount(0);

  const inspector = page.getByRole("complementary", { name: "Head inspector" });
  await headButton(page, "L5H5").click();
  await expect(inspector.getByText("model:gpt2-small:head:L5H5")).toBeVisible();
  await expect(inspector.getByText("repeated_tokens_v1:L10", { exact: true })).toBeVisible();
  await expect(inspector.getByText(/^induction_attention/)).toBeVisible();
  await expect(inspector.getByText(/n = 8 prompts/).first()).toBeVisible();
  await expect(inspector.getByText("Experimental").first()).toBeVisible();

  await page.getByLabel("Experiment").selectOption("nonrepeated_tokens_v1:L10");
  await expect(inspector.getByText(/Control run/)).toBeVisible();
  await expect(inspector.getByText(/keeps scores only/)).toBeVisible();
  expect(errors).toEqual([]);
});

test("measured bundle: heatmap draws real attention and outlines the exported mask", async ({ page }) => {
  const errors = trackErrors(page);
  await page.goto("/");
  const inspector = page.getByRole("complementary", { name: "Head inspector" });
  await headButton(page, "L5H5").click();

  const map = inspector.getByRole("img", { name: /21 by 21/ });
  await expect(map).toBeVisible();
  // The canvas really has pixels on it (not a placeholder).
  const painted = await map.evaluate((canvas: HTMLCanvasElement) => {
    const data = canvas.getContext("2d")!.getImageData(0, 0, canvas.width, canvas.height).data;
    let opaque = 0;
    for (let i = 3; i < data.length; i += 4) opaque += data[i]! > 0 ? 1 : 0;
    return opaque;
  });
  expect(painted).toBeGreaterThan(0);

  await inspector.getByRole("button", { name: /cells averaged by induction_attention/ }).click();
  await expect(map).toHaveAccessibleName(/outlined: 10 cells averaged by induction_attention v1\.0\.0/);

  // Hover the induction cell (q 12, k 3) of the 21×21 map: L = 10, k = q - L + 1.
  await map.scrollIntoViewIfNeeded();
  const box = (await map.boundingBox())!;
  const cell = box.width / 21;
  await page.mouse.move(box.x + 3.5 * cell, box.y + 12.5 * cell);
  const readout = inspector.getByTestId("heatmap-readout");
  await expect(readout).toContainText("q 12");
  await expect(readout).toContainText("k 3");
  await expect(readout).toContainText("in mask");
  await expect(readout).toContainText(/weight \d\.\d{2}/);

  // A future cell is reported as not attendable, never as a weight.
  await page.mouse.move(box.x + 15.5 * cell, box.y + 2.5 * cell);
  await expect(readout).toContainText("future position: not attendable");
  expect(errors).toEqual([]);
});

test("measured bundle: the published IOI circuit overlays its 26 heads in 7 roles", async ({ page }) => {
  const errors = trackErrors(page);
  await page.goto("/");
  const sidebar = page.getByRole("complementary", { name: "Circuits and heads" });
  const ioi = sidebar.getByRole("button", { name: /^Indirect Object Identification \(IOI\)/ });
  await expect(ioi).toContainText("26 heads");
  await expect(ioi).toContainText("Published annotation");
  await ioi.click();
  await expect(ioi).toHaveAttribute("aria-pressed", "true");
  for (const role of ["Previous Token", "Duplicate Token", "Induction", "S-Inhibition", "Name Mover",
    "Negative Name Mover", "Backup Name Mover"]) {
    await expect(sidebar.getByText(role, { exact: true })).toBeVisible();
  }

  const inspector = page.getByRole("complementary", { name: "Head inspector" });
  await headButton(page, "L7H3").click();
  await expect(inspector.getByText("S-Inhibition", { exact: true })).toBeVisible();
  await expect(inspector.getByTestId("published-vs-measured")).toContainText("do not test");
  expect(errors).toEqual([]);
});


/** The committed GPT-2 bundle, indexed exactly as the app does. */
function loadGpt2View() {
  const dir = path.resolve(process.cwd(), "../data/bundles/gpt2-small");
  const read = (name: string): unknown => JSON.parse(readFileSync(path.join(dir, name), "utf8"));
  return buildGraphView(ManifestSchema.parse(read("manifest.json")), GraphSchema.parse(read("graph.json")));
}

/** Screen point of a world position under the app's canonical camera. */
function screenPoint(world: Vec3, view: ReturnType<typeof loadGpt2View>, box: { x: number; y: number; width: number; height: number }) {
  const aspect = box.width / box.height;
  const pose = canonicalCameraPose(view.bounds, aspect);
  const camera = new PerspectiveCamera(CAMERA_FOV, aspect, 0.1, 500);
  camera.position.set(...pose.position);
  camera.lookAt(...pose.target);
  camera.updateMatrixWorld();
  const ndc = new Vector3(...world).project(camera);
  return { x: box.x + ((ndc.x + 1) / 2) * box.width, y: box.y + ((1 - ndc.y) / 2) * box.height };
}

test("measured bundle: IOI relations open from the list with every evidence record", async ({ page }) => {
  const errors = trackErrors(page);
  await page.goto("/");
  const sidebar = page.getByRole("complementary", { name: "Circuits and heads" });
  await sidebar.getByRole("button", { name: /^Indirect Object Identification \(IOI\)/ }).click();

  const relations = sidebar.getByRole("list", { name: "Circuit relations" });
  await expect(relations.getByRole("button")).toHaveCount(8);
  await expect(sidebar.getByText("Relations (8 · 27 evidence records)")).toBeVisible();

  await relations.getByRole("button", { name: /^S-Inhibition heads → Name Mover queries/ }).click();
  const inspector = page.getByRole("complementary", { name: "Relation inspector" });
  await expect(inspector.getByText("Published experiment")).toBeVisible();
  await expect(inspector.getByText(/Where: §3\.2, Figure 4b/)).toBeVisible();
  await expect(inspector.getByRole("list", { name: "Evidence records" }).getByRole("listitem")).toHaveCount(4);

  await page.keyboard.press("Escape");
  await expect(page.getByRole("complementary", { name: "Head inspector" })).toBeVisible();
  expect(errors).toEqual([]);
});

test("measured bundle: clicking a ribbon in 3D opens that relation", async ({ page }) => {
  const errors = trackErrors(page);
  await page.goto("/");
  await page
    .getByRole("complementary", { name: "Circuits and heads" })
    .getByRole("button", { name: /^Indirect Object Identification \(IOI\)/ })
    .click();

  const view = loadGpt2View();
  const ioi = "circuit:gpt2-small:ioi:wang-2022";
  const groups = getVisualEdgeGroups(view, ioi);
  const layout = getCircuitLayout(view, ioi, groups);
  const target = groups.find((g) => g.conditions.length > 0)!; // Backup Name Movers → logits
  const curve = layout.edges.find((e) => e.groupId === target.id)!;
  // A point a third of the way along, away from the channel label at the middle.
  const world = curve.points[Math.round(curve.points.length / 3)]!;

  const canvas = page.getByTestId("model-scene").locator("canvas");
  const inspector = page.getByRole("complementary", { name: "Relation inspector" });
  await expect(async () => {
    const box = (await canvas.boundingBox())!;
    const point = screenPoint(world, view, box);
    await page.mouse.click(point.x, point.y);
    await expect(inspector.getByText("Backup Name Mover heads → logits")).toBeVisible({ timeout: 1_000 });
  }).toPass({ timeout: 15_000 });
  await expect(inspector.getByText("Only after the Name Mover heads are knocked out.")).toBeVisible();
  expect(errors).toEqual([]);
});
