// Captures the README screenshots from the running app (real WebGL canvas, committed bundles).
// Usage, with the dev server or a deployed site running:
//   npm run screenshots                         # http://localhost:5173/
//   npm run screenshots -- https://example.org/  # any deployment
// Writes PNGs to docs/assets/. Nothing is mocked: every pixel is the app on the GPT-2 bundle.
import { chromium, type Page } from "@playwright/test";
import path from "node:path";

const BASE = process.argv[2] ?? "http://localhost:5173/";
const OUT = path.resolve(process.cwd(), "../docs/assets");
const VIEWPORT = { width: 1600, height: 960 };
const ZOOM_STEPS = 3;

const IOI = /^Indirect Object Identification \(IOI\)/;
const RELATION = /^S-Inhibition heads → Name Mover queries/;

/** Lets the camera animation and lazy assets settle, and parks the pointer off the canvas. */
async function settle(page: Page, ms = 1500) {
  await page.mouse.move(4, VIEWPORT.height - 4);
  await page.waitForTimeout(ms);
}

async function open(page: Page) {
  await page.goto(BASE);
  await page.getByRole("status").filter({ hasText: "Measured on openai-community/gpt2" }).waitFor();
  await page.getByTestId("model-scene").locator("canvas").waitFor();
}

async function shoot(page: Page, name: string) {
  const file = path.join(OUT, name);
  await page.screenshot({ path: file });
  console.log(`wrote ${file}`);
}

const browser = await chromium.launch();
const errors: string[] = [];
const page = await browser.newPage({ viewport: VIEWPORT, deviceScaleFactor: 1 });
page.on("console", (msg) => {
  if (msg.type() === "error") errors.push(msg.text());
});
page.on("pageerror", (err) => errors.push(err.message));

// 1. The published IOI circuit over the 144-head atlas, framed by "Focus camera".
await open(page);
const sidebar = page.getByRole("complementary", { name: "Circuits and heads" });
await sidebar.getByRole("button", { name: IOI }).click();
await sidebar.getByRole("button", { name: "Focus camera on Indirect Object Identification (IOI)" }).click();
await settle(page);
// Zoom in a little with the wheel, as a visitor would; the focus pose leaves a wide margin.
const canvasBox = (await page.getByTestId("model-scene").locator("canvas").boundingBox())!;
await page.mouse.move(canvasBox.x + canvasBox.width / 2, canvasBox.y + canvasBox.height / 2);
for (let i = 0; i < ZOOM_STEPS; i++) {
  await page.mouse.wheel(0, -200);
  await page.waitForTimeout(100);
}
await settle(page);
await shoot(page, "ioi-circuit.png");

// 2. A relation from the paper next to attention observed on an IOI prompt.
await page.getByLabel("Experiment").selectOption("ioi_templates_v1:BABA-t00");
await sidebar.getByRole("list", { name: "Circuit relations" }).getByRole("button", { name: RELATION }).click();
const relation = page.getByRole("complementary", { name: "Relation inspector" });
await relation.getByTestId("attention-alignment").waitFor();
await settle(page);
await shoot(page, "relation-inspector.png");

// 3. The full atlas and an induction head's attention heatmap with its scored cells outlined.
await open(page);
await page.getByRole("button", { name: /^L5H5\b/ }).click();
const inspector = page.getByRole("complementary", { name: "Head inspector" });
await inspector.getByRole("button", { name: /cells averaged by induction_attention/ }).click();
const map = inspector.getByRole("img", { name: /21 by 21/ });
// Scroll the panel so the map and the readout line under it are both in view.
await inspector.getByTestId("heatmap-readout").evaluate((el) => el.scrollIntoView({ block: "end" }));
await settle(page, 800);
// Hover the induction cell (q 12, k 3) so the readout shows a real weight.
const box = (await map.boundingBox())!;
const cell = box.width / 21;
await page.mouse.move(box.x + 3.5 * cell, box.y + 12.5 * cell);
await inspector.getByTestId("heatmap-readout").filter({ hasText: "in mask" }).waitFor();
await shoot(page, "head-inspector.png");

await browser.close();
if (errors.length > 0) {
  console.error(`console errors:\n${errors.join("\n")}`);
  process.exit(1);
}
