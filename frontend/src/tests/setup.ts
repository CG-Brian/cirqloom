import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

// Testing Library only auto-cleans when Vitest globals are on; we import vitest explicitly.
afterEach(cleanup);

// jsdom has no 2D canvas. Return null (what a browser does for an unsupported context) instead
// of letting jsdom log "Not implemented" on every heatmap render; drawing is browser-tested.
if (typeof HTMLCanvasElement !== "undefined") {
  HTMLCanvasElement.prototype.getContext = (() => null) as typeof HTMLCanvasElement.prototype.getContext;
}
