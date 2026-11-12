import path from "node:path";
import { fileURLToPath } from "node:url";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

const here = path.dirname(fileURLToPath(import.meta.url));

// Public path the app is served under. "/" locally; a GitHub Pages project site sets
// CIRQLOOM_BASE_PATH=/cirqloom/ at build time (see .github/workflows/pages.yml).
const base = process.env.CIRQLOOM_BASE_PATH ?? "/";
if (!base.startsWith("/") || !base.endsWith("/")) {
  throw new Error(`CIRQLOOM_BASE_PATH must start and end with "/", got ${JSON.stringify(base)}`);
}

export default defineConfig({
  base,
  plugins: [react(), tailwindcss()],
  // Serve ../data as static files: /fixtures/gpt2-small-demo/manifest.json
  publicDir: path.resolve(here, "../data"),
  test: {
    environment: "jsdom",
    setupFiles: ["src/tests/setup.ts"],
    include: ["src/**/*.test.{ts,tsx}"],
  },
});
