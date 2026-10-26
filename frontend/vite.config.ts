import path from "node:path";
import { fileURLToPath } from "node:url";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

const here = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  plugins: [react(), tailwindcss()],
  // Serve ../data as static files: /fixtures/gpt2-small-demo/manifest.json
  publicDir: path.resolve(here, "../data"),
  test: {
    environment: "jsdom",
    setupFiles: ["src/tests/setup.ts"],
    include: ["src/**/*.test.{ts,tsx}"],
  },
});
