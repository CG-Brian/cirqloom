import { defineConfig, devices } from "@playwright/test";

// Two targets, one test file:
// - default: the Vite dev server on a dedicated port, so the run never collides with a dev
//   server already on 5173;
// - CIRQLOOM_BASE_PATH set (e.g. "/cirqloom/"): the production build served by `vite preview`
//   under that subpath, the way GitHub Pages serves it. `npm run e2e:pages` builds first.
// Tests navigate with relative URLs ("./"), so they resolve under either base.
const basePath = process.env.CIRQLOOM_BASE_PATH;
const PORT = basePath ? 5198 : 5199;

export default defineConfig({
  testDir: "e2e",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [["list"], ["html", { open: "never" }]] : "list",
  use: {
    baseURL: `http://localhost:${PORT}${basePath ?? "/"}`,
    trace: "retain-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    command: basePath
      ? `npx vite preview --port ${PORT} --strictPort`
      : `npm run dev -- --port ${PORT} --strictPort`,
    url: `http://localhost:${PORT}${basePath ?? "/"}`,
    reuseExistingServer: !process.env.CI,
    timeout: 60_000,
  },
});
