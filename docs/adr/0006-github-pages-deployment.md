# ADR 0006: Static deployment to GitHub Pages under a subpath

Status: accepted, 2026-11-11

## Context

The M3 release needs a public demo that runs without Python, model weights or a server. The app is already static: it reads `manifest.json`, `graph.json` and lazily loaded assets through `StaticBundleProvider`, and Vite serves `data/` as its public directory. A GitHub Pages project site is served under a subpath (`https://<owner>.github.io/cirqloom/`), but the app fetched bundles from root-absolute paths (`/bundles/...`), which would 404 there. ADR 0001 also said bundles larger than the demo fixture should not be committed.

## Decision

- **Base path at build time.** `vite.config.ts` reads `CIRQLOOM_BASE_PATH` (default `/`; must start and end with `/`). `App.tsx` builds bundle URLs from `import.meta.env.BASE_URL`, so the same code works locally (`/`) and on Pages (`/cirqloom/`). No other root-absolute URL exists in the app.
- **Deployment workflow.** `.github/workflows/pages.yml` builds with `CIRQLOOM_BASE_PATH=/<repo>/`, runs the full Playwright suite against `vite preview` of that exact build under the subpath, and only then uploads and deploys `frontend/dist` with `actions/deploy-pages`. Pages must be set to "GitHub Actions" as its source (a one-time repository setting).
- **Same tests, two targets.** E2E tests navigate with relative URLs (`./`, `./?bundle=fixture`). `playwright.config.ts` targets the dev server by default and the built preview when `CIRQLOOM_BASE_PATH` is set (`npm run e2e:pages`).
- **The measured bundle stays committed** (12 MB on disk; `graph.json` 7.5 MB, about 0.5 MB gzipped, which GitHub Pages applies). This supersedes the size guidance in ADR 0001 for the demo bundle: committing it keeps the demo, CI validation and the reproducibility check on one versioned artifact. Revisit (release assets, compact JSON, or per-prompt scores in a lazy asset) if the bundle grows past about 25 MB or first load becomes slow.

## Consequences

- A broken subpath URL fails the deploy job before anything is published.
- The deployed site contains only `frontend/dist`: the built app plus `data/` (both bundles). No model weights, no `.venv`, no `AGENTS.md`.
- Other static hosts work with the same build, given the right `CIRQLOOM_BASE_PATH`.
