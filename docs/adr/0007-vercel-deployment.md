# ADR 0007: Vercel deployment at the root path

Status: accepted, 2026-11-12

## Context

The public demo needs a stable URL at the root of a domain. ADR 0006 made the build base-path aware and added a GitHub Pages workflow for a project subpath (`/cirqloom/`). The app is a static Vite build; its public directory is `../data` (outside `frontend/`), so the host must let the build read files above the app folder. No Python, model weights or server code belong in the deployment.

## Decision

- **Vercel Git integration**, project Root Directory `frontend`. Vercel's default setting "Include files outside the root directory in the Build Step" stays on, so `vite build` copies `../data` (both committed bundles) into `dist/` exactly as it does locally. Nothing is duplicated in the repository.
- **`frontend/vercel.json`** pins the build: framework `vite`, `npm ci`, `CIRQLOOM_BASE_PATH=/ npm run build`, output `dist`. No functions, rewrites or headers: the app is one page, state lives in the query string (`?bundle=fixture`), and Vercel compresses JSON responses itself.
- **GitHub Pages stays** as configured in ADR 0006. Both hosts build the same commit; only the base path differs.
- A Git deployment clones tracked files only, so git-ignored local files (`AGENTS.md`, `.venv`, `.env`, `.claude/`) cannot reach it. Do not deploy with the Vercel CLI from a working copy without a `.vercelignore`.

## Verification

A clean clone (tracked files only) built under Node 24 with the commands above produces `dist/` containing `index.html`, `assets/`, `bundles/gpt2-small/` and `fixtures/gpt2-small-demo/`. After each production deploy, check `/`, `/bundles/gpt2-small/manifest.json`, `/bundles/gpt2-small/graph.json` and `/fixtures/gpt2-small-demo/manifest.json`, and run the browser suite against the live URL.

## Consequences

- If the "Include files outside the root directory" setting is turned off, the build still succeeds but ships without bundles, and the app shows a load error. Keep it on.
- `npm run screenshots -- <url>` can capture the README images from either deployment.
