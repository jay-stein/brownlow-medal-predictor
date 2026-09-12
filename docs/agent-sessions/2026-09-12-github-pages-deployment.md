# Session: GitHub Pages Deployment

**Date:** 2026-09-12
**Branch:** merged via `feat/2026-prediction` (PR #1) and `agent/pages-hardening`
**Commits:** `ca71a99` ci: deploy the forecast web app to GitHub Pages; `f681166` Merge pull request #1

## Goal

Publish the interactive 2026 Brownlow forecast to GitHub Pages.

## Files Changed

| File | Change |
|---|---|
| `.github/workflows/deploy-pages.yml` | New — build `web/` with `npm ci` and deploy the static site via `actions/deploy-pages` |
| `README.md` | Live site link at the top and in the visualisation section |
| `docs/agent-sessions/2026-09-12-github-pages-deployment.md` | This summary |

## Commands Executed

- `gh api -X POST .../pages -f build_type=workflow` — enabled Pages (Actions source)
- `gh repo edit ... --visibility public` — GitHub Pages on the Free plan requires a public repository
- `gh pr create` + `gh pr merge --merge` — merged the feature branch to `main`
- `gh run watch` — build and deploy jobs succeeded
- `Invoke-WebRequest` checks of the live index and forecast payload

## Results

- Live site: https://jay-stein.github.io/brownlow-medal-predictor/
- Index returns 200; `forecast_2026.json` returns 200 with the full 314 KB payload
- Vite `base: './'` and `import.meta.env.BASE_URL` keep assets and data resolving correctly under the `/brownlow-medal-predictor/` path
- Workflow action majors refreshed to `checkout@v7`, `setup-node@v7`, `upload-pages-artifact@v5`, `deploy-pages@v5` to clear Node 20 deprecation warnings

## Important Decisions

1. **Public repository**: GitHub Pages for private repositories requires a paid plan; with the user's choice, the repository was made public after a secret scan of all 95 tracked files (no hits).
2. **Actions-based Pages** (`build_type: workflow`) rather than a `gh-pages` branch: the site is rebuilt from `web/` on every push to `main`, so the published page always matches the source.
3. **Deploy from `main`**, with the feature branch merged by PR to respect the branch workflow.

## Blockers / Follow-ups

- The forecast payload is regenerated manually (`forecast --web-json`); a follow-up could add a scheduled workflow to refresh it once votes are available.
- The repo is now public: review license before wider sharing (none added yet).
