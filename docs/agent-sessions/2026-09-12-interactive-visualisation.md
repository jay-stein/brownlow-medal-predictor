# Session: Interactive 2026 Forecast Visualisation

**Date:** 2026-09-12
**Branch:** `feat/2026-prediction`
**Commit:** `5ad5438` feat(web): add interactive 2026 Brownlow forecast visualisation

## Goal

Build an interactive React visualisation of the 2026 Brownlow forecast: pick a player, see their round-by-round and cumulative vote worm with simulation overlays (most likely path, 50% and 90% bands), styled around the 2026 Brownlow.

## Files Changed

| File | Change |
|---|---|
| `src/brownlow/simulate.py` | Round tracking: cumulative mean and quantiles per round, per-round increment distributions, sampled simulation paths; `forecast_export` payload builder |
| `src/brownlow/cli.py` | `forecast --web-json/--web-players/--web-paths`; export metadata |
| `tests/test_simulate.py` | Round monotonicity/parity test and export payload test (47 tests total) |
| `web/package.json`, `web/vite.config.js`, `web/index.html` | Vite + React + Recharts app scaffold |
| `web/src/App.jsx` | Layout, contender selector/search, stat cards, mode and overlay toggles, footnotes |
| `web/src/WormChart.jsx` | Cumulative worm with 50/90 bands, mean/median lines, sampled simulation overlays; per-round band bars |
| `web/src/styles.css` | 2026 Brownlow theme: deep navy, gold gradients, Playfair Display / Inter, medallion motif |
| `web/src/teams.js` | Team colours for chips and dots |
| `web/public/forecast_2026.json` | Generated payload: 60 contenders, 25 rounds, 60 sampled paths each (314 KB) |
| `README.md`, `docs/forecast-2026.md`, `.gitignore` | Docs, run instructions, ignore `web/dist` and `web/node_modules` |

## Commands Executed

- `uv run python -m brownlow.cli forecast --model ranking --season 2026 --n-sims 10000 --web-json web/public/forecast_2026.json --web-players 60 --web-paths 60`
- `cd web && npm install && npm run build` (831 modules, 560 KB JS / 159 KB gzip)
- Preview smoke test: `npm run preview` served index (200) and payload (200, 314 KB)
- `uv run pytest` (47 passed), `uv run ruff check .` (clean)

## Results

- The published forecast is unchanged by the export (`tau = 0.845`, effect scale 0.4): Nick Daicos 40.97 expected votes, 90% interval 32–49, 79.9% first-or-joint.
- Payload sanity: 60 players × 25 rounds × 60 paths; round labels `OR, R1…R24`; spec sensitivity range included per player (Daicos 79.9–92.3%).
- Build and static serving verified; the app reads `forecast_2026.json` via Vite's `BASE_URL`, so the same bundle works at any static host path.

## Important Decisions

1. **Separate RNG for path sampling** (`seed + 1`) so enabling the web export does not shift allocation draws; the published numbers are identical with and without `--web-json`.
2. **Aggregates over 10,000 simulations, paths from 60 sampled simulations**: full per-sim per-round arrays would be hundreds of MB, while quantile snapshots per round plus a path sample keep the payload at ~300 KB.
3. **Bands rendered with stacked transparent areas/bars** (5th percentile as an invisible base, then the band width), which gives clean 50% and 90% fans without a custom SVG layer.
4. **Top-5 probability counts boundary ties**, consistent with the season-level metrics.

## Blockers / Follow-ups

- Optional: deploy `web/dist` to GitHub Pages or Vercel for sharing.
- Optional: add an observed worm for backtest seasons once 2026 votes exist, and a head-to-head player comparison mode.
- Eligibility still not applied at the award stage.
