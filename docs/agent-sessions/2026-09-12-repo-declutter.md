# Session: Repository Declutter

**Date:** 2026-09-12
**Branch:** `agent/repo-tidy`
**Commits:** `d0041d3` docs: clarify Brownlow vote allocation wording; `ad1a202` chore: remove legacy notebooks, models and superseded artefacts

## Goal

Tidy the repository after the redesign: remove the legacy notebook pipeline and model binaries, and make the docs reflect the current package.

## What Was Removed

| Path | Contents | Size |
|---|---|---|
| `notebooks/` | 12 legacy pipeline notebooks plus stray model files | 15.0 MB |
| `models/` | 23 legacy ensemble/model JSONs and metadata | 9.0 MB |
| `archive/` | original 2022 notebook | 0.6 MB |
| `result/` | legacy Monte Carlo plots | 0.3 MB |
| `backcast/` | legacy backtest leaderboards (3 CSVs) | < 0.1 MB |
| `R/fitzroy_data_extract.Rmd` | superseded pointer document | — |
| `data/` snapshots | AFL.com.au, Betfair, ESPN, WheeloRatings, bonus map files | 0.8 MB |

Everything removed remains recoverable from git history.

## What Changed

| File | Change |
|---|---|
| `README.md` | Rewritten around the current pipeline: how it works, backtest and calibration results, structure, quick start, data, roadmap, caveats, licence |
| `data/README.md` | Rewritten for regeneration only; notes that earlier prediction snapshots live in history |
| `src/brownlow/paths.py` | Removed unused `MODELS_DIR`, `RESULT_DIR`, `BACKCAST_DIR` |
| `src/brownlow/features.py` | Docstring no longer references the deleted notebook |
| `.gitignore`, `pyproject.toml` | Dropped stale ignore rules and lint excludes for the removed folders |
| `docs/methodology.md` | Vote allocation wording clarified: "votes are awarded to 3 players only per match — summing to 6 total votes (3 + 2 + 1)" |

## Commands Executed

- `git rm -r archive notebooks models backcast result` plus `git rm` for the Rmd and data snapshots
- Removed leftover ignored artefacts from disk (`models/ensemble_predictions.csv`, `result/*.csv`)
- `uv run pytest` (47 passed), `uv run ruff check .` (clean)
- Reference sweep with `git grep` for `notebooks/`, `models/`, `backcast/`, `result/`, `archive/`

## Results

- Tracked files: **105 → 57**; repository size **26.4 MB → 0.8 MB**
- No code references to removed paths; tests and lint unaffected
- The active project is now just: R extraction scripts, the `src/brownlow` package, tests, the web app, docs, and project configuration

## Important Decisions

1. **Full declutter** chosen by the user; legacy notebooks were fully superseded by the package, and the parent repo's history also retains a copy.
2. **README rewritten rather than patched**, since the legacy pipeline sections no longer had source files to link to.
3. **Stale local artefacts deleted** too (legacy per-game CSVs under `result/` and `models/`), so the working tree matches the clean repository.

## Blockers / Follow-ups

- Optional: regenerate a screenshot of the web app for the README hero image (the old legacy plot was removed).
- Open roadmap items unchanged: award-stage eligibility, match-block bootstrap model variation, and a retrospective check of the 2026 forecast.
