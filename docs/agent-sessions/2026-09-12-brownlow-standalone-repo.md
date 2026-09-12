# Session: Standalone Brownlow Medal Predictor Repo

**Date:** 2026-09-12
**Branches:** `main` + `feat/2026-prediction` (this repo), `agent/untrack-brownlow-standalone` (parent repo)
**Commits:** `5df3343` chore: initial commit; `faecdfd` chore(brownlow): untrack project moved to standalone repo

## Goal

Extract the Brownlow Medal Predictor from the `data-science-notebooks` monorepo into a standalone private GitHub repository with proper documentation (project README, data regeneration guide, Python requirements), then create a `feat/2026-prediction` branch ready for next season's work.

## Files Changed

### New repo: `brownlow-medal-predictor` (52 files, 24.4 MB)

| File | Change |
|---|---|
| `README.md` | New — project overview, pipeline walkthrough, notebook run order, data provenance, environment, roadmap, caveats |
| `data/README.md` | New — committed vs regenerated datasets + fitzRoy regeneration steps |
| `requirements.txt` | New — pandas, numpy, matplotlib, seaborn, plotly, xgboost, scikit-learn, shap, tqdm, openpyxl |
| `.gitignore` | New — excludes the 74 MB fitzRoy extracts, `models/ensemble_predictions.csv`, `result/*.csv`, and `.copilot-tracking/` |
| Existing content | Notebooks, R script, model JSONs, backcast CSVs, and result PNGs imported as-is |

### Parent repo: `data-science-notebooks`

| File | Change |
|---|---|
| `.gitignore` | Added `Projects/brownlow/` under "Specific large files/folders" |
| `Projects/brownlow/**` | Untracked from index (39 files); working tree left untouched on disk |

## Commands Executed

- `git init -b main`, `git add -A`, `git commit -m "chore: initial commit"`
- `gh repo create jay-stein/brownlow-medal-predictor --private --source=. --remote=origin --push`
- `git checkout -b feat/2026-prediction`, `git push -u origin feat/2026-prediction`
- Parent: `git checkout -b agent/untrack-brownlow-standalone`, `git rm -r --cached --quiet Projects/brownlow`, `git add .gitignore`, `git commit`
- Verification: `git status`, `git check-ignore -v`, `git ls-remote`, `gh repo view`, `gh api .../contents/README.md`
- Session export: `opencode export ses_f6bd955a6ffe5pYwCUMiAdwNn3` (saved to `%TEMP%\opencode\brownlow-session-export.json`)

## Important Decisions

### 1. Fresh git history instead of subtree split
The parent repo's history for `Projects/brownlow` was a single "Initial commit: organised Jupyter notebook collection", so a fresh history lost nothing meaningful and avoided unnecessary complexity.

### 2. Data strategy — exclude large, include small
The three fitzRoy extracts (74 MB) and derived CSVs are regenerable, so they stay out of git. Small prediction-source snapshots (~0.8 MB: AFL.com.au, Betfair, ESPN, Wheelo, bonus map) are committed so the pipeline can be re-run without re-scraping, and `data/README.md` documents regeneration.

### 3. Parent cleanup as an isolated commit
`git rm -r --cached` removed only brownlow paths from the parent index on a dedicated branch, leaving the 114 unrelated uncommitted changes in the parent untouched. The branch is local only and still needs to be merged.

### 4. 2026 branch created from main
`feat/2026-prediction` was created and pushed immediately, so season work can start without touching `main`.

## Blockers / Follow-ups

- Parent branch `agent/untrack-brownlow-standalone` is not yet pushed or merged into `master`; until merged, the parent repo still tracks brownlow in `master`'s tree.
- Notebooks still reference the legacy hardcoded path `C:\Users\mrjay\python\jupyter_notebooks\Projects\brownlow\data\` — fix on the 2026 branch.
- `save_models = False` in the v2 notebook means only 10 of 100 ensemble members are persisted.
- The 2026 branch is currently identical to `main`; update `test_year`/season logic and data extracts as part of the 2026 work.
