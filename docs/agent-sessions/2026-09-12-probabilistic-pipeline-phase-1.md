# Session: Probabilistic Pipeline — Phase 1 + Plackett-Luce Core

**Date:** 2026-09-12
**Branch:** `feat/2026-prediction`
**Commits:** `9328cb0` feat(pipeline): audited data layer, chronological folds and Plackett-Luce core; `7dc7e7b` docs: document uv workflow and redesign phase status

## Goal

Implement Phases 0–2 of the Brownlow model redesign: a uv-managed package skeleton, a repaired data pipeline with audited labels, chronological match-grouped evaluation, explicit CV round selection, and the Plackett–Luce allocation core — validated end to end on a development fold.

## Files Changed

| File | Change |
|---|---|
| `pyproject.toml`, `.python-version`, `uv.lock` | New uv-managed project (Python 3.12; xgboost 3.4.1, scikit-learn 1.9.1) |
| `src/brownlow/naming.py` | Punctuation-insensitive compact name keys, team aliases, generational-suffix stripping |
| `src/brownlow/features.py` | 120-feature engineering ported verbatim from the v2 notebook (verified identical) |
| `src/brownlow/ingest.py` | Two-layer player crosswalk and three-state labels (`voted` / `zero` / `unresolved`) |
| `src/brownlow/folds.py` | Chronological match-grouped folds (dev 2021–2024, final untouched 2025) |
| `src/brownlow/model.py` | Baseline XGBoost with explicit match-grouped CV round selection |
| `src/brownlow/pl.py` | Plackett–Luce NLL, temperature fitting, exact `P(0/1/2/3)` marginals |
| `src/brownlow/cli.py` | `uv run python -m brownlow.cli audit` |
| `tests/` | 21 unit tests (naming, crosswalk, features, folds, Plackett–Luce) |
| `.gitignore` | Ignore `data/processed/` |
| `README.md` | uv workflow, redesign status, updated repository structure |
| `requirements.txt` | Removed (superseded by `pyproject.toml` + `uv.lock`) |

## Commands Executed

- `uv sync` — created the Python 3.12 environment
- `uv run pytest` — 21 passed
- `uv run ruff check .` — clean (legacy `notebooks/` and `archive/` excluded)
- `uv run python -m brownlow.cli audit` — crosswalk, label audit, audit CSVs and parquet
- Smoke test on the `dev_2024` fold: `folds.split_by_season` → `FeaturePreprocessor` → `model.select_best_round` (4-fold grouped CV, 200 max rounds) → `fit_utilities` → `pl.fit_tau`
- Git: two commits on `feat/2026-prediction`, pushed to origin

## Results

- **Crosswalk**: 1,166 players accepted by exact name, 19 by the surname/team/date fallback; 0 ambiguous, 0 unmatched.
- **Labels**: all 1,359 fixtures resolved; `voted` = 3 × matches for every season (2018: 594 … 2024: 621). Only 5 genuinely missing AFL Tables rows quarantined (all 2024); 2025 is unlabelled by design.
- **Smoke test (2024 fold, reduced rounds)**: fold best iterations 86–104 (selected 94), fold MAE 0.116–0.119, fitted `tau = 0.41`, mean allocation NLL 8.90 → 7.52 per match, top-3 slot overlap 60.2%.

## Important Decisions

1. **Crosswalk repair without fuzzy matching**: compact keys remove punctuation/accents/spaces and strip generational suffixes (`Jr`, `Jnr`, `II`–`IV`); unresolved identities then use a surname + date + team fallback scored by first-name similarity. This resolved every previously unmatched player (`ALWYN DAVEY JNR`, `ROBERT HANSEN JR`, `THOMAS BERRY`, `BAILEY J. WILLIAMS`, all the `O'…` names).
2. **Team agreement is required for layer-1 acceptance.** A single same-name candidate on the wrong team is deferred rather than accepted, preventing wrong-player joins (Sam Reid, Bailey Williams).
3. **Conservative zero labels.** A match only establishes genuine zeros when all three of its vote recipients map to stats rows; otherwise the whole fixture is quarantined. Five 2024 player-games with no AFL Tables row remain `unresolved`.
4. **Explicit round selection.** `model.select_best_round` runs GroupKFold over matches and takes the mean best iteration, replacing the ambiguous `len(cv_results)` logic.
5. **`tau` fitting is not yet leak-free.** The smoke fitted temperature on the evaluation season itself (8.90 → 7.52 NLL); Phase 2b must fit it on earlier-fold out-of-sample predictions before evaluating 2024.

## Blockers / Follow-ups

- **Phase 2b**: per-fold frozen scores, leak-free temperature fitting, allocation log-loss backtest.
- **Phase 3**: player-season effect scale (empirical grid), match-block bootstrap ensemble, one shared simulator.
- **Phase 4**: eligibility flag from the AFL API Brownlow endpoint, applied at the award stage.
- Five quarantined 2024 player-games are data-source gaps to monitor.
- Legacy notebooks still contain hardcoded absolute data paths.
