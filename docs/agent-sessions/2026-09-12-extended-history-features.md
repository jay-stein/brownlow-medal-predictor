# Session: Extended History (2012+) and New Features

**Date:** 2026-09-12
**Branch:** `feat/2026-prediction`
**Commit:** `f175289` feat(data): extend history to 2012 and add CBA and height features

## Goal

Extend the training history via fitzRoy and evaluate three proposed features: centre bounce attendances, player height relative to position, and fourth-quarter disposals.

## Data Availability Findings

- R 4.3.1 and fitzRoy 1.6.0 were already installed (just not on PATH).
- The AFL API (Champion Data source) covers **2012–2026**; AFL Tables Brownlow votes cover **2012–2025**.
- **Centre bounce attendances** (`extendedStats.centreBounceAttendances`) exist only from **2021** (100% coverage 2021–2026, absent before).
- **Height** is available from the AFL squad endpoint (`fetch_squad_afl`), keyed by Champion Data `providerId`, with 7 position groups; full coverage 2012–2026.
- **Fourth-quarter disposals are not available**: the AFL API `playerStats/match/{id}` returns totals only and ignores period query parameters. A different source would be required.

## Files Changed

| File | Change |
|---|---|
| `R/extract_fitzroy.R` | New — player stats and results 2012–2026, AFL Tables votes 2012–2025 |
| `R/extract_squads.R` | New — 270 squad fetches (18 teams × 15 seasons), height/position per player-season |
| `R/fitzroy_data_extract.Rmd` | Reduced to a pointer to the new scripts |
| `src/brownlow/paths.py` | New file names plus `PLAYER_DETAILS_CSV` |
| `src/brownlow/features.py` | CBA added to raw/proportion features; height and position-relative height; 124 features; native-missing handling |
| `src/brownlow/ingest.py` | `load_player_details`; robust duplicate `SEASON` handling |
| `src/brownlow/folds.py` | Development folds 2015–2024, final 2025, production 2012–2025 |
| `src/brownlow/simulate.py` | Drop rows with missing player ids |
| `src/brownlow/cli.py` | Audit loads player details |
| `tests/test_features.py` | Updated feature counts; height and native-missing tests |
| `README.md`, `data/README.md` | Extended results, scripts and file maps |

## Commands Executed

- `Rscript R/extract_fitzroy.R` (9 min), `Rscript R/extract_squads.R` (2 min)
- `uv run python -m brownlow.cli audit`
- `uv run python -m brownlow.cli baseline --model ranking --seasons 2013-2025` (~9 min total)
- `uv run python -m brownlow.cli evaluate --model ranking --seasons 2015-2025`
- `uv run python -m brownlow.cli calibrate-effects --model ranking --seasons 2015-2025 --tune-seasons 2015-2023`
- Winner/spread diagnostics across all 11 evaluation seasons
- `uv run pytest` (44 passed), `uv run ruff check .` (clean)

## Results

Data audit (2012–2025): 1,824 players crosswalked (1,769 by name, 55 by surname/team fallback), **zero unmatched or ambiguous**, all fixtures resolved, voted = 3 × matches every season. Quarantined rows: 1 (2015), 5 (2024), 83 (2025).

Ranking model backtest (leak-free tau 0.75–0.83), 2015–2025:

- Allocation NLL 4.36–6.16; skill vs uniform 0.46–0.62; top-1 0.51–0.67; top-3 slot overlap 0.60–0.70.
- Untouched final fold 2025: NLL 5.50, skill 0.52, top-1 0.51.

Season spread calibration (effect scale 0.4, contender set = observed top-15):

| Metric | Result |
|---|---|
| Contender 90% coverage | mean 0.885 (nominal 0.90; range 0.73–1.00) |
| Contender 50% coverage | mean 0.486 (nominal 0.50) |
| Simulated-vs-observed Spearman | 0.73–0.78 every season |
| Favourite won | 5 of 11 seasons |
| Eventual winner mean P(win) | 0.36 (range 0.02–0.94) |

With the extended history the effect scale moved back to **0.4** (contender CRPS 2.609 vs 2.646 at 0.0), and 2024 contender 90% coverage improved from 0.40 (7-season run) to 0.93.

## Important Decisions

1. **CBA is a native-missing feature**: kept in the feature list with values missing pre-2021 and proportion features prevented from zero-filling, so XGBoost handles the era rather than the pipeline pretending they are zero.
2. **Height relative to same-position peers within the match** (peer mean excluding self, falling back to the match mean) rather than raw height alone.
3. **Folds re-based on 2012**: ten development seasons (2015–2024), 2025 reserved as the untouched final evaluation and now reported once.
4. **Effect scale re-calibrated on the extended data** (0.4); the grid is flat between 0.0 and 0.4, but coverage improves monotonically with scale.

## Blockers / Follow-ups

- **2026 forecast is ready**: 2026 player stats and results are extracted; run `baseline --model ranking --seasons 2026` then simulate with the calibrated tau and effect scale (no `forecast` CLI command yet).
- **Fourth-quarter disposals** need a different data source (Footywire/AFL Tables quarter splits are not in fitzRoy's AFL API).
- The regression baseline has not been re-run on the extended data (ranking is the production candidate).
- Voting-history features and Phase 3b (match-block bootstrap) remain open.
