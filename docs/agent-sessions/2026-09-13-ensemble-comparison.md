# Session: Legacy vs Current vs Ensemble Comparison

**Date:** 2026-09-13
**Branch:** `agent/ensemble`
**Goal:** Compare the original overconfident model, the current production model, and a stacking ensemble on a metric that captures both accuracy and confidence.

## Metric

- **Headline: contender CRPS** over rolling seasons — a proper score that punishes both wrong centres and false confidence.
- **Trust gates: 50% and 90% interval coverage** (nominal 0.50/0.90) plus width.
- **Accuracy: favourite-won count, winner's prior probability**, plus match NLL for allocation models.
- **Composite: rank-sum** across CRPS, coverage deviations and favourite hits.

## What Was Built

1. **`legacy.py`** — a faithful reconstruction from the original notebooks: 100 feature-bagged pseudo-Huber XGBoost regressors on the original 120-feature set (70% feature sample per model), `pred_mean`/`pred_std` from the ensemble spread, independent Normal season counts with no match budget. Rolled with the same time-ordered protocol so the comparison isolates the method.
2. **`ensemble.py`** — loss-based stacking of `ranking_coaches`, `ranking_pl` and a random-forest regressor with coaches' votes; simplex weight grid, τ and σ selected on earlier out-of-sample seasons by the equal-weight z-combination of match NLL and contender CRPS.
3. **`compare-approaches`** CLI command producing the shared comparison table.

### Design correction

The first ensemble standardised member utilities **within each match**, which removed the per-match spread the Plackett–Luce allocation depends on — even a pure member could not reproduce its own model (pure coaches CRPS 4.55 vs 2.85). Switching to **global per-member standardisation** preserved within-match gaps while making members commensurate; a pure member then reproduces its raw model up to a temperature rescale (2021 3.03 vs 3.05; 2024 3.86 vs 3.84). All reported ensemble numbers use the corrected operator.

## Results (rolling 2021–2025, suspensions applied)

| Approach | Contender CRPS | 90% coverage | 50% coverage | Favourite won | Winner prior P | Match NLL | Rank-sum |
|---|---|---|---|---|---|---|---|
| Legacy | 4.77 | 0.20 | 0.09 | 0/5 | 0.000 | n/a | 12 |
| **Current production** | **2.85** | 0.88 | 0.52 | **2/5** | **0.217** | **5.24** | **5** |
| Stacking ensemble | 2.65 | 0.93 | 0.57 | 1/5 | 0.186 | 5.26 | 7 |

- The legacy model is catastrophically overconfident: 20% of 90% intervals covered, no favourite hits, zero winner probability on average — the exact failure the redesign targeted.
- The ensemble has the best vote-distribution calibration (CRPS −7%, coverage 0.93) but dilutes the top pick (1/5 favourites, winner P 0.186).
- A hindsight oracle over the ensemble grid reaches CRPS 2.37, so selection quality — not the member pool — is the binding constraint on the ensemble.

## Decision

**Keep the current `ranking_coaches` production model as the shipped forecast.** It wins the composite rank-sum and retains the sharpest medal signal; the ensemble is documented as the better-calibrated alternative to revisit once more medal outcomes accumulate. No production artefacts changed and the live forecast was not regenerated.

## Commands Executed

- `legacy-roll --seasons 2018-2025` (100-model training per season; 13 min total)
- `ensemble-roll --seasons 2018-2025 --taus 0.3,0.4,0.5,0.6,0.7` (~14 min for 1,800 combinations)
- `compare-approaches`
- `uv run pytest` (86 passed), `uv run ruff check .` (clean)

## Files Changed

- `src/brownlow/legacy.py`, `src/brownlow/ensemble.py` (new)
- `src/brownlow/cli.py` (`legacy-roll`, `ensemble-roll`, `compare-approaches`)
- `src/brownlow/simulate.py` (shared `summarise_totals` / `eligible_mask`)
- `tests/test_legacy.py`, `tests/test_ensemble.py` (new)
- `README.md`, `docs/methodology.md` (comparison write-up)
- `docs/agent-sessions/2026-09-13-ensemble-comparison.md` (this file)

## Blockers / Follow-ups

- Ensemble selection overfits with only five rolling outcomes; add award metrics to the selection criterion once more counts are available.
- Combine the Plackett–Luce objective with coaches' votes (currently the PL member has no coaches).
- 2026 count remains the only live out-of-sample target.
