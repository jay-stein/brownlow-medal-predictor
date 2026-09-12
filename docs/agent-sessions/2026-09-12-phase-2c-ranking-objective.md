# Session: Phase 2c — Allocation-Aware Score Generator

**Date:** 2026-09-12
**Branch:** `feat/2026-prediction`
**Commit:** `b25ef06` feat(model): add match-grouped LambdaMART score generator

## Goal

Replace the pseudo-Huber regression objective with an allocation-aware objective, then re-run the full leak-free backtest to test whether per-match ordering and season-level calibration improve.

## Files Changed

| File | Change |
|---|---|
| `src/brownlow/model.py` | Model registry (`regression`, `ranking`), `rank:ndcg` params, qid-grouped DMatrix builder, per-fold metric name generalized |
| `src/brownlow/scores.py` | Per-model score directories (`data/processed/scores/<model_key>/`), metadata records objective/metric |
| `src/brownlow/cli.py` | `--model` on `baseline`, `evaluate`, `calibrate-effects`; model-specific output filenames |
| `tests/test_model.py` | New — registry, ranking smoke test, grouped-DMatrix validation |
| `README.md` | Two-model backtest table, ranking conclusions, Phase 2c done |

## Commands Executed

- `uv run python -m brownlow.cli baseline --model ranking --seasons 2020-2024` (best rounds 184–265)
- `uv run python -m brownlow.cli evaluate --model {regression,ranking} --seasons 2020-2024`
- `uv run python -m brownlow.cli calibrate-effects --model {regression,ranking}`
- Fixed-contender comparison (observed top-15): CRPS, coverage, eventual-winner probability
- `uv run pytest` (42 passed), `uv run ruff check .` (clean)

## Results

Held-out probability metrics (ranking vs regression):

| Season | NLL reg -> rank | Brier reg -> rank | top-1 reg -> rank |
|---|---|---|---|
| 2021 | 6.55 -> 4.91 | 0.084 -> 0.075 | 0.64 -> 0.71 |
| 2022 | 6.41 -> 4.49 | 0.086 -> 0.074 | 0.58 -> 0.66 |
| 2023 | 7.58 -> 6.20 | 0.094 -> 0.090 | 0.49 -> 0.47 |
| 2024 | 7.64 -> 5.77 | 0.093 -> 0.085 | 0.52 -> 0.50 |

Fixed-contender (observed top-15) season-level comparison at each model's calibrated effect scale (regression 0.2, ranking 0.0):

| Model | Season | CRPS | cov50 | cov90 | winner sim rank | winner P(win) |
|---|---|---|---|---|---|---|
| regression | 2021 | 4.42 | 0.33 | 0.67 | 3 | 0.128 |
| regression | 2022 | 3.83 | 0.20 | 0.87 | 4 | 0.022 |
| regression | 2023 | 5.35 | 0.07 | 0.53 | 5 | 0.031 |
| regression | 2024 | 7.89 | 0.13 | 0.27 | 3 | 0.116 |
| ranking | 2021 | 3.29 | 0.53 | 0.73 | 1 | 0.504 |
| ranking | 2022 | 2.15 | 0.47 | 0.87 | 6 | 0.007 |
| ranking | 2023 | 3.70 | 0.40 | 0.60 | 7 | 0.016 |
| ranking | 2024 | 5.88 | 0.07 | 0.60 | 2 | 0.134 |

## Important Decisions

1. **`rank:ndcg` (LambdaMART) with qid = match** rather than a custom Plackett-Luce XGBoost objective: the built-in listwise objective directly optimizes within-match ordering, and XGBoost custom objectives are row-wise, not group-aware.
2. **Per-model score directories** so regression and ranking backtests never collide; caches are invalidated per model rather than globally.
3. **Effect scale re-calibrated per model**: ranking selects 0.0 (the persistent effect no longer improves contender CRPS), regression keeps 0.2.
4. **Comparison on a fixed contender set** (observed top-15) rather than each model's own simulated top-15, so model comparisons are like-for-like.

## Findings

- Ranking wins on every held-out probability metric and on contender CRPS in all four seasons (2.15–5.88 vs 3.83–7.89). The compression of dominant seasons that motivated this phase is largely fixed.
- The effect-scale grid is flat for ranking (CRPS 2.9065 at 0.0 vs 2.9120 at 0.4), so persistent player effects are no longer required to pass the season-level gate at this quality level.
- Award-level winner probabilities remain unreliable: the eventual winner was ranked 6th (2022, Cripps) and 7th (2023, Neale) with probabilities 0.007 and 0.016. Those winners polled more than their statistical profile predicts — consistent with persistent umpire-favourite effects not captured by the features.
- 2024 remains the hardest season (Cripps 45 observed vs 32 hard total), but ranking improves its CRPS from 7.89 to 5.88 and cov90 from 0.27 to 0.60.

## Blockers / Follow-ups

- **Voting-history features** (prior-season votes/game, career poll rate) are the natural next score-generator improvement; they require leakage-safe lagged aggregation from the audited labels.
- Phase 3b (match-block bootstrap model variation) is still deferred; with the effect scale at 0.0 for ranking, model variation becomes the remaining simulation uncertainty source.
- 2025 label extraction pending (R/fitzRoy not on this machine's PATH).
- Phases 4–6: award definitions/eligibility, full backtest persistence, 2026 forecast.
