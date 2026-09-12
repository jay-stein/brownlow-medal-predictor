# Session: Phase 2b Backtests + Phase 3a Season Simulator

**Date:** 2026-09-12
**Branch:** `feat/2026-prediction`
**Commits:** `eb0757a` feat(backtest): add leak-free allocation evaluation and season simulator; plus a small README roadmap fix

## Goal

Complete Phase 2b (per-season out-of-sample scores, leak-free temperature fitting, allocation probability backtests) and Phase 3a (persistent player-season effect, shared Monte Carlo simulator, empirical effect-scale calibration) of the Brownlow redesign.

## Files Changed

| File | Change |
|---|---|
| `src/brownlow/scores.py` | New — per-season score generation with chronological training, caching under `data/processed/scores/`, `ROUND_YEAR` in persisted keys |
| `src/brownlow/evaluate.py` | New metrics — leak-free taus, allocation NLL, multiclass Brier, top-k ranking, reliability, CRPS, interval coverage |
| `src/brownlow/simulate.py` | New — persistent player-season effect simulator, season metrics, effect-scale grid |
| `src/brownlow/cli.py` | `baseline`, `evaluate`, `calibrate-effects` commands |
| `tests/` | 18 new tests (39 total) for CRPS, coverage, leak-free protocol, simulator invariants, grid |
| `README.md` | Backtest results table, quick-start backtest commands, phase status |

## Commands Executed

- `uv run python -m brownlow.cli baseline --seasons 2020-2024 --force` (per-season training, ~1 min total)
- `uv run python -m brownlow.cli evaluate --seasons 2020-2024`
- `uv run python -m brownlow.cli calibrate-effects --seasons 2021-2024 --tune-seasons 2021-2023`
- `uv run pytest` (39 passed) and `uv run ruff check .` (clean)
- Ad-hoc diagnostics: hard 3/2/1 totals vs observed, tau-multiplier grid, joint tau x effect grid

## Results

Chronological backtest (tau fitted only on earlier out-of-sample seasons):

| Season | tau | NLL | NLL tau=1 | Brier | top-1 | top-3 overlap |
|---|---|---|---|---|---|---|
| 2021 | 0.264 | 6.55 | 8.53 | 0.084 | 0.64 | 0.67 |
| 2022 | 0.310 | 6.41 | 8.62 | 0.086 | 0.58 | 0.70 |
| 2023 | 0.314 | 7.58 | 8.90 | 0.094 | 0.49 | 0.60 |
| 2024 | 0.339 | 7.64 | 8.90 | 0.093 | 0.52 | 0.60 |

- Leak-free tau nearly matches the per-season oracle and cuts NLL by 1.3–2.0 nats versus unfitted scores; Brier improves from ~0.106 to 0.084–0.094.
- Season simulator with one persistent player effect: selected scale 0.2 by contender CRPS on 2021–2023; contender CRPS (3.06–3.50) beats the hard 3/2/1 point-forecast MAE (4.07–6.20).
- Contender 90% interval coverage at the selected scale: 0.80 / 0.93 / 0.73 / 0.40 (2021–2024) — under-covering, especially 2024.

## Important Decisions

1. **Leak-free tau protocol**: scores for season Y come from a model trained only on seasons < Y; tau for Y is fitted on pooled out-of-sample matches from cached seasons < Y. Season 2020 is a tau source with no prior (tau=1 fallback).
2. **Effect scale selected by contender CRPS**, not overall CRPS, because the award is decided by the leading players. The grid shows 0.2 optimal among {0, 0.02, 0.05, 0.1, 0.2, 0.4}.
3. **The effect is drawn once per player per simulation** and held across the season; drawing per match would average away the persistent uncertainty.
4. **Score caching** makes backtests reproducible and cheap; `--force` retrains.
5. **Fixed model variation in v1**: the simulator uses the single baseline score generator; the match-block bootstrap ensemble is deferred to Phase 3b per the minimal-v1 plan.

## Findings and Diagnostics

- Hard 3/2/1 totals from the same scores are much closer to observations than PL simulated means (2024: Daicos 37 vs 38 observed, Cripps 31 vs 45; simulated means 29.9 and 23.4). The probabilistic allocation smooths the top end.
- Joint tau x effect grid shows a genuine tension: match NLL prefers the soft fitted tau (6.85 mean NLL at 1.0x) while season CRPS prefers sharper (3.06 at 0.6x), and intervals widen with the effect scale. This is the signature of missing overdispersion/persistent concentration rather than a fixable width problem.
- 2024 is the hardest season (record polling by the winner); no zero-mean effect can fix a centre that is 20 votes low.

## Blockers / Follow-ups

- **Score quality**: replace the pseudo-Huber objective with an allocation-aware objective (PL likelihood or a match-grouped ranking objective) before adding model-variation complexity — this is the planned Phase 2.3 and the diagnostics support it.
- **Sharpness trade-off**: decide whether to keep the match-likelihood tau (calibrated per-match probabilities) or add a documented season-level sharpness calibration.
- Phase 3b: match-block bootstrap ensemble for model variation.
- 2025 label extraction still pending — R/fitzRoy is not on this machine's PATH; the R script needs a run on a machine with R.
- Phases 4–6: award definitions/eligibility, full backtest persistence, 2026 forecast.
