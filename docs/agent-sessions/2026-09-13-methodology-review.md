# Session: Methodology Review Implementation

**Date:** 2026-09-13
**Branch:** `agent/methodology-review`
**Goal:** Implement the statistics professor's review (probability audit, fully rolling validation, joint calibration, historical player effects, recent-era experiments, eligibility) and then integrate a newly discovered AFL Coaches Association dataset before finalising the model, forecasts and documentation.

## Commits

- `b07b1bf` test(pl): audit marginals against explicit formulas and simulation
- `577ebb9` fix(pl): keep marginals finite when sharp temperatures underflow
- `f0c8662` feat(model): select boosting rounds on time-ordered season folds
- `a185ac1` feat(validation): joint tau/sigma calibration and fully rolling backtest
- `04b32ca` feat(history): partially pooled historical player effects
- `ff892df` feat(eligibility): apply verified award-stage suspension rules
- `e05aa35` feat(model): add recency, CBA-ablation and Plackett-Luce training variants
- `972109f`, `b613c9b` feat(coaches): AFLCA per-match votes scraper and integration
- `66d8f82` feat(model): explicit training-device override with CPU default
- `7d115bf` feat(web): mark ineligible players and update forecast copy
- `87a4036` docs: rewrite README, methodology and 2026 forecast around the final model

## Review Items and Outcomes

1. **Probability audit.** `pl.pl_marginals` matches the explicit sequential sums to 1e-16 and Gumbel-max Monte Carlo within noise; each vote slot sums to 1 and expected votes to 6. The audit found a real bug: underflow at sharp temperatures produced NaN (fixed with zero-denominator handling).
2. **Temperature reconciliation.** The 8.5–9.6 → 7.5–7.6 nat claim mixed the old pseudo-Huber baseline with the ranking model. On identical ranking scores, fitting τ moves NLL by only −0.20 to +0.29 nats, inside the review's rescaling bound; the gain belongs to the score generator. Documentation corrected.
3. **Fully rolling validation.** Inner `GroupKFold` replaced by time-ordered season folds; every target season now trains on earlier labels, selects τ/σ/α on earlier out-of-sample seasons, and is forecast once. Contenders are defined by simulated mean at forecast time; season-by-season coverage and width are reported.
4. **Joint (τ, σ) calibration.** Match NLL is integrated over the same persistent effects the simulator draws; the predeclared equal-weight z-combination with contender CRPS selects production `τ=0.7, σ=0.5`. Integrating the effects removed most of the old match/season tension.
5. **Historical player effects.** Partially pooled residuals (observed minus Plackett–Luce expectation), shrunken and optionally decayed. Rolling selection improved contender CRPS in 7/8 seasons (mean −0.16 on the coaches model); production 100 games / mapping 4 / no decay.
6. **Recent-era experiments.** Recency window (6 seasons), recency weighting (half-life 4), CBA ablation and a custom Plackett–Luce XGBoost objective were all trained and rolled; only the coaches' votes variant showed a clear improvement.
7. **Eligibility.** Ineligible leading vote-getters 2012–2025 extracted from the Wikipedia Brownlow articles, and 32 in-season 2026 suspensions from the Zero Hanger MRO tracker. Suspended players keep votes but are excluded from win determination; the web app marks them.

## New Data Source: AFL Coaches Association

- Per-match combined panel votes (0–10 per player), 2004+; scraped 2012–2026: **19,559 rows**.
- Resolution by season + date + club with suffix/nickname/alias handling: **19,559/19,559 matched, every match sums to 30**.
- Used as `COACH_VOTES` and `COACH_VOTES_SHARE` features. The coaches-augmented model won the variant comparison (contender CRPS 2.85 vs 3.17 baseline; favourite won 2/5 vs 1/5).

## Final Forecast (2026)

Post-round-24 conditional, 10,000 simulations, 32 suspended players excluded:

| Player | Exp. votes | 90% interval | First-or-joint |
|---|---|---|---|
| Nick Daicos | 45.4 | 37–53 | 93.0% (82–99% across specs) |
| Bailey Smith | 31.7 | 22–42 | 4.5% |
| Patrick Cripps | 30.5 | 20–40 | 3.4% |
| Marcus Bontempelli | 29.7 | 22–37 | 1.0% |

## Key Decisions

1. **Coaches-augmented LambdaMART is the production model** (`ranking_coaches`), with joint calibration and historical effects; all other variants are retained as documented experiments.
2. **CPU stays the training default** (`BROWNLOW_XGB_DEVICE` overrides). GPU hist benchmarked at parity (41.6s vs 42.3s per season) and changes best-rounds, so CPU keeps cached scores consistent.
3. **Eligibility sources are committed and sourced**, not inferred; the 2026 list is a dated snapshot that must be refreshed after each tribunal sitting.
4. **The forecast boundary is post-season**: match statistics and coaches' votes observed, only umpire votes simulated. This is stated in the README, methodology and web footer.

## Commands Executed (selection)

- `baseline --model ranking_coaches --seasons 2015-2026`
- `calibrate-joint --model ranking_coaches --seasons 2015-2025 --tune-seasons 2015-2025`
- `player-effects --model ranking_coaches --seasons 2015-2025`
- `calibrate-joint` / `rolling-backtest` for `ranking`, `ranking_recent`, `ranking_weighted`, `ranking_nocba`, `ranking_pl`, `ranking_coaches`
- `forecast --model ranking_coaches --season 2026 --n-sims 10000 --sensitivity-scales 0.3,0.5,0.7 --web-json web/public/forecast_2026.json`
- `fetch-coaches --seasons 2012-2026`
- `uv run pytest` (80 passed), `uv run ruff check .` (clean)

## Blockers / Follow-ups

- Combine the Plackett–Luce objective with the coaches' features (best match NLL plus the human signal).
- A live in-season variant using only information available at each round.
- Match-block bootstrap ensemble for model-parameter uncertainty.
- Refresh `data/eligibility/suspensions_2026.csv` after any further tribunal sittings.
- The 2026 count is the only outstanding out-of-sample target.
