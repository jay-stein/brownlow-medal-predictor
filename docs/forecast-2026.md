# 2026 Brownlow Forecast

**Generated:** 2026-09-20, post-round-24 conditional forecast (match statistics, AFL Coaches Association votes and season form are observed; only the hidden umpire votes are simulated)
**Model:** `ranking_season` - match-grouped LambdaMART (`rank:ndcg`) on 139 features: match statistics, the coaches' per-match panel votes, and leave-one-game-out season aggregates. Trained on 2012-2025 labels (123,087 player-games, 221 boosting rounds for 2026)
**Calibration:** temperature `tau = 0.8` and persistent-effect scale `sigma = 0.4`, selected jointly on effect-integrated match NLL and contender CRPS over 2015-2025 out-of-sample seasons; persistent effects follow a variance-normalised Student-t distribution with 4 degrees of freedom
**Historical player effects:** shrinkage 100 games, mapping 4, no recency decay (selected on 2015-2025)
**Eligibility:** 32 players suspended during the 2026 home-and-away season are excluded from the medal; their votes still count
**Simulations:** 10,000 Plackett-Luce season draws
**Raw outputs:** `data/processed/forecast/forecast_2026_ranking_season.csv`
**Interactive version:** [`web/`](../web) and <https://jay-stein.github.io/brownlow-medal-predictor/>

| Player | Team | Exp. votes | 90% interval | P(outright) | P(first or joint) | P(top 5) | Spec sensitivity (first or joint) |
|---|---|---|---|---|---|---|---|
| Nick Daicos | Collingwood | 43.7 | 36-51 | 88.2% | **90.3%** | 99.6% | 70.0-94.7% |
| Bailey Smith | Geelong Cats | 31.9 | 24-41 | 3.7% | 4.9% | 85.6% | 3.4-8.5% |
| Marcus Bontempelli | Western Bulldogs | 30.2 | 23-37 | 0.8% | 1.2% | 80.9% | 0.6-3.1% |
| Patrick Cripps | Carlton | 30.1 | 21-39 | 2.6% | 3.2% | 73.8% | 2.0-6.8% |
| Will Ashcroft | Brisbane Lions | 27.5 | 21-34 | 0.3% | 0.5% | 59.4% | 0.2-1.5% |
| Jai Newcombe | Hawthorn | 24.8 | 18-33 | 0.4% | 0.5% | 31.2% | 0.2-1.7% |
| Jordan Dawson | Adelaide Crows | 23.0 | 17-30 | 0.0% | 0.1% | 16.2% | 0.0-0.6% |
| Isaac Heeney | Sydney Swans | 22.6 | 16-30 | 0.1% | 0.1% | 15.5% | 0.0-0.6% |
| Lachie Neale | Brisbane Lions | 21.8 | 15-29 | 0.1% | 0.2% | 14.2% | 0.1-0.7% |
| Max Gawn | Melbourne | 21.0 | 12-31 | 0.3% | 0.4% | 14.3% | 0.2-1.6% |

*Jason Horne-Francis (19.6 expected votes) is ineligible after a 2026 suspension; 32 players in total are excluded from the medal while their simulated votes still affect the count.*

## How to read this

- **Nick Daicos is the clear favourite**: 43.7 expected votes, a 90% interval of 36-51, and a 90.3% first-or-joint probability (70-95% across the 0.3-0.7 effect-scale specifications).
- **The season-form model is slightly less certain than the coaches-only model was** (90.3% versus 93.0%) but better calibrated on rolling seasons (contender CRPS 2.63 versus 2.85). That is the intended trade: sharper where the evidence is strong, wider where sustained form is volatile.
- **Bailey Smith and Patrick Cripps are the plausible alternatives**, at 4.9% and 3.2% first-or-joint. The remaining players share the tail.
- **The spread is the point**: even for Daicos the 90% interval spans 15 votes, and roughly a 10% chance remains that someone else wins.

## Calibration context

Rolling out-of-sample backtest (2021-2025; every season's model trained only on earlier labels, calibration and effects selected only on earlier out-of-sample seasons, suspensions applied):

| Season | Selected (tau, sigma) | Contender CRPS | 90% coverage | 50% coverage | Favourite won | Winner's prior P |
|---|---|---|---|---|---|---|
| 2021 | 0.7, 0.4 | 2.78 | 0.93 | 0.47 | yes | 44.0% |
| 2022 | 0.7, 0.5 | 1.87 | 1.00 | 0.73 | no | 17.0% |
| 2023 | 0.8, 0.3 | 2.05 | 0.93 | 0.53 | no | 1.9% |
| 2024 | 0.8, 0.3 | 3.79 | 0.73 | 0.13 | yes | 49.5% |
| 2025 | 0.8, 0.4 | 2.65 | 0.93 | 0.73 | no | 3.8% |
| **Mean** | | **2.63** | **0.91** | **0.52** | **2/5** | **23.2%** |

The model is honest rather than decisive. Five rolling seasons and five medal outcomes cannot prove winner-probability calibration; the variant comparison (coaches, season form, context, Plackett-Luce, ensemble) is documented in [`methodology.md`](methodology.md).

## Caveats

- **Post-season forecast boundary.** 2026 match statistics, coaches' votes and season aggregates are observed; only the hidden umpire votes are simulated. This is the appropriate boundary for a count-night forecast, but it is not a live mid-season forecast.
- **Eligibility is applied.** Ineligible players keep their simulated votes (affecting others' totals) but are excluded from outright, joint-first and winner determination.
- **Heavy-tailed effects are an empirical choice.** The Student-t(4) persistent-effect shape was adopted on rolling 2021-2025 CRPS and coverage; it allows breakout seasons at the right frequency but is not a structural model of why they happen.
- **No quarter-level features.** Fourth-quarter disposals are not publicly available (see `data/README.md`).
- **2024-style vote inflation is hard to model.** The 2024 count produced record totals; the model's centre for such seasons is still low even after the historical player effects and heavy-tailed season effects.

## Reproducing

```bash
uv run python -m brownlow.cli audit                                    # includes the AFLCA coaches votes
uv run python -m brownlow.cli baseline --model ranking_season --seasons 2015-2026
uv run python -m brownlow.cli calibrate-joint --model ranking_season --seasons 2015-2025 \
  --tune-seasons 2015-2025 --tag _t4 --effect-distribution student_t --effect-t-df 4
uv run python -m brownlow.cli player-effects --model ranking_season --seasons 2015-2025
uv run python -m brownlow.cli rolling-backtest --model ranking_season --report-seasons 2021-2025 \
  --min-evidence 3 --n-sims 1000 --tag _t4 --effect-distribution student_t --effect-t-df 4
uv run python -m brownlow.cli fetch-reports --season 2026              # official match report excerpts
uv run python -m brownlow.cli forecast --model ranking_season --season 2026 --tag _t4 \
  --effect-distribution student_t --effect-t-df 4 --n-sims 10000 \
  --sensitivity-scales 0.3,0.5,0.7 --web-json web/public/forecast_2026.json
```

Scrape or refresh the coaches' votes first with `uv run python -m brownlow.cli fetch-coaches --seasons 2012-2026`.
