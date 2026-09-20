# 2026 Brownlow Forecast

**Generated:** 2026-09-20, post-round-24 conditional forecast (match statistics, AFL Coaches Association votes and season form are observed; only the hidden umpire votes are simulated)
**Model:** `ranking_season` - match-grouped LambdaMART (`rank:ndcg`) on 139 features: match statistics, the coaches' per-match panel votes, and leave-one-game-out season aggregates. Trained on 2012-2025 labels (123,087 player-games, 221 boosting rounds for 2026)
**Calibration:** temperature `tau = 0.8` and persistent-effect scale `sigma = 0.4`, selected jointly on effect-integrated match NLL and contender CRPS over 2015-2025 out-of-sample seasons; persistent effects follow a variance-normalised Student-t distribution with 4 degrees of freedom
**Historical player effects:** shrinkage 50 games, mapping 2, no recency decay (selected under Student-t(4) on 2015-2025; the Gaussian selected 100/4)
**Eligibility:** 32 players suspended during the 2026 home-and-away season are excluded from the medal; their votes still count
**Simulations:** 10,000 Plackett-Luce season draws
**Raw outputs:** `data/processed/forecast/forecast_2026_ranking_season.csv`
**Interactive version:** [`web/`](../web) and <https://jay-stein.github.io/brownlow-medal-predictor/>

| Player | Team | Exp. votes | 90% interval | P(outright) | P(first or joint) | P(top 5) | Spec sensitivity (first or joint) |
|---|---|---|---|---|---|---|---|
| Nick Daicos | Collingwood | 43.3 | 35-51 | 88.4% | **90.7%** | 99.6% | 70.1-94.9% |
| Bailey Smith | Geelong Cats | 31.9 | 24-41 | 4.2% | 5.7% | 87.4% | 4.2-9.6% |
| Marcus Bontempelli | Western Bulldogs | 29.8 | 23-36 | 0.9% | 1.5% | 81.6% | 0.7-3.2% |
| Will Ashcroft | Brisbane Lions | 27.6 | 21-34 | 0.4% | 0.6% | 65.1% | 0.2-1.8% |
| Patrick Cripps | Carlton | 27.3 | 18-37 | 1.4% | 1.8% | 58.5% | 0.9-4.6% |
| Jai Newcombe | Hawthorn | 23.7 | 17-31 | 0.3% | 0.4% | 26.6% | 0.2-1.5% |
| Jordan Dawson | Adelaide Crows | 23.1 | 17-30 | 0.0% | 0.1% | 20.1% | 0.0-0.7% |
| Isaac Heeney | Sydney Swans | 22.5 | 16-30 | 0.1% | 0.1% | 17.3% | 0.0-0.6% |
| Lachie Neale | Brisbane Lions | 21.3 | 14-29 | 0.1% | 0.1% | 13.6% | 0.1-0.7% |
| Max Gawn | Melbourne | 20.7 | 12-30 | 0.3% | 0.4% | 14.8% | 0.1-1.6% |

*Jason Horne-Francis (19.6 expected votes) is ineligible after a 2026 suspension; 32 players in total are excluded from the medal while their simulated votes still affect the count.*

## How to read this

- **Nick Daicos is the clear favourite**: 43.3 expected votes, a 90% interval of 35-51, and a 90.7% first-or-joint probability (70-95% across the 0.3-0.7 effect-scale specifications).
- **The season-form model is slightly less certain than the coaches-only model was** (90.7% versus 93.0%) but better calibrated on rolling seasons (contender CRPS 2.55 with effects versus 2.85). That is the intended trade: sharper where the evidence is strong, wider where sustained form is volatile.
- **Bailey Smith is the main alternative** at 5.7% first-or-joint, with Cripps (1.8%) and Bontempelli (1.5%) close behind. The remaining players share the tail.
- **The spread is the point**: even for Daicos the 90% interval spans 16 votes, and roughly a 9% chance remains that someone else wins.

## Calibration context

Rolling backtest (2021-2025; every season's model trained only on earlier labels, calibration and effects selected only on earlier out-of-sample seasons, suspensions applied). These seasons informed several design choices, so this is a **post-selection diagnostic**; a fully automated selection across all candidate families on evidence alone scores 2.65 CRPS with 1/5 favourites, and **2026 is the only untouched target**.

| Season | Selected (tau, sigma) | Contender CRPS | 90% coverage | 50% coverage | Favourite won | Winner's prior P |
|---|---|---|---|---|---|---|
| 2021 | 0.7, 0.4 | 2.78 | 0.93 | 0.47 | yes | 44.0% |
| 2022 | 0.7, 0.5 | 1.87 | 1.00 | 0.73 | no | 17.0% |
| 2023 | 0.8, 0.3 | 2.05 | 0.93 | 0.53 | no | 1.9% |
| 2024 | 0.8, 0.3 | 3.79 | 0.73 | 0.13 | yes | 49.5% |
| 2025 | 0.8, 0.4 | 2.65 | 0.93 | 0.73 | no | 3.8% |
| **Mean** | | **2.63** | **0.91** | **0.52** | **2/5** | **23.2%** |

With the historical player effects applied at the shape-consistent selection (50/2/0), the effect-adjusted diagnostic scores **CRPS 2.55, 25.9% average winner probability, 2/5 favourites** and 96% ninety-interval coverage. The model is honest rather than decisive. Five rolling seasons and five medal outcomes cannot prove winner-probability calibration; the variant comparison (coaches, season form, context, Plackett-Luce, ensemble) is documented in [`methodology.md`](methodology.md).

**Pre-registration.** The production configuration in this document is frozen until the 2026 count: no further feature, shape, calibration or effect-selection changes will be made against 2026. The forecast is the test.

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
uv run python -m brownlow.cli player-effects --model ranking_season --seasons 2015-2025 --tag _t4 \
  --effect-distribution student_t --effect-t-df 4
uv run python -m brownlow.cli rolling-backtest --model ranking_season --report-seasons 2021-2025 \
  --min-evidence 3 --n-sims 1000 --tag _t4 --effect-distribution student_t --effect-t-df 4
uv run python -m brownlow.cli fetch-reports --season 2026              # official match report excerpts
uv run python -m brownlow.cli forecast --model ranking_season --season 2026 --tag _t4 \
  --effect-distribution student_t --effect-t-df 4 --n-sims 10000 \
  --sensitivity-scales 0.3,0.5,0.7 --web-json web/public/forecast_2026.json
```

Scrape or refresh the coaches' votes first with `uv run python -m brownlow.cli fetch-coaches --seasons 2012-2026`.
