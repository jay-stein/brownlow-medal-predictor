# 2026 Brownlow Forecast

**Generated:** 2026-09-13, post-round-24 conditional forecast (match statistics, AFL Coaches Association votes and season form are observed; only the hidden umpire votes are simulated)
**Model:** `ranking_season` - match-grouped LambdaMART (`rank:ndcg`) on 139 features: match statistics, the coaches' per-match panel votes, and leave-one-game-out season aggregates. Trained on 2012-2025 labels (123,087 player-games, 221 boosting rounds for 2026)
**Calibration:** temperature `tau = 0.8` and persistent-effect scale `sigma = 0.4`, selected jointly on effect-integrated match NLL and contender CRPS over 2015-2025 out-of-sample seasons
**Historical player effects:** shrinkage 100 games, mapping 4, no recency decay (selected on 2015-2025)
**Eligibility:** 32 players suspended during the 2026 home-and-away season are excluded from the medal; their votes still count
**Simulations:** 10,000 Plackett-Luce season draws
**Raw outputs:** `data/processed/forecast/forecast_2026_ranking_season.csv`
**Interactive version:** [`web/`](../web) and <https://jay-stein.github.io/brownlow-medal-predictor/>

| Player | Team | Exp. votes | 90% interval | P(outright) | P(first or joint) | P(top 5) | Spec sensitivity (first or joint) |
|---|---|---|---|---|---|---|---|
| Nick Daicos | Collingwood | 43.9 | 36-52 | 88.7% | **91.6%** | 99.9% | 72.4-95.5% |
| Bailey Smith | Geelong Cats | 32.1 | 23-41 | 4.2% | 5.8% | 84.9% | 3.6-11.5% |
| Patrick Cripps | Carlton | 30.0 | 20-40 | 2.6% | 3.6% | 75.9% | 2.1-8.8% |
| Marcus Bontempelli | Western Bulldogs | 30.0 | 23-37 | 0.8% | 1.4% | 76.3% | 0.6-3.8% |
| Will Ashcroft | Brisbane Lions | 27.5 | 21-34 | 0.2% | 0.3% | 56.6% | 0.2-1.7% |
| Jai Newcombe | Hawthorn | 24.7 | 17-33 | 0.2% | 0.3% | 21.8% | 0.1-1.9% |
| Isaac Heeney | Sydney Swans | 23.0 | 16-31 | 0.0% | 0.0% | 22.0% | 0.0-0.3% |
| Jordan Dawson | Adelaide Crows | 22.9 | 16-30 | 0.0% | 0.0% | 18.8% | 0.0-0.2% |
| Lachie Neale | Brisbane Lions | 22.0 | 14-31 | 0.0% | 0.0% | 18.6% | 0.0-0.0% |
| Jason Horne-Francis | Port Adelaide | 20.2 | 13-28 | - | - | 8.8% | - |

*Jason Horne-Francis is ineligible (suspended in 2026); his simulated votes still affect the count but he cannot win the medal.*

## How to read this

- **Nick Daicos is the clear favourite**: 43.9 expected votes, a 90% interval of 36-52, and a 91.6% first-or-joint probability (72-96% across the 0.3-0.7 effect-scale specifications).
- **The season-form model is slightly less certain than the coaches-only model** (91.6% versus 93.0%) but better calibrated on rolling seasons (contender CRPS 2.62 versus 2.85). That is the intended trade: sharper where the evidence is strong, wider where sustained form is volatile.
- **Bailey Smith and Patrick Cripps are the plausible alternatives**, at 5.8% and 3.6% first-or-joint. The remaining players share the tail.
- **The spread is the point**: even for Daicos the 90% interval spans 16 votes, and roughly an 8% chance remains that someone else wins.

## Calibration context

Rolling out-of-sample backtest (2021-2025; every season's model trained only on earlier labels, calibration and effects selected only on earlier out-of-sample seasons, suspensions applied):

| Season | Selected (tau, sigma) | Contender CRPS | 90% coverage | 50% coverage | Favourite won | Winner's prior P |
|---|---|---|---|---|---|---|
| 2021 | 0.7, 0.4 | 2.78 | 0.93 | 0.53 | yes | 45.5% |
| 2022 | 0.7, 0.4 | 1.89 | 0.93 | 0.67 | no | 17.4% |
| 2023 | 0.7, 0.4 | 2.09 | 1.00 | 0.73 | no | 2.5% |
| 2024 | 0.7, 0.4 | 3.58 | 0.87 | 0.27 | yes | 48.5% |
| 2025 | 0.7, 0.4 | 2.77 | 0.87 | 0.67 | no | 2.6% |
| **Mean** | | **2.62** | **0.92** | **0.57** | **2/5** | **23.3%** |

The model is honest rather than decisive. Five rolling seasons and five medal outcomes cannot prove winner-probability calibration; the variant comparison (coaches, season form, context, Plackett-Luce, ensemble) is documented in [`methodology.md`](methodology.md).

## Caveats

- **Post-season forecast boundary.** 2026 match statistics, coaches' votes and season aggregates are observed; only the hidden umpire votes are simulated. This is the appropriate boundary for a count-night forecast, but it is not a live mid-season forecast.
- **Eligibility is applied.** Ineligible players keep their simulated votes (affecting others' totals) but are excluded from outright, joint-first and winner determination.
- **No quarter-level features.** Fourth-quarter disposals are not publicly available (see `data/README.md`).
- **2024-style vote inflation is hard to model.** The 2024 count produced record totals; the model's centre for such seasons is still low even after the historical player effects.

## Reproducing

```bash
uv run python -m brownlow.cli audit                                    # includes the AFLCA coaches votes
uv run python -m brownlow.cli baseline --model ranking_season --seasons 2015-2026
uv run python -m brownlow.cli calibrate-joint --model ranking_season --seasons 2015-2025 --tune-seasons 2015-2025
uv run python -m brownlow.cli player-effects --model ranking_season --seasons 2015-2025
uv run python -m brownlow.cli fetch-reports --season 2026              # official match report excerpts
uv run python -m brownlow.cli forecast --model ranking_season --season 2026 --n-sims 10000 \
  --sensitivity-scales 0.3,0.5,0.7 --web-json web/public/forecast_2026.json
```

Scrape or refresh the coaches' votes first with `uv run python -m brownlow.cli fetch-coaches --seasons 2012-2026`.
