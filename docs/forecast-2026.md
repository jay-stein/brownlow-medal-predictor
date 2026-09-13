# 2026 Brownlow Forecast

**Generated:** 2026-09-13, post-round-24 conditional forecast (match statistics **and** AFL Coaches Association votes are observed; only the hidden umpire votes are simulated)
**Model:** `ranking_coaches` - match-grouped LambdaMART (`rank:ndcg`) on 124 match-stat features plus the coaches' per-match panel votes, trained on 2012-2025 labels (123,087 player-games, 230 boosting rounds for 2026)
**Calibration:** temperature `tau = 0.7` and persistent-effect scale `sigma = 0.5`, selected jointly on effect-integrated match NLL and contender CRPS over 2015-2025 out-of-sample seasons
**Historical player effects:** shrinkage 100 games, mapping 4, no recency decay (selected on 2015-2025)
**Eligibility:** 32 players suspended during the 2026 home-and-away season are excluded from the medal; their votes still count (see below)
**Simulations:** 10,000 Plackett-Luce season draws
**Raw outputs:** `data/processed/forecast/forecast_2026_ranking_coaches.csv`
**Interactive version:** [`web/`](../web) and <https://jay-stein.github.io/brownlow-medal-predictor/>

| Player | Team | Exp. votes | 90% interval | P(outright) | P(first or joint) | P(top 5) | Spec sensitivity (first or joint) |
|---|---|---|---|---|---|---|---|
| Nick Daicos | Collingwood | 45.4 | 37-53 | 90.9% | **93.0%** | 99.9% | 82.2-98.6% |
| Bailey Smith | Geelong Cats | 31.7 | 22-42 | 3.3% | 4.5% | 82.0% | 1.2-8.6% |
| Patrick Cripps | Carlton | 30.5 | 20-40 | 2.5% | 3.4% | 75.9% | 0.7-6.6% |
| Marcus Bontempelli | Western Bulldogs | 29.7 | 22-37 | 0.7% | 1.0% | 76.3% | 0.2-2.8% |
| Will Ashcroft | Brisbane Lions | 27.0 | 20-33 | 0.1% | 0.2% | 56.6% | 0.0-0.6% |
| Jai Newcombe | Hawthorn | 23.2 | 16-31 | 0.1% | 0.1% | 21.8% | 0.0-0.8% |
| Isaac Heeney | Sydney Swans | 23.0 | 16-31 | 0.0% | 0.0% | 22.0% | 0.0-0.3% |
| Jordan Dawson | Adelaide Crows | 22.9 | 16-30 | 0.0% | 0.0% | 18.8% | 0.0-0.2% |
| Lachie Neale | Brisbane Lions | 22.0 | 14-31 | 0.0% | 0.0% | 18.6% | 0.0-0.0% |
| Jason Horne-Francis | Port Adelaide | 20.2 | 13-28 | - | - | 8.8% | - |

*Jason Horne-Francis is ineligible (suspended in 2026); his simulated votes still affect the count but he cannot win the medal. P(first or joint) for eligible players reflects the next-eligible rule.*

## How to read this

- **Nick Daicos is the clear favourite**: 45.4 expected votes, a 90% interval of 37-53, and a 93.0% first-or-joint probability (82-99% across the 0.3-0.7 effect-scale specifications). Adding the coaches' panel votes to the stats-only model lifted his first-or-joint probability from 80% to 93% and his expected total from 41.0 to 45.4.
- **Bailey Smith and Patrick Cripps are the plausible alternatives**, at 4.5% and 3.4% first-or-joint. The remaining players share the tail.
- **The spread is the point**: even for Daicos the 90% interval spans 16 votes, and a 7% chance remains that someone else wins.

## Calibration context

Rolling out-of-sample backtest (2021-2025; every season's model trained only on earlier labels, calibration and effects selected only on earlier out-of-sample seasons, suspensions applied):

| Season | Selected (tau, sigma) | Contender CRPS | 90% coverage | 50% coverage | Favourite won | Winner's prior P |
|---|---|---|---|---|---|---|
| 2021 | 0.7, 0.4 | 3.05 | 0.87 | 0.47 | yes | 36.2% |
| 2022 | 0.7, 0.4 | 1.80 | 1.00 | 0.73 | no | 11.8% |
| 2023 | 0.7, 0.4 | 2.33 | 0.87 | 0.60 | no | 3.4% |
| 2024 | 0.7, 0.4 | 3.84 | 0.80 | 0.20 | yes | 51.7% |
| 2025 | 0.8, 0.4 | 3.21 | 0.87 | 0.60 | no | 5.6% |
| **Mean** | | **2.85** | **0.88** | **0.52** | **2/5** | **21.7%** |

The model is honest rather than decisive. It is also tested on one count at a time: five rolling seasons and five medal outcomes cannot prove winner-probability calibration. The recalibrated variant comparison (recency windows, CBA ablation, direct Plackett-Luce training) is documented in [`methodology.md`](methodology.md#8-validation-protocol).

## Caveats

- **Post-season forecast boundary.** 2026 match statistics and coaches' votes are observed; only the hidden umpire votes are simulated. This is the appropriate boundary for a count-night forecast, but it is not a live mid-season forecast.
- **Eligibility is applied.** Ineligible players keep their simulated votes (affecting others' totals) but are excluded from outright, joint-first and winner determination.
- **No quarter-level features.** Fourth-quarter disposals are not publicly available (see `data/README.md`).
- **2024-style vote inflation is hard to model.** The 2024 count produced record totals; the model's centre for such seasons is still low even after the historical player effects.

## Reproducing

```bash
uv run python -m brownlow.cli audit                                  # includes the AFLCA coaches votes
uv run python -m brownlow.cli baseline --model ranking_coaches --seasons 2015-2026
uv run python -m brownlow.cli calibrate-joint --model ranking_coaches --seasons 2015-2025 --tune-seasons 2015-2025
uv run python -m brownlow.cli player-effects --model ranking_coaches --seasons 2015-2025
uv run python -m brownlow.cli forecast --model ranking_coaches --season 2026 --n-sims 10000 \
  --sensitivity-scales 0.3,0.5,0.7 --web-json web/public/forecast_2026.json
```

Scrape or refresh the coaches' votes first with `uv run python -m brownlow.cli fetch-coaches --seasons 2012-2026`.
