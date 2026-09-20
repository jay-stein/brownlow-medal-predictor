# Brownlow Medal Predictor

Predicts AFL Brownlow Medal votes for every game of a season and simulates the count to produce calibrated vote distributions and medal probabilities, including award-stage suspension eligibility.

**Live 2026 forecast:** https://jay-stein.github.io/brownlow-medal-predictor/

[![License: MIT](https://img.shields.io/badge/license-MIT-d4af37.svg)](LICENSE)

The mathematical treatment — Brownlow mechanics, the joint calibration protocol, the uncertainty model and known limitations — is in [`docs/methodology.md`](docs/methodology.md). The 2026 write-up is in [`docs/forecast-2026.md`](docs/forecast-2026.md).

## How it works

1. **Data extraction (R + Python)** — [`R/extract_fitzroy.R`](R/extract_fitzroy.R) pulls AFL API player stats and match results (2012–2026) plus AFL Tables player stats and Brownlow votes (2012–2025) via [`fitzRoy`](https://github.com/jimmyday12/fitzRoy); [`R/extract_squads.R`](R/extract_squads.R) adds height and position; `brownlow.cli fetch-coaches` scrapes the **AFL Coaches Association** per-match panel votes (each panel awards 5-4-3-2-1, so 0–10 per player) from 2012 onwards.
2. **Audited labels** — `ingest.py` builds a Champion Data ↔ AFL Tables crosswalk and attaches three-state labels: `voted`, genuine `zero`, and `unresolved` (quarantined from training, never treated as zero).
3. **Features** — 124 per player-game: raw stats, engineered match context, within-team shares, position, and height. The production model adds the coaches' votes (`COACH_VOTES`, 0–10) and the player's share of their team's match votes, for **126 features**. Centre-bounce attendances (2021+) and a few heights are left natively missing for XGBoost.
4. **Score generator** — a match-grouped LambdaMART ranker (`rank:ndcg`). Boosting rounds are chosen by **time-ordered season validation**; every target season's model trains only on earlier labels. The production model (`ranking_season`) adds the coaches' votes and **leave-one-game-out season form** (per-game stat means, coach-vote totals/rate/rank, team win rate); alternatives evaluated on the same rolling seasons included recent-season windows, recency weighting, a CBA ablation, a direct Plackett–Luce objective and Elo/travel context features. The season-form model won at contender CRPS 2.62 versus 3.17 for the all-history baseline.
5. **Allocation and calibration** — `pl.py` turns utilities into the ordered 3-2-1 distribution with exact `P(0/1/2/3)` marginals, audited against enumeration and simulation. Temperature τ and persistent-effect scale σ are selected **jointly**: match NLL is integrated over the same player-season effects the simulator draws, and the predeclared criterion combines that with contender CRPS. Production: τ = 0.8, σ = 0.4, with variance-normalised **Student-t(4)** persistent effects; the heavier tails beat the Gaussian on rolling contender CRPS (2.63 vs 2.68) and 90% coverage (0.91 vs 0.85) at an unchanged favourite hit rate, and a 10%/3× Gaussian mixture is statistically equivalent.
6. **Player effects** — each simulation holds one persistent player-season deviation across all of a player's matches, and a shrunken historical effect nudges chronically underrated players using prior-season residuals (observed minus Plackett–Luce expectation), never raw vote totals.
7. **Award-stage eligibility** — suspended players keep their simulated votes (they affect everyone's totals) but cannot win the medal: outright and joint-first probabilities are computed over eligible players only. Sources are committed and auditable.
8. **Interactive visualisation** — the React app in [`web/`](web) shows contender worms with 50%/90% bands, sampled simulations, a round-vote heatmap, and an "ineligible" marker for suspended players.

## Results

Data audit (2012–2025): **1,824 players crosswalked** across Champion Data and AFL Tables with zero unmatched or ambiguous, and **19,559 / 19,559 AFLCA vote rows resolved** — every match sums to the full 30 coaches' votes.

Rolling out-of-sample records for the production model (each season's model, calibration and player effects use only earlier data; award probabilities respect suspensions):

| Season | τ, σ | Contender CRPS | 90% coverage | 50% coverage | Favourite won | Winner prior P |
|--------|------|----------------|--------------|--------------|---------------|----------------|
| 2021 | 0.7, 0.4 | 2.78 | 0.93 | 0.47 | yes | 44.0% |
| 2022 | 0.7, 0.5 | 1.87 | 1.00 | 0.73 | no | 17.0% |
| 2023 | 0.8, 0.3 | 2.05 | 0.93 | 0.53 | no | 1.9% |
| 2024 | 0.8, 0.3 | 3.79 | 0.73 | 0.13 | yes | 49.5% |
| 2025 | 0.8, 0.4 | 2.65 | 0.93 | 0.73 | no | 3.8% |
| **Mean** | | **2.63** | **0.91** | **0.52** | **2/5** | **23.2%** |

The model is honest rather than decisive: the eventual winner averaged a 23.2% pre-count probability and the favourite won 2 of the 5 most recent counts. 2024 remains the standing failure mode — record vote inflation sat outside the centre of every specification, though heavy-tailed season effects narrowed the miss.

**Approach comparison** (rolling 2021–2025, suspensions applied; legacy = the original 100-model Normal-draw design, reconstructed faithfully):

| Approach | Contender CRPS | 90% coverage | Favourite won | Winner prior P |
|---|---|---|---|---|
| Legacy | 4.77 | 0.20 | 0/5 | 0.000 |
| **Current production (season form, Student-t(4) effects)** | **2.63** | **0.91** | **2/5** | **0.232** |
| Stacking ensemble (LambdaMART + PL + forest) | 2.65 | 0.93 | 1/5 | 0.186 |

Both redesign-era approaches are in a different class from the legacy baseline. The season-form production model edges the ensemble on CRPS as well as the top-of-count signal (its row reflects the adopted Student-t(4) effects); the ensemble remains the better-calibrated alternative. The full write-up is in [`docs/methodology.md`](docs/methodology.md#approach-comparison-legacy-vs-current-vs-ensemble).

**2026 forecast** (post-round-24 conditional, 10,000 simulations, 32 suspended players excluded from the medal): **Nick Daicos 43.7 expected votes (90% interval 36–51), 90.3% first-or-joint**, spanning 70–95% across effect-scale specifications; Bailey Smith 31.9 (4.9%), Patrick Cripps 30.1 (3.2%) and Marcus Bontempelli 30.2 (1.2%) are the alternatives.

## Repository Structure

```text
brownlow/
├── R/
│   ├── extract_fitzroy.R             # AFL API players/results + AFL Tables votes
│   └── extract_squads.R              # AFL API squads: height and position
├── data/
│   ├── eligibility/                  # committed suspension records (auditable)
│   │   ├── ineligible_brownlow.csv   # ineligible leading vote-getters 2012-2025
│   │   └── suspensions_2026.csv      # 2026 in-season suspensions (MRO tracker)
│   └── README.md                     # regeneration steps (raw data is gitignored)
├── docs/
│   ├── methodology.md                # full mathematical approach and validation
│   ├── forecast-2026.md              # 2026 forecast report
│   └── agent-sessions/               # development session summaries
├── src/brownlow/
│   ├── ingest.py                     # crosswalk + audited three-state labels
│   ├── features.py                   # 124-feature engineering (CBA, height)
│   ├── aflca.py                      # AFLCA coaches-vote scraper and resolver
│   ├── folds.py                      # chronological, match-grouped folds
│   ├── model.py                      # LambdaMART + Plackett-Luce objectives, device
│   ├── scores.py                     # cached per-season out-of-sample scores
│   ├── evaluate.py                   # leak-free tau, NLL/Brier/CRPS/coverage
│   ├── validation.py                 # joint calibration and rolling backtests
│   ├── history.py                    # partially pooled historical player effects
│   ├── eligibility.py                # award-stage suspension rules
│   ├── simulate.py                   # persistent-effect season simulator
│   ├── pl.py                         # Plackett-Luce NLL, tau fit, exact marginals
│   └── cli.py                        # all commands (see Quick Start)
├── tests/                            # unit tests
├── web/                              # interactive React forecast visualisation
├── .github/workflows/deploy-pages.yml
├── pyproject.toml
├── LICENSE, NOTICE.md
└── README.md
```

## Quick Start

```bash
git clone https://github.com/jay-stein/brownlow-medal-predictor.git
cd brownlow-medal-predictor
uv sync
```

Data (the large extracts are gitignored and rebuilt locally):

```bash
# in R: source("R/extract_fitzroy.R"); source("R/extract_squads.R")   (~10 min)
uv run python -m brownlow.cli fetch-coaches --seasons 2012-2026       # ~6 min
uv run python -m brownlow.cli audit                                   # builds data/processed/
```

Train, calibrate and forecast:

```bash
uv run python -m brownlow.cli baseline --model ranking_season --seasons 2015-2026
uv run python -m brownlow.cli calibrate-joint --model ranking_season --seasons 2015-2025 --tune-seasons 2015-2025
uv run python -m brownlow.cli player-effects --model ranking_season --seasons 2015-2025
uv run python -m brownlow.cli rolling-backtest --model ranking_season --report-seasons 2021-2025
uv run python -m brownlow.cli fetch-reports --season 2026
uv run python -m brownlow.cli forecast --model ranking_season --season 2026 --n-sims 10000 \
  --sensitivity-scales 0.3,0.5,0.7 --web-json web/public/forecast_2026.json
```

Older commands (`baseline --model ranking`, `calibrate-effects`, `evaluate`) remain available for the stats-only baseline. Run the tests with `uv run pytest`.

Training uses CPU by default; set `BROWNLOW_XGB_DEVICE=cuda` to use a CUDA build (benchmarked at parity on this workload, and it changes rounding slightly).

### Interactive visualisation

```bash
cd web
npm install
npm run dev        # local dev server
npm run build      # static build in web/dist
```

The app has six views: **Contenders** (ranked leaderboard and the selected player's cumulative worm with 50%/90% bands), **Top 10 race** (cumulative expected votes for the ten most likely winners with a summary table), **Teams Totals** (every club's expected vote total with its 90% range and leading players; click a club to expand its cumulative worm or per-round votes), **All Matches** (round-by-round tiles with predicted 3-2-1 probabilities, actual scorelines, venue, game stats, coaches' votes, and expandable official match report excerpts from AFL.com.au), **Matches by Team** (pick any club to get its full season in one chronological scroll, with that club's players grouped first and highlighted), and **Past Winners** (2014–2025: who the rolling model expected before each count, who actually won, and where the medallist ranked). Method, performance, shortcomings and credits live on **Nerdy Stuff**. The app reads `web/public/forecast_2026.json`, regenerated by the `forecast` command above. A hosted copy deploys to GitHub Pages from `main` via [`.github/workflows/deploy-pages.yml`](.github/workflows/deploy-pages.yml).

## Data

The large datasets are **not committed**; see [`data/README.md`](data/README.md) for regeneration. The small eligibility records are committed because they are curated and auditable.

| Regenerated locally (gitignored) | Source |
|----------------------------------|--------|
| `player_stats_2012_2026_fitzroy.csv` | Champion Data via `fitzRoy::fetch_player_stats()` (AFL API, 2012+) |
| `team_stats_2012_2026_fitzroy.csv` | `fitzRoy::fetch_results_afl()` |
| `brownlow_stats_2012_2025_fitzroy.csv` | AFL Tables via `fetch_player_stats_afltables()` (includes `Brownlow.Votes`) |
| `player_details_2012_2026_afl.csv` | AFL API squads: height and position per player-season |
| `aflca_votes.csv` | AFLCA Champion Player leaderboards: per-match panel votes (2012+) |

## Roadmap

- [x] Audited data layer, time-ordered validation and rolling backtests.
- [x] Joint (τ, σ) calibration with effect-integrated match probabilities.
- [x] Historical player effects and direct Plackett–Luce training variant.
- [x] AFLCA coaches' votes as features (the winning variant).
- [x] Award-stage suspension eligibility with committed sources.
- [x] **Approach comparison** against the reconstructed legacy baseline and a stacking ensemble (`legacy-roll`, `ensemble-roll`, `compare-approaches`); the current model wins the composite scorecard, the ensemble wins on distributional calibration.
- [x] **Season-form features** (leave-one-game-out aggregates) promoted to production: rolling contender CRPS 2.62, 90% coverage 0.92, favourite hit rate 2/5, winner prior P 23.3%.
- [ ] **Plackett–Luce objective + coaches' votes combined** (best match NLL plus the human signal).
- [ ] **Live in-season variant** that updates weekly with only the information available at that round.
- [ ] **Match-block bootstrap ensemble** for model-parameter uncertainty, with award metrics in the selection criterion.
- [ ] **Retrospective check** of the 2026 forecast once votes are counted.

## Caveats

- **Post-season boundary**: the production forecast observes 2026 match statistics **and** coaches' votes, so it is a count-night conditional forecast, not a pre-season or live one. Coaches' votes publish weekly, so a live variant is possible with a smaller information set.
- **Quarantined labels**: a handful of AFL Tables player-game rows are unresolved and excluded from training; 2026 has no umpire votes yet.
- **Eligibility is applied from published trackers**: in-season suspensions are covered, but tribunal rules also include finals and pre-season carry-overs, and the current-season list must be refreshed after each sitting.
- **No quarter-level features**: fourth-quarter disposals are not publicly available (see `data/README.md`).
- **Estimates, not certainties**: vote totals, intervals and probabilities are model outputs validated on the rolling seasons above.

## Licence and attribution

Released under the [MIT License](LICENSE); data attribution is in [NOTICE.md](NOTICE.md). The MIT license covers the code and documentation; the underlying football data belongs to its providers and remains subject to their terms.

- AFL player statistics, match results and squad details via the AFL API (Champion Data), fetched with [`fitzRoy`](https://github.com/jimmyday12/fitzRoy) by [James Day](https://github.com/jimmyday12) and contributors.
- Historical player statistics and Brownlow votes via [AFL Tables](https://afltables.com), also fetched with `fitzRoy`.
- AFL Coaches Association Champion Player votes via the published [AFLCA leaderboards](https://aflcoaches.com.au/awards/the-aflca-champion-player-of-the-year-award/leaderboard).
- Official match report headlines and short excerpts via [AFL.com.au](https://www.afl.com.au) match centre pages, with links back to each article.
- Club logo images via the [Squiggle](https://squiggle.com.au) API; club names and logos remain the trademarks of their clubs.
- Suspension eligibility cross-checked against the Wikipedia Brownlow Medal articles and the published Zero Hanger MRO/tribunal tracker.
- Earlier iterations compared external predictions from the AFL.com.au Brownlow endpoint, Betfair, ESPN and WheeloRatings; that code lives in the repository history.
