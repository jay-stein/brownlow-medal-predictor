# Brownlow Medal Predictor

Predicts AFL Brownlow Medal votes for every game of a season and simulates the count to produce calibrated vote distributions and medal probabilities.

**Live 2026 forecast:** https://jay-stein.github.io/brownlow-medal-predictor/

[![License: MIT](https://img.shields.io/badge/license-MIT-d4af37.svg)](LICENSE)

The full mathematical treatment — including the Brownlow fixed-budget mechanics, the uncertainty model and known limitations — is in [`docs/methodology.md`](docs/methodology.md).

## How it works

1. **Data extraction (R)** — [`R/extract_fitzroy.R`](R/extract_fitzroy.R) pulls AFL API player stats and match results (2012–2026) plus AFL Tables player stats and Brownlow votes (2012–2025) via [`fitzRoy`](https://github.com/jimmyday12/fitzRoy); [`R/extract_squads.R`](R/extract_squads.R) adds height and position per player-season.
2. **Audited labels** — `ingest.py` builds a Champion Data ↔ AFL Tables player crosswalk (compact name keys, surname/team/date fallback, generational-suffix handling) and attaches **three-state labels**:
   - `voted` — matched to a resolved match record and polled votes
   - `zero` — matched to a resolved match record and polled no votes (a genuine zero)
   - `unresolved` — identity join failed or a match's recipients did not all map; **quarantined from training**, never silently treated as zero
3. **Features** — 124 per player-game: 58 raw stats, 7 engineered (player points, home flag, margin, minutes in front, outcome, captain), 56 within-team share features, position, plus player height and height relative to same-position peers. Centre-bounce attendances are available from 2021 and left as native-missing before then.
4. **Score generator** — a match-grouped LambdaMART ranker (`rank:ndcg`, `model.py`) produces a utility per player-game, trained only on earlier seasons with match-grouped CV for round selection and train-only preprocessing.
5. **Plackett–Luce allocation** — `pl.py` turns utilities into a distribution over ordered 3–2–1 triples with one temperature `tau`, fitted leak-free on earlier out-of-sample seasons, and exact `P(0/1/2/3)` marginals.
6. **Season simulation** — `simulate.py` draws one persistent player-season effect per player, held across all their matches, then 10,000 Plackett–Luce counts per season. From the joint distribution come vote quantiles, outright/joint first-place and top-5 probabilities, and per-round `P(1)/P(2)/P(3)`.
7. **Interactive visualisation** — the React app in [`web/`](web) shows each contender's cumulative vote worm with 50%/90% bands, sampled simulation overlays, and a round-by-round vote probability heatmap.

## Results

Data audit (2012–2025): **1,824 players crosswalked** across Champion Data and AFL Tables, **zero unmatched or ambiguous**, every fixture resolved, and only genuinely missing AFL Tables rows quarantined (1 in 2015, 5 in 2024, 83 in 2025).

Chronological backtest (`--model ranking`; scores trained only on earlier seasons, `tau` fitted leak-free on earlier out-of-sample seasons), 2015–2025:

| Season | allocation NLL | skill vs uniform | top-1 hit | top-3 slot overlap |
|--------|----------------|------------------|-----------|--------------------|
| 2015 | 5.08 | 0.55 | 0.59 | 0.66 |
| 2016 | 4.64 | 0.59 | 0.63 | 0.69 |
| 2017 | 4.58 | 0.59 | 0.60 | 0.69 |
| 2018 | 4.83 | 0.57 | 0.55 | 0.69 |
| 2019 | 4.53 | 0.60 | 0.61 | 0.70 |
| 2020 | 5.62 | 0.50 | 0.57 | 0.64 |
| 2021 | 4.79 | 0.58 | 0.67 | 0.69 |
| 2022 | 4.36 | 0.62 | 0.62 | 0.70 |
| 2023 | 6.16 | 0.46 | 0.51 | 0.60 |
| 2024 | 5.80 | 0.49 | 0.52 | 0.61 |
| 2025 | 5.50 | 0.52 | 0.51 | 0.64 |

Season spread calibration (persistent player-season effect at the calibrated scale 0.4; contender set = observed top-15):

| Metric | Result |
|--------|--------|
| Contender 90% interval coverage | mean **0.885** (nominal 0.90; range 0.73–1.00) |
| Contender 50% interval coverage | mean **0.486** (nominal 0.50) |
| Simulated-vs-observed Spearman correlation | 0.73–0.78 in every season |
| Favourite (highest simulated mean) won | 5 of 11 seasons |
| Eventual winner's mean P(win) | **0.36** (range 0.02–0.94) |

The model is honest rather than decisive: the eventual winner carried a 36% average pre-count probability and was ranked first in 5 of 11 seasons. The 2024 and 2025 winners (45 and 39 votes) were over-performers the model ranked 2nd and 9th, yet contender intervals still covered 73% and 87% of that season's leading players.

## Repository Structure

```text
brownlow/
├── R/extract_fitzroy.R               # AFL API players/results + AFL Tables votes
├── R/extract_squads.R                # AFL API squads: height and position
├── data/                             # raw inputs + processed outputs (gitignored)
├── docs/
│   ├── methodology.md                # mathematical approach and uncertainty treatment
│   ├── forecast-2026.md              # 2026 forecast report
│   └── agent-sessions/               # development session summaries
├── src/brownlow/                     # uv-managed pipeline package
│   ├── ingest.py                     # crosswalk + audited three-state labels
│   ├── features.py                   # 124-feature engineering (incl. CBA and height)
│   ├── folds.py                      # chronological, match-grouped folds
│   ├── model.py                      # regression + ranking (LambdaMART) score generators
│   ├── scores.py                     # cached per-season out-of-sample scores
│   ├── evaluate.py                   # leak-free tau, NLL/Brier/CRPS/coverage metrics
│   ├── simulate.py                   # persistent-effect season simulator
│   ├── pl.py                         # Plackett-Luce NLL, tau fit, exact marginals
│   └── cli.py                        # audit / baseline / evaluate / calibrate-effects / forecast
├── tests/                            # unit tests
├── web/                              # interactive React forecast visualisation
├── .github/workflows/deploy-pages.yml
├── pyproject.toml                    # uv-managed project and dependencies
├── LICENSE, NOTICE.md
└── README.md
```

## Quick Start

```bash
git clone https://github.com/jay-stein/brownlow-medal-predictor.git
cd brownlow-medal-predictor
uv sync
```

Rebuild the audited dataset (crosswalk + three-state labels → `data/processed/`):

```bash
uv run python -m brownlow.cli audit
```

Run the chronological backtest (per-season scores are cached and reused):

```bash
uv run python -m brownlow.cli baseline --model ranking --seasons 2013-2025  # out-of-sample scores
uv run python -m brownlow.cli evaluate --model ranking --seasons 2015-2025  # leak-free tau + metrics
uv run python -m brownlow.cli calibrate-effects --model ranking             # effect scale grid
uv run python -m brownlow.cli forecast --model ranking --season 2026        # season simulation
```

The 2026 forecast is written up in [`docs/forecast-2026.md`](docs/forecast-2026.md). Run the tests with `uv run pytest`.

### Interactive visualisation

```bash
cd web
npm install
npm run dev        # local dev server
npm run build      # static build in web/dist
```

The app reads `web/public/forecast_2026.json`, which is regenerated with:

```bash
uv run python -m brownlow.cli forecast --model ranking --season 2026 --n-sims 10000 \
  --web-json web/public/forecast_2026.json
```

A hosted copy is deployed to GitHub Pages from `main` by [`.github/workflows/deploy-pages.yml`](.github/workflows/deploy-pages.yml): https://jay-stein.github.io/brownlow-medal-predictor/

## Data

The datasets are **not committed**: the large raw extracts are regenerated with `fitzRoy` and all derived outputs are rebuilt from them. See [`data/README.md`](data/README.md) for the full regeneration steps.

| Regenerated locally (gitignored) | Source |
|----------------------------------|--------|
| `player_stats_2012_2026_fitzroy.csv` | Champion Data via `fitzRoy::fetch_player_stats()` (AFL API, 2012+) |
| `team_stats_2012_2026_fitzroy.csv` | `fitzRoy::fetch_results_afl()` |
| `brownlow_stats_2012_2025_fitzroy.csv` | AFL Tables via `fitzRoy::fetch_player_stats_afltables()` (includes `Brownlow.Votes`) |
| `player_details_2012_2026_afl.csv` | AFL API squads (`fetch_squad_afl`): height and position per player-season |

Processed artifacts are written to `data/processed/` by `brownlow.cli audit` (player crosswalk, per-season label audit, unmatched/ambiguous lists, `labelled_player_games.parquet`), with score caches and evaluation tables under `data/processed/scores/` and `data/processed/evaluation/`.

## Environment

Python dependencies are declared in [`pyproject.toml`](pyproject.toml) and locked by `uv` (`uv sync` creates `.venv/` and `uv.lock`). The web app uses Node (Vite + React + Recharts). R (≥ 4.1) packages for data extraction: `fitzRoy` (≥ 1.8.0), `dplyr`, `knitr`/`rmarkdown`.

## Roadmap

- [x] Audited data layer, chronological match-grouped backtests and leak-free evaluation.
- [x] Plackett–Luce allocation with fitted temperature and exact marginals.
- [x] Match-grouped LambdaMART score generator replacing the pseudo-Huber baseline.
- [x] Persistent player-season uncertainty, calibrated by contender CRPS.
- [x] Extended history (2012–2025 labels) and new features (centre-bounce attendances, height).
- [x] Award probabilities (outright, joint, first-or-joint, top-5) and the 2026 forecast.
- [x] Interactive visualisation and GitHub Pages deployment.
- [ ] **Eligibility at the award stage** (suspensions) once an in-season source is available.
- [ ] **Model variation**: match-block bootstrap ensemble of score generators.
- [ ] **Retrospective check** of the 2026 forecast once the votes are counted.

## Caveats

- **Quarantined labels**: the AFL Tables export is missing a handful of player-game rows (1 in 2015, 5 in 2024, 83 in 2025); these are marked `unresolved` rather than zero, and 2026 has no labels yet.
- **Eligibility is not applied**: suspended players are not filtered at the award stage; their match votes still affect other players' totals.
- **No quarter-level features**: fourth-quarter disposals are not publicly available (see `data/README.md`).
- **Estimates, not certainties**: vote totals, intervals and probabilities are model outputs validated on held-out seasons, not guarantees.

## Licence and attribution

Released under the [MIT License](LICENSE); data attribution is in [NOTICE.md](NOTICE.md). The MIT license covers the code and documentation; the underlying football data belongs to its providers and remains subject to their terms.

- AFL player statistics, match results and squad details via the AFL API (Champion Data), fetched with [`fitzRoy`](https://github.com/jimmyday12/fitzRoy) — the excellent R package by [James Day](https://github.com/jimmyday12) and contributors.
- Historical player statistics and Brownlow votes via [AFL Tables](https://afltables.com), also fetched with `fitzRoy`.
- Earlier iterations compared external predictions from the AFL.com.au Brownlow endpoint, Betfair, ESPN and WheeloRatings; that code lives in the repository history.
