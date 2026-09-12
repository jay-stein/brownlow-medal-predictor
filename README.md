# Brownlow Medal Predictor

Predicts AFL Brownlow Medal votes for every game of a season, then simulates the count to estimate the winner, top pollers, and vote distributions.

> **Redesign in progress** (branch `feat/2026-prediction`): the project is moving from the notebook regression pipeline to a probabilistic **Plackett–Luce allocation model** with explicitly calibrated uncertainty, evaluated on chronological, match-grouped backtests. See [Probabilistic pipeline](#probabilistic-pipeline-in-progress).

![2025 Brownlow XGBoost Prediction](result/brownlow_seaborn.png)

## How It Works (legacy notebook pipeline)

1. **Data extraction (R)** — [`R/fitzroy_data_extract.Rmd`](R/fitzroy_data_extract.Rmd) downloads per-game player stats, match results, and historical Brownlow votes (2018–2024) via the [`fitzRoy`](https://cran.r-project.org/package=fitzRoy) package and writes them to `data/`.
2. **Feature pipeline (Python)** — [`notebooks/xgboost_predict_brownlow_v2.ipynb`](notebooks/xgboost_predict_brownlow_v2.ipynb) merges player and team stats, excludes finals, joins Brownlow votes as the training label, and engineers features: player points, venue/outcome, margin, minutes in front, and per-team `*_prop` share features.
3. **Model training** — `xgboost.cv` (10-fold, early stopping) selects the boosting round for a `reg:pseudohubererror` / MAE objective. 100 models are then trained on random feature subsets with different seeds; predictions are aggregated into `pred_mean`, `pred_median`, `pred_std`, `pred_p05`, `pred_p95`.
4. **Vote scoring** — Players are ranked within each match by predicted float votes; the top three per match receive 3/2/1 Brownlow points.
5. **Monte Carlo** — [`notebooks/montecarlo_predict_brownlow_v1.ipynb`](notebooks/montecarlo_predict_brownlow_v1.ipynb) samples 1,000 vote values per player-game from `Normal(pred_mean, pred_std)`, re-ranks every simulated match, and aggregates season totals, win likelihood, and vote ranges.
6. **Backtesting** — [`notebooks/xgboost_brownlow_backcast.ipynb`](notebooks/xgboost_brownlow_backcast.ipynb) re-runs the pipeline with a chosen season held out (`test_year`, default 2022) and writes predicted leaderboards to `backcast/`.

## Probabilistic pipeline (in progress)

The `src/brownlow/` package implements the redesign. Its first version keeps the existing 120 features and uses the current model's scores as a baseline utility generator, while replacing the loss, allocation layer, simulation structure, and backtesting protocol.

1. **Audited labels** — `ingest.py` builds a Champion Data ↔ AFL Tables player crosswalk (compact name keys, surname/team/date fallback, generational-suffix handling) and attaches **three-state labels**:
   - `voted` — matched to a resolved match record and polled votes
   - `zero` — matched to a resolved match record and polled no votes (a genuine zero)
   - `unresolved` — identity join failed or a match's recipients did not all map; **quarantined from training**, never silently treated as zero
2. **Chronological, match-grouped evaluation** — `folds.py` trains on earlier seasons and evaluates on the next (`2018–20 → 2021`, …, final untouched `2018–24 → 2025`). Every match stays in one partition and preprocessing is fit on training seasons only.
3. **Explicit round selection** — `model.py` runs match-grouped CV and takes the mean best iteration across folds, rather than relying on a truncated CV history.
4. **Plackett–Luce allocation** — `pl.py` models each match as a distribution over ordered 3–2–1 triples, with a single temperature `tau` fitted on out-of-sample scores and exact `P(0/1/2/3)` marginals in closed form.

Current data audit (2018–2024): **all 1,359 fixtures resolved**, every one of the 4,092 match recipients mapped, zero unmatched players, and only 5 genuinely missing AFL Tables rows quarantined.

Chronological backtest (scores trained only on earlier seasons; `tau` fitted leak-free on earlier out-of-sample seasons):

| Season | leak-free tau | allocation NLL | NLL at tau=1 | uniform NLL | Brier | top-1 hit | top-3 slot overlap |
|--------|---------------|----------------|--------------|-------------|-------|-----------|--------------------|
| 2021 | 0.264 | 6.55 | 8.53 | 11.42 | 0.084 | 0.64 | 0.67 |
| 2022 | 0.310 | 6.41 | 8.62 | 11.42 | 0.086 | 0.58 | 0.70 |
| 2023 | 0.314 | 7.58 | 8.90 | 11.42 | 0.094 | 0.49 | 0.60 |
| 2024 | 0.339 | 7.64 | 8.90 | 11.42 | 0.093 | 0.52 | 0.60 |

The fitted temperature nearly matches the per-season oracle and cuts NLL by 1.3–2.0 nats against unfitted scores. Season simulation with one persistent player-season effect (calibrated by contender CRPS) beats the hard 3/2/1 point forecast on CRPS, but 90% intervals for leading contenders still under-cover (0.47–0.93), showing the score generator compresses dominant seasons.

## Repository Structure

```text
brownlow/
├── R/fitzroy_data_extract.Rmd        # fitzRoy download script (run first)
├── archive/                          # original 2022 notebook
├── backcast/                         # legacy backtest leaderboards
├── data/                             # raw inputs + processed outputs (gitignored)
├── docs/agent-sessions/              # agent session summaries
├── models/                           # legacy trained models and ensemble metadata
├── notebooks/                        # legacy pipeline notebooks
├── result/                           # legacy Monte Carlo outputs and plots
├── src/brownlow/                     # uv-managed pipeline package
│   ├── ingest.py                     # crosswalk + audited three-state labels
│   ├── features.py                   # 120-feature engineering (ported verbatim)
│   ├── folds.py                      # chronological, match-grouped folds
│   ├── model.py                      # baseline XGBoost + explicit round selection
│   ├── pl.py                         # Plackett-Luce NLL, tau fit, exact marginals
│   └── cli.py                        # `python -m brownlow.cli audit`
├── tests/                            # unit tests for the new package
├── pyproject.toml                    # uv-managed project and dependencies
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
uv run python -m brownlow.cli baseline --seasons 2020-2024     # out-of-sample scores per season
uv run python -m brownlow.cli evaluate --seasons 2020-2024     # leak-free tau + allocation metrics
uv run python -m brownlow.cli calibrate-effects                # player-season effect scale grid
```

Run the tests:

```bash
uv run pytest
```

Follow the [data regeneration steps](data/README.md) to rebuild the large `fitzRoy` extracts before running the legacy notebooks or the new pipeline end to end.

## Data

The large raw datasets are **not committed** — they are regenerated with `fitzRoy` (see [`data/README.md`](data/README.md)). Small prediction-source snapshots and the scraping notebooks are committed so the pipeline can be re-run without re-scraping.

| Committed in `data/` | Source |
|----------------------|--------|
| `afl_website_2025.csv` | AFL API Brownlow award endpoint (scraper notebook) |
| `betfair_2025.csv` | Betfair Data Supplier API (scraper notebook) |
| `2025_espn_predictions.csv` | ESPN Brownlow predictor |
| `espn_historical_predictions.xlsx` | ESPN 2021–2024 predictions (for evaluation) |
| `wheelo-brownlow-predictions.csv` | WheeloRatings |
| `bonus_player_team_map.csv` | Manual player→team fixes for name matching |

| Regenerated locally (gitignored) | Source |
|----------------------------------|--------|
| `player_stats_2018_2025_fitzroy.csv` | Champion Data via `fitzRoy::fetch_player_stats()` |
| `team_stats_2018_2025_fitzroy.csv` | `fitzRoy::fetch_results_afl()` |
| `brownlow_stats_2018_2024_fitzroy.csv` | AFL Tables via `fitzRoy::fetch_player_stats_afltables()` (includes `Brownlow.Votes`) |

Processed artifacts are written to `data/processed/` by `brownlow.cli audit`: the player crosswalk, the per-season label audit, unmatched/ambiguous player lists, and `labelled_player_games.parquet`.

## Notebooks

Run in this order for a full season prediction with the legacy pipeline:

| # | Notebook | Purpose |
|---|----------|---------|
| 1 | `scrape_aflcom.ipynb` | Scrapes AFL.com.au Brownlow predictor votes → `data/afl_website_2025.csv` |
| 2 | `scrape_betfair.ipynb` | Scrapes Betfair Brownlow vote data → `data/betfair_2025.csv` |
| 3 | `xgboost_predict_brownlow_v2.ipynb` | **Main pipeline**: feature engineering, CV, 100-model ensemble, per-game predictions |
| 4 | `montecarlo_predict_brownlow_v1.ipynb` | 1,000-simulation Monte Carlo, win likelihood, plots |
| 5 | `xgboost_brownlow_backcast.ipynb` | Backtests a held-out season and records predicted leaderboards |

Supporting notebooks:

- `xgboost_predict_brownlow.ipynb` / `xgboost_predict_brownlow_v1.ipynb` — earlier single-model iterations.
- `monte-carlo-espn-afl-betfair.ipynb` — aligns external predictions (ESPN, AFL.com.au, Betfair) to teams/rounds for blending experiments.
- `espn_vote_vs_outcome.ipynb` — evaluates ESPN historical predictions against actual Brownlow votes.

## Models and Outputs

- `models/ensemble_model_info.json` — metadata for all 100 legacy ensemble members (seed + feature subset each).
- `models/xgb_afl_model_0..9*.json` — saved sample of legacy ensemble members.
- `models/xgboost_brownlow_simple.json`, `models/xgb_final_from_cv.json` — single-model baselines.
- `models/ensemble_predictions.csv` — aggregated predictions per 2025 player-game (input to the legacy Monte Carlo, regenerated by notebook 3).
- `result/bronwlon_mc_output_v1.csv` — per-game float predictions from the legacy pipeline.
- `result/brownlow_seaborn.png` — simulated top-5 vote distributions.
- `backcast/run_for_{2022,2023,2024}.csv` — legacy backtest predicted leaderboards.

The 2025 simulation projects the leaders as **Nick Daicos, Bailey Smith, Noah Anderson, Caleb Serong, and Tom Green** (see plot above).

## Environment

Python dependencies are declared in [`pyproject.toml`](pyproject.toml) and locked by `uv` (`uv sync` creates `.venv/` and `uv.lock`). R (≥ 4.1) packages for data extraction: `fitzRoy` (≥ 1.8.0), `dplyr`, `knitr`/`rmarkdown`.

## Roadmap

Redesign phase status:

- [x] **Phase 1 — data repair**: audited crosswalk, three-state labels, chronological match-grouped folds, train-only preprocessing, explicit CV round selection.
- [x] **Phase 2a — allocation core**: Plackett–Luce NLL, temperature fitting, exact `P(0/1/2/3)` marginals, unit tested.
- [x] **Phase 2b — allocation model**: per-season out-of-sample scores, leak-free `tau` on earlier folds, allocation log-loss and Brier backtests.
- [x] **Phase 3a — persistent uncertainty**: player-season effect (one calibrated scale, contender-CRPS grid) and a shared simulator used for historical and future seasons.
- [ ] **Phase 3b — model variation**: match-block bootstrap ensemble of score generators.
- [ ] **Phase 4 — award definitions**: outright vs joint first-place probabilities, top-5 tie handling, player eligibility applied at the award stage (the AFL API Brownlow endpoint exposes an `eligible` flag).
- [ ] **Phase 4 — award definitions**: outright vs joint first-place probabilities, top-5 tie handling, player eligibility applied at the award stage (the AFL API Brownlow endpoint exposes an `eligible` flag).
- [ ] **Phase 5 — full backtest persistence**: allocation log-loss, multiclass Brier/calibration, CRPS and interval coverage (50/80/95%), award probabilities.
- [ ] **Phase 6 — 2026 forecast**: post-round-24 conditional forecast with quantile reporting and a sensitivity range across defensible model specifications.

Other follow-ups:

- [ ] Remove legacy hardcoded data paths from the notebooks so the pipeline runs from any clone location.
- [ ] Persist all 100 legacy ensemble members (`save_models = True`) for full reproducibility.

## Caveats

- **Legacy hardcoded paths**: several notebooks read from `C:\Users\mrjay\python\jupyter_notebooks\Projects\brownlow\data\...`. Update these to this repository's `data/` folder (or relative `../data/...`) before running.
- **Gitignored artifacts**: the large `data/*fitzroy.csv` extracts, `data/processed/`, `models/ensemble_predictions.csv`, and `result/*.csv` are not committed and must be regenerated.
- **Quarantined labels**: five 2024 player-games are missing from the AFL Tables export and are marked `unresolved` rather than zero. The remaining unresolved rows are the unlabelled 2025 season, which is expected.
- **Notebook sizes**: `montecarlo_predict_brownlow_v1.ipynb` (~8.6 MB) and other notebooks embed large outputs.

## Credits

- AFL data via [`fitzRoy`](https://github.com/jimmyday12/fitzRoy) (AFL Tables, Footy Wire, The Squiggle).
- Brownlow predictor votes via AFL.com.au API and Betfair Data Supplier API.
- ESPN and WheeloRatings predictions for external comparison.
