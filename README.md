# Brownlow Medal Predictor

Predicts AFL Brownlow Medal votes for every game of a season with a feature-bagged XGBoost ensemble, then runs Monte Carlo simulations of the count to estimate the winner, top pollers, and vote distributions.

The current model targets the **2025 season**: 100 XGBoost models — each trained on a random 70% subset of 120 engineered features — predict a float vote value per player per game. 1,000 simulated counts convert those predictions into 3/2/1 votes and season totals.

![2025 Brownlow XGBoost Prediction](result/brownlow_seaborn.png)

## How It Works

1. **Data extraction (R)** — [`R/fitzroy_data_extract.Rmd`](R/fitzroy_data_extract.Rmd) downloads per-game player stats, match results, and historical Brownlow votes (2018–2024) via the [`fitzRoy`](https://cran.r-project.org/package=fitzRoy) package and writes them to `data/`.
2. **Feature pipeline (Python)** — [`notebooks/xgboost_predict_brownlow_v2.ipynb`](notebooks/xgboost_predict_brownlow_v2.ipynb) merges player and team stats, excludes finals, joins Brownlow votes as the training label, and engineers features: player points, venue/outcome, margin, minutes in front, and per-team `*_prop` share features.
3. **Model training** — `xgboost.cv` (10-fold, early stopping) selects the boosting round for a `reg:pseudohubererror` / MAE objective. 100 models are then trained on random feature subsets with different seeds; predictions are aggregated into `pred_mean`, `pred_median`, `pred_std`, `pred_p05`, `pred_p95`.
4. **Vote scoring** — Players are ranked within each match by predicted float votes; the top three per match receive 3/2/1 Brownlow points.
5. **Monte Carlo** — [`notebooks/montecarlo_predict_brownlow_v1.ipynb`](notebooks/montecarlo_predict_brownlow_v1.ipynb) samples 1,000 vote values per player-game from `Normal(pred_mean, pred_std)`, re-ranks every simulated match, and aggregates season totals, win likelihood, and vote ranges.
6. **Backtesting** — [`notebooks/xgboost_brownlow_backcast.ipynb`](notebooks/xgboost_brownlow_backcast.ipynb) re-runs the pipeline with a chosen season held out (`test_year`, default 2022) and writes predicted leaderboards to `backcast/`.

## Repository Structure

```text
brownlow/
├── R/
│   └── fitzroy_data_extract.Rmd        # fitzRoy download script (run first)
├── archive/
│   └── Brownlow-2022-v3.ipynb          # original 2022 end-to-end notebook
├── backcast/
│   └── run_for_{2022,2023,2024}.csv    # backtest predicted leaderboards
├── data/                               # datasets — see data/README.md
├── models/                             # trained models and ensemble metadata
├── notebooks/                          # pipeline notebooks
├── result/                             # Monte Carlo outputs and plots
├── requirements.txt                    # Python dependencies
└── README.md
```

## Notebooks

Run in this order for a full season prediction:

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

## Quick Start

```bash
git clone https://github.com/jay-stein/brownlow-medal-predictor.git
cd brownlow-medal-predictor
pip install -r requirements.txt
```

Then follow the [data regeneration steps](data/README.md) to rebuild the large `fitzRoy` extracts, and run the notebooks in the order listed above.

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

## Models and Outputs

- `models/ensemble_model_info.json` — metadata for all 100 ensemble members (seed + feature subset each).
- `models/xgb_afl_model_0..9*.json` — saved sample of ensemble members.
- `models/xgboost_brownlow_simple.json`, `models/xgb_final_from_cv.json` — single-model baselines.
- `models/ensemble_predictions.csv` — aggregated predictions per 2025 player-game (input to Monte Carlo, regenerated by notebook 3).
- `result/bronwlon_mc_output_v1.csv` — per-game float predictions from the current pipeline.
- `result/brownlow_seaborn.png` — simulated top-5 vote distributions.
- `backcast/run_for_{2022,2023,2024}.csv` — backtest predicted leaderboards.

The 2025 simulation projects the leaders as **Nick Daicos, Bailey Smith, Noah Anderson, Caleb Serong, and Tom Green** (see plot above).

## Environment

Python packages are listed in [`requirements.txt`](requirements.txt):

```text
pandas numpy matplotlib seaborn plotly xgboost scikit-learn shap tqdm openpyxl
```

R (≥ 4.1) packages: `fitzRoy` (≥ 1.8.0), `dplyr`, `knitr`/`rmarkdown`.

## Roadmap

- [ ] **2026 season prediction** — update the data extracts, retrain the ensemble, and produce the 2026 Monte Carlo outputs on the `feat/2026-prediction` branch.
- [ ] Remove legacy hardcoded data paths from the notebooks so the pipeline runs from any clone location.
- [ ] Persist all 100 ensemble members (`save_models = True`) for full reproducibility.

## Caveats

- **Legacy hardcoded paths**: several notebooks read from `C:\Users\mrjay\python\jupyter_notebooks\Projects\brownlow\data\...`. Update these to this repository's `data/` folder (or relative `../data/...`) before running.
- **Gitignored artifacts**: the large `data/*fitzroy.csv` extracts, `models/ensemble_predictions.csv`, and `result/*.csv` are not committed and must be regenerated.
- **Ensemble persistence**: the v2 notebook has `save_models = False`; only members 0–9 are stored in `models/`. Set `save_models = True` to persist all 100.
- **Notebook sizes**: `montecarlo_predict_brownlow_v1.ipynb` (~8.6 MB) and other notebooks embed large outputs.

## Credits

- AFL data via [`fitzRoy`](https://github.com/jimmyday12/fitzRoy) (AFL Tables, Footy Wire, The Squiggle).
- Brownlow predictor votes via AFL.com.au API and Betfair Data Supplier API.
- ESPN and WheeloRatings predictions for external comparison.
