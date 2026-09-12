# Data

This folder holds the project datasets. The large raw extracts are **gitignored** (they total well over 100 MB) and are regenerated with the R scripts in [`../R`](../R). Small prediction-source snapshots are committed so the pipeline can be re-run without re-scraping.

## Committed files

| File | Source |
|------|--------|
| `afl_website_2025.csv` | AFL API Brownlow award endpoint (`notebooks/scrape_aflcom.ipynb`) |
| `betfair_2025.csv` | Betfair Data Supplier API (`notebooks/scrape_betfair.ipynb`) |
| `2025_espn_predictions.csv` | ESPN Brownlow predictor |
| `espn_historical_predictions.xlsx` | ESPN 2021–2024 predictions (evaluation) |
| `wheelo-brownlow-predictions.csv` | WheeloRatings |
| `bonus_player_team_map.csv` | Manual player→team fixes for name matching |

## Regenerating the large extracts

| File | Source |
|------|--------|
| `player_stats_2012_2026_fitzroy.csv` | AFL API (Champion Data) via `fitzRoy::fetch_player_stats()`, 2012–2026 |
| `team_stats_2012_2026_fitzroy.csv` | `fitzRoy::fetch_results_afl()`, 2012–2026 |
| `brownlow_stats_2012_2025_fitzroy.csv` | AFL Tables via `fitzRoy::fetch_player_stats_afltables()`, includes `Brownlow.Votes` |
| `player_details_2012_2026_afl.csv` | AFL API squads via `fitzRoy:::fetch_squad_afl()`: height and position per player-season |

Run both scripts from the repository root:

```r
# install.packages("fitzRoy")  # R >= 4.1
source("R/extract_fitzroy.R")   # ~10 minutes: players, results, votes
source("R/extract_squads.R")    # ~2 minutes: heights and positions
```

Command line equivalent:

```bash
Rscript R/extract_fitzroy.R
Rscript R/extract_squads.R
```

Notes:

- The AFL API (Champion Data) player stats and results start in **2012**; Brownlow vote labels are available for **2012–2025**.
- Centre bounce attendances (`extendedStats.centreBounceAttendances`) are only populated from **2021**; the pipeline keeps them as native-missing features so XGBoost handles the era.
- Player per-quarter stats are **not** exposed by the AFL API; quarter-level features (e.g. fourth-quarter disposals) would need a different source.

## Processed outputs

`uv run python -m brownlow.cli audit` writes to `data/processed/`:

- `player_crosswalk.csv` — Champion Data ↔ AFL Tables player id mapping with match scores
- `label_audit_by_season.csv` — voted / genuine-zero / unresolved counts per season
- `unmatched_players.csv`, `ambiguous_players.csv` — crosswalk audit trails
- `labelled_player_games.parquet` — the feature table with three-state labels

`baseline`, `evaluate` and `calibrate-effects` add per-season score caches and evaluation tables under `data/processed/scores/` and `data/processed/evaluation/`.
