# Data

The datasets are **not committed**. The large raw extracts are regenerated with the R scripts in [`../R`](../R), and every derived output is rebuilt from them.

## Regenerating the extracts

| File | Source |
|------|--------|
| `player_stats_2012_2026_fitzroy.csv` | AFL API (Champion Data) via `fitzRoy::fetch_player_stats()`, 2012–2026 |
| `team_stats_2012_2026_fitzroy.csv` | `fitzRoy::fetch_results_afl()`, 2012–2026 |
| `brownlow_stats_2012_2025_fitzroy.csv` | AFL Tables via `fitzRoy::fetch_player_stats_afltables()`, includes `Brownlow.Votes` |
| `player_details_2012_2026_afl.csv` | AFL API squads via `fitzRoy:::fetch_squad_afl()`: height and position per player-season |
| `aflca_votes.csv` | AFL Coaches Association Champion Player leaderboards: per-match combined panel votes (5-4-3-2-1 from each panel, 0–10 per player), 2012–2026 |

Run both R scripts from the repository root, then the coaches scraper:

```r
# install.packages("fitzRoy")  # R >= 4.1
source("R/extract_fitzroy.R")   # ~10 minutes: players, results, votes
source("R/extract_squads.R")    # ~2 minutes: heights and positions
```

```bash
uv run python -m brownlow.cli fetch-coaches --seasons 2012-2026   # ~6 minutes
```

Command line equivalent for the R extracts:

```bash
Rscript R/extract_fitzroy.R
Rscript R/extract_squads.R
```

Notes:

- The AFL API (Champion Data) player stats and results start in **2012**; Brownlow vote labels are available for **2012–2025**.
- Centre bounce attendances (`extendedStats.centreBounceAttendances`) are only populated from **2021**; the pipeline keeps them as native-missing features so XGBoost handles the era.
- **Player per-quarter stats are not publicly available.** The public AFL API returns match totals only and ignores period query parameters; the AFL's own period endpoints (`/cfs/afl/stats/match/...` and `/cfs/afl/periodStats/match/...`) return HTTP 403 Forbidden (Stats Pro premium), and the `statspro` host does not expose those paths publicly. fitzRoy, AFL Tables, Footywire, Squiggle and the public DFS Australia download all provide totals only. Fourth-quarter disposals would require a paid data provider.
- The AFLCA coach votes are published per round; all 2012–2026 rows resolve to a dataset player-game by season, date and club, and every match sums to the full 30 votes.
- Small **eligibility records are committed** under `data/eligibility/`: `ineligible_brownlow.csv` (ineligible leading vote-getters 2012–2025, sourced from the Wikipedia Brownlow articles) and `suspensions_<season>.csv` (full in-season suspension lists from the published MRO tracker; the 2026 file is a dated snapshot).

## Processed outputs

`uv run python -m brownlow.cli audit` writes to `data/processed/`:

- `player_crosswalk.csv` — Champion Data ↔ AFL Tables player id mapping with match scores
- `label_audit_by_season.csv` — voted / genuine-zero / unresolved counts per season
- `unmatched_players.csv`, `ambiguous_players.csv` — crosswalk audit trails
- `labelled_player_games.parquet` — the feature table with three-state labels

`baseline`, `evaluate`, `calibrate-joint`, `rolling-backtest`, `player-effects` and `forecast` add per-season score caches, evaluation tables (including the joint τ/σ grid, rolling backtests and player-effect grids) and forecast outputs under `data/processed/scores/`, `data/processed/evaluation/` and `data/processed/forecast/`.

Earlier iterations also used small prediction snapshots (AFL.com.au, Betfair, ESPN, WheeloRatings) for external comparisons; those files and the notebooks that consumed them are preserved in the repository history.
