# Data

This folder holds the project datasets. The large raw extracts are **gitignored** (they total ~74 MB) and are regenerated with the R script in [`../R/fitzroy_data_extract.Rmd`](../R/fitzroy_data_extract.Rmd). Small prediction-source snapshots are committed so the pipeline can be re-run without re-scraping.

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

The following files are produced by `fitzRoy` and are **not committed**:

| File | Source |
|------|--------|
| `player_stats_2018_2025_fitzroy.csv` | `fitzRoy::fetch_player_stats()` (Champion Data) |
| `team_stats_2018_2025_fitzroy.csv` | `fitzRoy::fetch_results_afl()` |
| `brownlow_stats_2018_2024_fitzroy.csv` | `fitzRoy::fetch_player_stats_afltables()` (includes `Brownlow.Votes`) |

Steps:

1. Install R (≥ 4.1) and the package:

   ```r
   install.packages("fitzRoy")
   ```

2. Open `R/fitzroy_data_extract.Rmd` and update the output `file_path` to this repository's `data/` folder:

   ```r
   file_path = "/path/to/brownlow-medal-predictor/data"
   ```

3. Knit or run all chunks. The script fetches seasons 2018–2025 and writes the three CSVs into `data/`.

Alternatively, run equivalent commands directly:

```r
library(fitzRoy)

player_stats_all <- dplyr::bind_rows(
  lapply(2018:2025, function(s) fitzRoy::fetch_player_stats(season = s))
)
team_stats_all <- dplyr::bind_rows(
  lapply(2018:2025, function(s) fitzRoy::fetch_results_afl(season = s))
)
brownlow_votes_all <- dplyr::bind_rows(
  lapply(2018:2024, function(s) fitzRoy::fetch_player_stats_afltables(season = s))
)

write.csv(player_stats_all, "data/player_stats_2018_2025_fitzroy.csv")
write.csv(team_stats_all, "data/team_stats_2018_2025_fitzroy.csv")
write.csv(brownlow_votes_all, "data/brownlow_stats_2018_2024_fitzroy.csv")
```
