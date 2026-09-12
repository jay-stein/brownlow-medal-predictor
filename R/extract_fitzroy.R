#!/usr/bin/env Rscript
# Extended fitzRoy extraction for the Brownlow project.
#
# Run from the repository root:
#   & "C:\Program Files\R\R-4.3.1\bin\Rscript.exe" R/extract_fitzroy.R
#
# Outputs (data/):
#   player_stats_2012_2026_fitzroy.csv    AFL API player stats (Champion Data, 2012+)
#   team_stats_2012_2026_fitzroy.csv      AFL API match results (2012+)
#   brownlow_stats_2012_2025_fitzroy.csv  AFL Tables player stats + Brownlow votes

suppressPackageStartupMessages(library(fitzRoy))

player_seasons <- 2012:2026
result_seasons <- 2012:2026
vote_seasons <- 2012:2025

out_dir <- "data"
dir.create(out_dir, showWarnings = FALSE)

drop_list_columns <- function(df) {
  keep <- !vapply(df, is.list, logical(1))
  dropped <- names(df)[!keep]
  if (length(dropped) > 0) {
    message("  dropping list columns: ", paste(dropped, collapse = ", "))
  }
  df[, keep, drop = FALSE]
}

fetch_all <- function(seasons, fetcher, label) {
  parts <- list()
  for (season in seasons) {
    message(sprintf("Fetching %s %d ...", label, season))
    result <- tryCatch(
      fetcher(season),
      error = function(e) {
        warning(sprintf("%s %d failed: %s", label, season, conditionMessage(e)), call. = FALSE)
        NULL
      }
    )
    if (!is.null(result) && nrow(result) > 0) {
      result$EXTRACT_SEASON <- season
      parts[[length(parts) + 1]] <- result
      message(sprintf("  %s %d: %d rows", label, season, nrow(result)))
    }
  }
  dplyr::bind_rows(parts)
}

player_stats <- fetch_all(
  player_seasons,
  function(season) fitzRoy::fetch_player_stats(season = season),
  "player"
)
player_stats <- drop_list_columns(player_stats)
write.csv(player_stats, file.path(out_dir, "player_stats_2012_2026_fitzroy.csv"))
message(sprintf("player_stats: %d rows, %d columns", nrow(player_stats), ncol(player_stats)))

team_stats <- fetch_all(
  result_seasons,
  function(season) fitzRoy::fetch_results_afl(season = season),
  "results"
)
team_stats <- drop_list_columns(team_stats)
write.csv(team_stats, file.path(out_dir, "team_stats_2012_2026_fitzroy.csv"))
message(sprintf("team_stats: %d rows, %d columns", nrow(team_stats), ncol(team_stats)))

brownlow_stats <- fetch_all(
  vote_seasons,
  function(season) fitzRoy::fetch_player_stats_afltables(season = season),
  "afltables"
)
brownlow_stats <- drop_list_columns(brownlow_stats)
write.csv(brownlow_stats, file.path(out_dir, "brownlow_stats_2012_2025_fitzroy.csv"))
message(sprintf("brownlow_stats: %d rows, %d columns", nrow(brownlow_stats), ncol(brownlow_stats)))

message("Done.")
