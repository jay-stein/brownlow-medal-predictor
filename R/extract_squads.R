#!/usr/bin/env Rscript
# Extract player squad details (height, position) from the AFL API.
#
# Run from the repository root:
#   & "C:\Program Files\R\R-4.3.1\bin\Rscript.exe" R/extract_squads.R
#
# Output (data/):
#   player_details_2012_2026_afl.csv  providerId, season, position, heightInCm, ...

suppressPackageStartupMessages(library(fitzRoy))

seasons <- 2012:2026
comp <- "AFLM"
out_path <- "data/player_details_2012_2026_afl.csv"

teams <- fitzRoy:::find_team_id(NULL, comp = comp)
message(sprintf("Teams: %d", nrow(teams)))

parts <- list()
for (season in seasons) {
  comp_seas_id <- fitzRoy:::find_season_id(season, comp)
  if (is.null(comp_seas_id) || length(comp_seas_id) == 0) {
    warning(sprintf("No comp season id for %d", season), call. = FALSE)
    next
  }
  for (index in seq_len(nrow(teams))) {
    team_id <- teams$id[index]
    team_name <- teams$name[index]
    squad <- tryCatch(
      fitzRoy:::fetch_squad_afl(
        teamId = team_id,
        team = team_name,
        compSeasonId = comp_seas_id,
        season = season
      ),
      error = function(e) {
        warning(
          sprintf("Squad %s %d failed: %s", team_name, season, conditionMessage(e)),
          call. = FALSE
        )
        NULL
      }
    )
    if (!is.null(squad) && nrow(squad) > 0) {
      parts[[length(parts) + 1]] <- squad
    }
  }
  message(sprintf("Season %d: %d squads collected", season, length(parts)))
}

details <- dplyr::bind_rows(parts)
write.csv(details, out_path)
message(sprintf("Wrote %s: %d rows, %d columns", out_path, nrow(details), ncol(details)))
