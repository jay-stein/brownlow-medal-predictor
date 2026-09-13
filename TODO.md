# TODO

## Promote `ranking_season` (season-form features) to production

Leave-one-game-out season aggregates improved rolling 2021–2025 contender CRPS from 2.85 to 2.62,
held the favourite hit rate (2/5) and raised the average winner probability (0.217 → 0.233), with
the gain in 4 of 5 seasons. The context-only variant is similar on CRPS but over-covers, and the
combined variant is worse than either block alone, so promote the season aggregates only.

- [ ] Train the missing seasons: `baseline --model ranking_season --seasons 2015-2017,2026`
- [ ] Joint calibration on the full window:
      `calibrate-joint --model ranking_season --seasons 2015-2025 --tune-seasons 2015-2025`
- [ ] Historical player effects: `player-effects --model ranking_season --seasons 2015-2025`
- [ ] Regenerate the forecast and web payload with `--model ranking_season`
- [ ] Update the web model label, README, methodology and `forecast-2026.md`
- [ ] Re-check the rolling records and the live site after deploy
