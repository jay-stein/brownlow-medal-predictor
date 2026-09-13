# TODO

## Promote `ranking_season` (season-form features) to production

Leave-one-game-out season aggregates improved rolling 2021–2025 contender CRPS from 2.85 to 2.62,
held the favourite hit rate (2/5) and raised the average winner probability (0.217 → 0.233), with
the gain in 4 of 5 seasons. The context-only variant is similar on CRPS but over-covers, and the
combined variant is worse than either block alone, so promote the season aggregates only.

- [x] Train the missing seasons: `baseline --model ranking_season --seasons 2015-2017,2026`
- [x] Joint calibration on the full window:
      `calibrate-joint --model ranking_season --seasons 2015-2025 --tune-seasons 2015-2025` (τ=0.8, σ=0.4)
- [x] Historical player effects: `player-effects --model ranking_season --seasons 2015-2025` (100/4/0, improves 6/8 seasons)
- [x] Regenerate the forecast and web payload with `--model ranking_season`
- [x] Update the web model label, README, methodology and `forecast-2026.md`
- [ ] Re-check the rolling records and the live site after deploy

## Web engagement polish

- [x] **Team logos** in the leaderboard, race legend, teams view and match cards (Squiggle CDN, with a colour-monogram fallback on a light disc so dark logos stay visible).
- [x] **Legend/guide** for the match abbreviations (`28d` = disposals, `9g` = goals, `cv` = coaches' votes, `p3/p2/p1` = vote probabilities).
- [x] **Light probability shading** on the match vote cells (darker = more likely) with a mini bar, kept subtle.
- [x] **Nerdy Stuff page** with the full method, rolling performance, model comparison, shortcomings and references.
- [x] **Top 10 race** promoted to its own page with a summary table.
- [x] **Expandable match tiles** showing the fuller official report excerpt, all vote candidates and alternative exact triples.
- [x] **Matches by Team** page with club chips and the selected club's players grouped first and highlighted.
- [x] **Past Winners** page: 2014–2025 rolling forecasts vs actual medallists, with hit/miss cards and summary stats (early seasons flagged as limited history).
- [x] **Team Totals expansion**: clicking a club opens its cumulative team worm or per-round expected votes, both with 50%/90% bands from the simulations.
