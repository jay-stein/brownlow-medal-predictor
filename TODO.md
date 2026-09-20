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
- [x] Re-check the rolling records and the live site after deploy

## Heavy-tailed persistent effects (Tier 2, adopted)

The Gaussian effect distribution understated extreme outcomes. Variance-normalised Student-t(4)
and a 10%/3x mixture both beat it on rolling 2021-2025 (CRPS 2.63 vs 2.68, 90% coverage 0.91 vs
0.85) at an unchanged favourite hit rate (2/5). Student-t(4) is adopted for parsimony.

- [x] `simulate.draw_effects` with normal/student-t/mixture, all variance-normalised
- [x] `--effect-distribution` and `--tag` on `calibrate-joint`, `rolling-backtest`, `forecast`
- [x] Rolling comparison: normal 2.684/0.853, Student-t 2.628/0.907, mixture 2.626/0.907
- [x] Adopt Student-t(4); regenerate the 2026 forecast and Past Winners (7/12 hits, 47.0% avg winner P)
- [x] Update README, methodology, `forecast-2026.md` and the Nerdy Stuff page
- [ ] Re-run the six-model comparison under Student-t(4) if the ensemble question is revisited

## Tier 1 context features (tested, not adopted)

Time-on-ground, umpire count/era and quarter-level team context were added to the audit and a
`ranking_tier1` variant was trained. Rolling 2021-2025: CRPS 2.677 vs 2.684 (noise), 90% coverage
0.880 vs 0.853 (better) but winner probability 0.200 vs 0.241 (worse) and match NLL slightly worse.
Kept as an experiment behind the model key; not promoted.

- [x] `ingest.attach_match_context`, `features.add_quarter_context`, `ranking_tier1`
- [x] Coverage verified (TOG 100%, umpire count 100% with 2026 era fallback, quarter context 92.8%)
- [x] Rolling comparison against `ranking_season`
- [ ] Optional: isolate TOG / umpire-era / quarter blocks to attribute the coverage gain

## Remaining Tier 1b / Tier 2 / ESPN work

- [ ] Shrunken umpire-crew tendency (umpire identity is in the raw data 2012-2025, unused)
- [ ] Coach-vote dispersion as a match-level temperature modifier
- [ ] Joint (tau, sigma, alpha-mapping) selection of the historical-effect strength
- [ ] AFL Media Brownlow predictor as an external benchmark variant
- [ ] ESPN play-by-play scraper (scoring plays, 2017+) and Q4 / late-Q3 momentum features

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
