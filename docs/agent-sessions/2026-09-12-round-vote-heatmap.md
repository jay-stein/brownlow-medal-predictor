# Session: Round Vote Probability Heatmap

**Date:** 2026-09-12
**Branch:** `agent/round-vote-heatmap`

## Goal

Make the "Round points" view of the web app easy to read: it should be immediately obvious which rounds a player is likely to poll 3 votes in.

## Files Changed

| File | Change |
|---|---|
| `src/brownlow/simulate.py` | Track per-round P(1), P(2), P(3) from the simulated increments; expose in `forecast_export` as `rounds.p1/p2/p3` |
| `tests/test_simulate.py` | Assertions for probability bounds and export keys |
| `web/src/RoundVotes.jsx` | New heatmap component: rows for 3/2/1 votes, gold intensity by probability, percentage labels, hot-round chips and a hover detail bar |
| `web/src/App.jsx` | Round mode now renders `RoundVotes`; toggle renamed "Round votes"; vote-swatch legend |
| `web/src/WormChart.jsx` | Simplified to the cumulative worm only (round mode moved out) |
| `web/src/styles.css` | Heatmap, chips, detail bar and vote swatch styles |
| `web/public/forecast_2026.json` | Regenerated with per-round probabilities |

## Commands Executed

- `uv run python -m brownlow.cli forecast --model ranking --season 2026 --n-sims 10000 --web-json web/public/forecast_2026.json --web-players 60 --web-paths 60`
- `uv run pytest` (47 passed), `uv run ruff check .` (clean)
- `cd web && npm run build`

## Results

The round view is now a 3-2-1 probability heatmap:

- Rows: 3 votes / 2 votes / 1 vote; columns: OR, R1–R24; cell brightness and percentage show the probability
- "Likeliest 3-vote rounds" chips give the headline directly (e.g. Nick Daicos: R6 89%, R16 87%, R3 86%)
- Byes and quiet rounds read as empty cells, so the difference between "will poll" and "won't play" is obvious
- Hovering a cell shows exact P(3), P(2), P(1) and the expected points for that round

## Important Decisions

1. **Probabilities computed in the simulator**, not estimated from quantiles: exact `(increment == 1/2/3).mean(axis=1)` per round per player.
2. **Heatmap over stacked bars**: brightness encodes likelihood directly and scales to 25 rounds without crowding, which was the readability problem with the previous bar view.
3. **Chips for the top three rounds by P(3)** answer the "round 17 they're definitely getting 3 votes" question without reading the grid.
4. **WormChart reduced to its cumulative role** since round mode now has a dedicated component.

## Blockers / Follow-ups

- Optional: click a hot round to jump the worm to that point, or add a team-coloured accent to the player's best rounds.
