# 2026 Brownlow Forecast

**Generated:** 2026-09-12, post-round-24 conditional forecast (only the hidden votes are simulated; match statistics are observed)
**Model:** `ranking` — match-grouped LambdaMART (`rank:ndcg`), trained on 2012–2025 labels (123,087 player-games, 268 boosting rounds)
**Temperature:** `tau = 0.845`, fitted leak-free on 2,555 out-of-sample matches (2013–2025)
**Persistent player effect:** scale 0.4, calibrated by contender CRPS on 2015–2023
**Simulations:** 10,000 Plackett–Luce season draws
**Raw outputs:** `data/processed/forecast/forecast_2026_ranking.csv` and `..._sensitivity.csv`
**Interactive version:** [`web/`](../web) — run `cd web && npm install && npm run dev` to explore the worms, bands and simulation overlays

| Player | Team | Exp. votes | 90% interval | P(outright) | P(joint) | P(first or joint) | P(top 5) | Spec sensitivity (first or joint) |
|---|---|---|---|---|---|---|---|---|
| Nick Daicos | Collingwood | 41.0 | 32–49 | 76.2% | 3.8% | 79.9% | 99.8% | 80–92% |
| Bailey Smith | Geelong Cats | 34.4 | 25–44 | 17.7% | 3.3% | 21.0% | 96.5% | 10–21% |
| Marcus Bontempelli | Western Bulldogs | 27.1 | 18–35 | 0.9% | 0.5% | 1.4% | 70.6% | 0–1% |
| Will Ashcroft | Brisbane Lions | 24.2 | 17–32 | 0.1% | 0.1% | 0.3% | 48.7% | 0–0% |
| Patrick Cripps | Carlton | 22.8 | 14–32 | 0.5% | 0.1% | 0.6% | 37.7% | 0–1% |
| Harry Sheezel | North Melbourne | 22.4 | 14–32 | 0.3% | 0.1% | 0.4% | 34.2% | 0–0% |
| Lachie Neale | Brisbane Lions | 21.4 | 14–30 | 0.1% | 0.0% | 0.1% | 26.8% | 0–0% |
| Jordan Dawson | Adelaide Crows | 21.1 | 13–29 | 0.1% | 0.0% | 0.1% | 23.9% | 0–0% |
| Isaac Heeney | Sydney Swans | 21.0 | 13–29 | 0.1% | 0.0% | 0.1% | 23.2% | 0–0% |
| Izak Rankine | Adelaide Crows | 19.5 | 13–26 | 0.0% | 0.0% | 0.0% | 10.2% | 0–0% |
| Zak Butters | Port Adelaide | 18.7 | 11–27 | 0.0% | 0.0% | 0.0% | 12.9% | 0–0% |
| Ed Richards | Western Bulldogs | 17.6 | 10–25 | 0.0% | 0.0% | 0.0% | 7.5% | 0–0% |
| Nasiah Wanganeen-Milera | St Kilda | 17.5 | 11–24 | 0.0% | 0.0% | 0.0% | 4.7% | 0–0% |
| Jai Newcombe | Hawthorn | 16.6 | 9–24 | 0.0% | 0.0% | 0.0% | 4.7% | 0–0% |
| Kysaiah Pickett | Melbourne | 16.5 | 11–22 | 0.0% | 0.0% | 0.0% | 1.6% | 0–0% |
| Jason Horne-Francis | Port Adelaide | 15.8 | 9–23 | 0.0% | 0.0% | 0.0% | 2.8% | 0–0% |
| Clayton Oliver | GWS GIANTS | 15.8 | 7–25 | 0.0% | 0.0% | 0.0% | 6.3% | 0–0% |
| Max Gawn | Melbourne | 15.3 | 7–25 | 0.0% | 0.1% | 0.1% | 5.4% | 0–0% |
| Andrew Brayshaw | Fremantle | 14.8 | 8–23 | 0.0% | 0.0% | 0.0% | 2.5% | 0–0% |
| Max Holmes | Geelong Cats | 14.6 | 8–22 | 0.0% | 0.0% | 0.0% | 1.6% | 0–0% |

*P(joint) is the probability of a tied first place; P(first or joint) is its sum with P(outright). Joint probabilities across players need not sum to 100%.*

## How to read this

- **Nick Daicos is a clear but not certain favourite**: 41.0 expected votes with a 90% interval of 32–49 and a 79.9% first-or-joint probability. Under the mildest spec (no persistent player effect) that probability is 92%; under the strongest it is 80%.
- **Bailey Smith is the main alternative**: 34.4 expected votes and a 21% first-or-joint probability (10–21% across specifications).
- **The spread is the point**: even for the favourite, the 90% interval spans 17 votes, and the model's historical backtest covered 88.5% of contenders at the nominal 90% level.
- **Calibration context**: across 2015–2025 the eventual winner averaged a 36% pre-count probability and the model's favourite won 5 of 11 seasons. Treat any single probability as model-conditional, not fate.

## Caveats

- **Eligibility is not applied.** The AFL Brownlow award API exposes an `eligible` flag only around the count; suspended players would be filtered at the award stage without removing their match votes (which still affect other players' totals). This is the main remaining Phase 4 item.
- **No quarter-level features** (fourth-quarter disposals are not publicly available; see `data/README.md`).
- **2026 squad data** comes from the same AFL API extraction as the training history; late-season list changes are as recorded by that source.
