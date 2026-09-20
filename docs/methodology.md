# Methodology

How the Brownlow prediction works, from the voting rule through to the probability statements — and, importantly, how uncertainty is treated. The original model was **overconfident in the final outcome**; correcting that was the central goal of the redesign, so it gets its own section.

## 1. The Brownlow as a fixed-budget allocation

The voting rule shapes the entire problem:

- **Votes are awarded to 3 players only per match — summing to 6 total votes (3 + 2 + 1).** Every home-and-away match awards one set of 3-2-1 to three different players, and since a player plays at most one match per round, it reads naturally as "three players and six votes per round". The pool is shared by everyone on the ground (~46 players).
- A player's ceiling is therefore **3 votes per round**; a bye or missed game is 0.
- League-wide, the season pool is exactly `6 × matches` (1,242 votes across 207 matches in 2026). The model cannot create or destroy votes — it only redistributes them.

### You are competing with your teammates

Within a match the game is **constant-sum**: expected points satisfy `Σᵢ E[vᵢ] = 6`, and for two teammates `i ≠ j`, increasing teammate `j`'s strength decreases `E[vᵢ]`. An elite teammate does not just take their own votes; they take *yours*.

This is not a corner case. Across the 2,753 voted matches in 2012–2025:

| Pattern | Matches | Share |
|---|---|---|
| All three recipients from one team | 1,347 | 48.9% |
| Two recipients from one team, one from the other | 1,406 | 51.1% |

(By pigeonhole, at least two of the three recipients always share a team.)

### Worked example

With the production temperature `τ = 0.7`, consider a match where player X has utility 2.0 and compare an elite teammate (utility 1.9) with quiet teammates (0.3 / 0.2, plus a fourth at -0.4):

| Scenario | P(X gets 3) | E[X] | E[others] |
|---|---|---|---|
| Elite teammate (1.9) | 49.2% | 2.38 | 2.29 / 0.71 / 0.62 |
| Quiet teammates (−0.4 / 0.3 / 0.2) | **83.5%** | **2.82** | 1.35 / 1.24 / 0.60 |

The elite teammate costs X ~0.44 expected votes and ~34 percentage points of P(3) *per game* — roughly 9 votes across a season. It is why a slightly lesser player who is the sole focal point can out-poll a superstar sharing a midfield.

### How the model represents it

The allocation layer is a **Plackett–Luce** model: the match is a sequential best-on-ground draw over the fixed 6-vote budget. With utilities `sᵢ` and weights `wᵢ = exp(sᵢ / τ)`:

```text
P(a gets 3, b gets 2, c gets 1) =
    w_a / Σw  ·  w_b / (Σw − w_a)  ·  w_c / (Σw − w_a − w_b)
```

Exactly one 3, one 2 and one 1 are awarded per match, so the constant-sum property is enforced by construction. The within-team share features give the score model a view of relative teammate standing; the competition itself is then handled by this layer (see [limitations](#9-modelling-weak-points)).

## 2. Problem statement

For each player `p` we want the distribution of their season total

```text
T_p = Σ_g v_{p,g}         v_{p,g} ∈ {0,1,2,3}
```

under the per-match constraint that the `v` values of each match are a permutation of `(3, 2, 1, 0, …, 0)`. From the joint distribution over all players we derive:

- `P(p wins outright) = P(T_p > max_{q≠p} T_q)`
- `P(p is joint first) = P(T_p = T_q for some q ≠ p and T_p is maximal)`
- `P(p is first or joint)`, `P(p finishes top five)`, vote quantiles and intervals.

This is a two-level problem: a discrete within-match allocation of a fixed budget, then season aggregation with correlated uncertainty.

## 3. Data and labels

- **AFL API (Champion Data)**, 2012–2026, via [`fitzRoy`](https://github.com/jimmyday12/fitzRoy): per-game player stats, match results, squad details (height, position).
- **AFL Tables**, 2012–2025, via `fitzRoy`: per-game player stats including `Brownlow.Votes` (14 label seasons).
- **AFL Coaches Association (AFLCA)**, 2012–2026: per-game combined coaching-panel votes (each panel awards 5-4-3-2-1, so a player scores 0–10 per match). These are human judgments of the game's standout players, independent of the umpires. The scraper resolves every row to a dataset player-game by season, date and club: **19,559 / 19,559 rows matched, every match sums to the full 30 votes**.
- **Identity crosswalk**: compact punctuation-insensitive name keys with generational-suffix stripping, then a surname + date + team fallback scored by first-name similarity. Result: 1,824 players mapped, **zero unmatched or ambiguous**.
- **Three-state labels** for every player-game:
  - `voted` — matched to a resolved match and polled 1–3 votes
  - `zero` — matched to a resolved match and genuinely polled no votes
  - `unresolved` — identity join failed, or the match's recipients did not all map; **quarantined from training**, never treated as zero

A match only yields genuine zeros when all three of its vote recipients map to stats rows. Examples: 5 quarantined rows in 2024, 83 in 2025, all of 2026 (no votes awarded yet).

## 4. Feature map

124 features per player-game:

| Block | Count | Notes |
|---|---|---|
| Position | 1 | categorical |
| Raw per-game stats | 58 | kicks, disposals, rating points, extended Champion Data stats, centre-bounce attendances |
| Engineered | 7 | `PLAYER_POINTS = 6·goals + behinds`, captain, home flag, team score, minutes in front, margin, outcome |
| Height | 2 | raw height; height minus the same-position peer mean in that match |
| Within-team shares | 56 | each stat divided by the team's match total (`*_prop`) |

Centre-bounce attendances (CBA) only exist from 2021. Rather than invent zeros, CBA is a **native-missing** feature: earlier seasons stay `NaN` and XGBoost routes missing values through its default branch. Height is a proxy for physical standout and is expressed relative to same-position peers, e.g. a tall midfielder versus other midfielders.

The production model adds three blocks on top of the 124 base features: the AFLCA panel votes (`COACH_VOTES`, 0–10) and the player's share of their team's match votes (`COACH_VOTES_SHARE`); and 13 leave-one-game-out season aggregates (per-game means for eight stats including coaches' votes, coach-vote total/rate/rank, team win rate and games played) for **139 features**. The panel votes encode what the two coaching panels thought of each game: a human, opposition-aware signal that raw stats miss, especially for players whose value is positioning, pressure or leadership rather than possessions. The season aggregates encode sustained form — the signal public vote models lean on most — under the same post-season information boundary the forecast already declares.

## 5. Score generator

A **LambdaMART ranker** (`rank:ndcg`, XGBoost) produces one real-valued utility `s_{p,g}` per player-game, with each match as a ranking group (`qid`) and relevance labels 0–3 votes. It replaced a pseudo-Huber / MAE regression baseline because it optimises **within-match ordering** — the quantity the 3-2-1 allocation actually depends on. The production model adds the AFLCA panel votes and leave-one-game-out season aggregates to the feature set.

Training protocol:

- For a target season `Y`, train only on labelled seasons `< Y`.
- Inside the training window, **time-ordered season validation** selects the number of boosting rounds: the last five seasons of the window each validate once against a model trained only on the seasons before it (the ten most recent rounds are used when the window contains only one season). This replaces a random match-grouped split, which avoided splitting matches but still trained on future eras.
- Preprocessing is fit on training-window seasons only: numeric imputation means, categorical encodings and the >5% missing-column drop are train-derived; no evaluation season informs them. Imputation means are computed once per window rather than per inner fold, a minor optimistic bias in round selection only (reported metrics never use them).
- The score's absolute scale is arbitrary (ranking is invariant to monotone transforms). A **positive rescaling** `s → s/τ` is absorbed by the temperature; a general monotone transform is not, because the allocation layer depends on the shape of the gaps between scores, not only their scale.

**Variant comparison.** Six candidates were trained and evaluated on the same rolling seasons (2018–2025 grids; rolling records for 2021–2025 with all settings selected on earlier out-of-sample seasons):

| Variant | Contender CRPS | 90% coverage | Favourite won | Winner prior P |
|---|---|---|---|---|
| `ranking_coaches` (previous production) | 2.85 | 0.88 | 2/5 | 0.217 |
| `ranking_weighted` (half-life 4 seasons) | 3.06 | 0.89 | 1/5 | 0.154 |
| `ranking_recent` (6-season window) | 3.07 | 0.91 | 1/5 | 0.132 |
| `ranking_nocba` (CBA features removed) | 3.08 | 0.89 | 1/5 | 0.150 |
| `ranking_pl` (direct ordered-triple likelihood) | 3.09 | 0.85 | 2/5 | 0.216 |
| `ranking` (all-history baseline) | 3.17 | 0.87 | 1/5 | 0.169 |
| `ranking_season` (**production**: leave-one-game-out season aggregates) | **2.62** | 0.92 | 2/5 | **0.233** |
| `ranking_context` (Elo, travel, close games) | 2.66 | 0.96 | 2/5 | 0.238 |
| `ranking_form` (aggregates + context) | 2.67 | 0.91 | 1/5 | 0.173 |

The coaches' votes are the one change with a clear signal; recency windows, recency weighting and the CBA ablation are all within noise of the baseline. Direct Plackett–Luce optimisation gives the best integrated match NLL (5.21 versus 5.24) but worse season CRPS and coverage, consistent with the review's caution that ranking quality and probability shape are different objectives. Combining the PL objective with coaches' votes is the natural next experiment.

**Season-form and context features.** Leave-one-game-out season aggregates (per-game stat means, coach-vote totals/rate/rank, team win rate) improve rolling contender CRPS to 2.62 with the same favourite hit rate and a higher average winner probability (0.233 versus 0.217); the gain holds in four of the five rolling seasons. Chronological Elo, interstate travel and close-game flags are similar on CRPS (2.66) but over-cover 90% intervals (0.96). Combining both blocks is *worse* than either alone (2.67, one favourite hit), which suggests the two feature families substitute for each other rather than add. The season-aggregate variant is now the production model (with heavy-tailed effects since §6); context features remain a documented experiment. The full pipeline depends on post-season information (the same boundary the forecast already declares).

## 6. Plackett–Luce allocation and temperature

Given utilities, the match distribution is the sequential Plackett–Luce model in §1. Marginal probabilities of receiving 0/1/2/3 votes are available in closed form by direct summation over the first two placements (`O(n²)` per match). The sequential denominator depends on the preceding picks, so symmetric-polynomial shortcuts from other weighted-subset models are **not** valid here; the implementation is audited against exhaustive enumeration on small synthetic matches and against Gumbel-max Monte Carlo simulation on full-size matches, and satisfies the exact invariants (each vote slot sums to one across players; expected match votes sum to six).

The **temperature τ** controls sharpness and the **persistent-effect scale σ** controls season-level spread. They are selected **jointly**, with both metrics computed on the same predictive distribution:

- For every candidate `(τ, σ)`, the match metric is the mean NLL of the observed ordered triples **integrated over the same persistent player-season effects the simulator draws**: each match probability is averaged over effect draws, so the match likelihood refers to the production distribution rather than an effect-free idealisation.
- The season metric is the contender CRPS of the simulated vote totals.
- Both metrics are standardised across the grid and combined with predeclared equal weights, breaking ties toward lower match NLL. The production selection over 2015–2025 is **τ = 0.8, σ = 0.4** (integrated match-NLL optimum τ = 0.8, σ = 0.6; contender-CRPS optimum τ = 0.7, σ = 0.3).

Under the effect-free likelihood the match-NLL and season-CRPS optima appeared to disagree on τ; integrating the effects removed most of that tension, leaving a small disagreement on σ only. A leak-free per-season τ by maximum likelihood on prior matches remains a useful diagnostic (0.60–0.75 through 2022, drifting toward 1.0 in 2023–2025; see §9).

### Effect shape

The **shape** of the persistent effects is itself a modelling choice. Holding (τ, σ) at the production values, replacing the Gaussian with a variance-normalised **Student-t (df = 4)** or a **10%/3× Gaussian mixture** changes the rolling record materially: contender CRPS improves from 2.684 to 2.628 and 90% coverage from 0.853 to 0.907, with the favourite hit rate unchanged (2 of 5) and the average winner probability moving from 0.241 to 0.232 (Student-t) or 0.225 (mixture). Re-calibrating the full grid under each shape selects the same production (τ = 0.8, σ = 0.4) and the two heavy-tailed options are statistically indistinguishable from each other (CRPS 2.628 versus 2.626; match NLL 5.112 versus 5.100). The Student-t(4) is adopted for parsimony; the mixture remains available through `--effect-distribution mixture`. The gain concentrates in 2023 (CRPS 2.37 → 2.05), the season whose count produced an extreme outcome the Gaussian could not generate often enough, and it survives in 2024 and 2025. Rolling records in §8 and the 2026 forecast use the Student-t(4) effects.

The historical-effect selection is shape-dependent: under the Gaussian the evidence selected shrinkage 100 / mapping 4, but under Student-t(4) it selects **shrinkage 50 / mapping 2** — with fatter tails already supplying extreme-season variance, a weaker residual nudge fits better. The shape-consistent rolling diagnostic (effect settings selected on evidence each season) scores **CRPS 2.55** with a **25.9% average winner probability** and 2/5 favourites, versus 2.63 / 23.2% / 2-of-5 effects-free. A fixed mapping-4 configuration reaches 2.50 / 31.7% / 3-of-5, but that configuration is selected on the full window and is **not** a clean rolling record (§8).

## 7. Uncertainty: fixing the original overconfidence

### What the original model did

The first-generation pipeline trained a feature-bagged XGBoost ensemble, ranked players within each match, and then ran 1,000 Monte Carlo counts by drawing each player-game independently from `Normal(pred_mean, pred_std²)`, where `pred_std` was the spread across the 100 ensemble members.

That design is structurally overconfident, for four reasons:

1. **`pred_std` measured the wrong thing.** The spread across models that share the same data, objective and hyperparameters reflects model *disagreement*, not the uncertainty of the outcome. Their errors are highly correlated, so the spread badly understates true predictive error.
2. **Independent per-game noise cancels over a season.** Idiosyncratic noise accumulates as `√n` over ~23 rounds and largely averages out. The uncertainty that decides a medal is **persistent**: a player the model systematically misjudges is misjudged in every round, and that bias accumulates as `n`. The simulation never varied the centre, only the noise around it.
3. **The distributional shape was wrong.** Votes are discrete, bounded `{0,1,2,3}`, and zero-inflated. A Normal surrogate has thin tails and symmetric mass; after ranking, it understates how often extreme outcomes (a huge game, a polling zero) occur.
4. **Monte Carlo precision masqueraded as model precision.** With 1,000 draws the simulated probabilities were smooth and repeatable (±1.6pp), which made them look authoritative while saying nothing about calibration.

### What the redesign does instead

- **Discrete allocation, not Gaussian noise.** Every simulation awards exactly 3-2-1 per match via Plackett–Luce. The noise model is the vote process itself, so the constant-sum competition and discreteness are structural, not approximate.
- **Calibrated temperature, but the gain is the score generator.** On the ranking scores, allocation log-loss is already 4.4–6.0 nats at τ=1 against 11.3–11.4 for a uniform allocation; rescaling alone moves it by only −0.20 to +0.29 nats per season. The large reduction from 8.5–9.6 nats (τ=1) to 7.5–7.6 belongs to the earlier pseudo-Huber regression baseline, whose score scale differs; that improvement is attributable to the score generator, not the temperature. τ and σ are now selected jointly on effect-integrated match NLL and contender CRPS (§6).
- **Persistent player-season effects.** Each simulation draws one deviation per player and holds it for **all** of their matches: `u = s + b`. Production uses a variance-normalised **Student-t distribution with 4 degrees of freedom** (σ = 0.4): heavier tails fit the rolling seasons better than a Gaussian at the same scale (CRPS 2.63 versus 2.68, 90% coverage 0.91 versus 0.85), and a 10%/3× mixture is statistically equivalent. On top of that, a **partially pooled historical player effect** `α_p` is added: a shrunken average of prior-season residuals — observed votes minus the model's own Plackett–Luce expectation — so a player the model chronically underrates is nudged up without double-counting raw vote totals. Shrinkage is in games and the residual history can decay; production uses 50 games with no decay and a mapping of 2 utility units per vote (selected under Student-t(4); the Gaussian selected 100/4). On rolling out-of-sample seasons the effect improved contender CRPS in **7 of 8 seasons** (mean −0.16) under the Gaussian and **6 of 8** (mean −0.10) under Student-t(4).
- **Simulation, then rolling validation.** For each target season the whole procedure is reproduced before its result is used: score model trained on earlier labels, τ/σ/α selected only on earlier out-of-sample seasons, then the season forecast once. Over 2021–2025 the production model's contender 90% intervals covered **91%** of observed totals (season range 73–100%) and 50% intervals covered **52%** (13–73%); the favourite won 2 of 5 and the eventual winner averaged a **23.2%** prior probability. Season-by-season coverage is reported rather than pooled. These records are post-selection diagnostics (§8).
- **Award-stage eligibility.** Suspended players are excluded from outright and joint-first determination (the next eligible player wins), while their simulated votes still count in the tally and affect everyone's totals. The 2012–2025 ineligible leading vote-getters are committed from the Wikipedia Brownlow articles (Fyfe 2014, Dangerfield 2017, Heeney 2024, and six others), and the 2026 list (32 players) from the published MRO tracker.
- **Sensitivity, not a single number.** Every forecast is reported across a small set of defensible specifications. For 2026, Nick Daicos's first-or-joint probability is 90.7% at the calibrated σ = 0.4 and spans 70–95% across σ = 0.3–0.7.
- **Quantiles, not extrema.** Reports use median, 50% and 90% intervals; simulated min/max were removed because they are unstable and widen with simulation count.

### What is still not modelled

Model-parameter uncertainty (uncertainty about the fitted score function itself) is approximated only through the persistent effect, the historical player effect and specification sensitivity. One trained model is used; a match-block bootstrap ensemble would capture shared model error more directly. The 2024 count remains the standing example: record vote inflation (Cripps 45) sat outside the centre of every specification, and 2024's 50% coverage fell to 0.20 even though the 90% band held. The historical effects are also the only player-specific direction in the model; they are learned from residuals but remain conservative.

## 8. Validation protocol and results

- **Rolling protocol.** For every target season `Y`: train the score model on labels `< Y`; select τ, σ and the player-effect settings using only earlier out-of-sample seasons under the predeclared criteria; forecast `Y` once and retain every prediction and outcome. Whole seasons move between training and validation, and the inner boosting-round selection is time-ordered as well (§5).
- **Reporting split.** Seasons 2015–2020 informed the original redesign and are development evidence; 2021–2025 are the rolling records of the final procedure; **2026 is the only live out-of-sample target** (votes pending). The earlier stats-only backtest is superseded by the variant comparison in §5 and the rolling table below.
- **Metrics**: effect-integrated match NLL, multiclass Brier for `P(0/1/2/3)`, top-1 hit rate, top-3 slot overlap, reliability of `P(polls)`, season CRPS, interval coverage and width, and eligibility-aware award hit rates.

Rolling score-model records (effects-free) for the frozen `ranking_season` configuration (Student-t(4) effects; grid on 2015–2025 evidence, each target season selecting its own parameters from earlier seasons). These are **post-selection diagnostics** — §8.1 quantifies the optimism:

| Season | τ, σ | Contender CRPS | 90% coverage | 50% coverage | Favourite won | Winner prior P |
|--------|------|----------------|--------------|--------------|---------------|----------------|
| 2021 | 0.7, 0.4 | 2.78 | 0.93 | 0.47 | yes | 44.0% |
| 2022 | 0.7, 0.5 | 1.87 | 1.00 | 0.73 | no | 17.0% |
| 2023 | 0.8, 0.3 | 2.05 | 0.93 | 0.53 | no | 1.9% |
| 2024 | 0.8, 0.3 | 3.79 | 0.73 | 0.13 | yes | 49.5% |
| 2025 | 0.8, 0.4 | 2.65 | 0.93 | 0.73 | no | 3.8% |
| **Mean** | | **2.63** | **0.91** | **0.52** | **2/5** | **23.2%** |

Per-match top-1 accuracy is 51–67%. The presented results do **not** establish an irreducible ceiling for umpire voting; the design goal is calibrated probabilities, not clairvoyance.

### 8.1 Selection leakage and honest bounds

The records above are diagnostics of a configuration that was *chosen* with knowledge of these same seasons. Coaches' votes, season-form features, the heavy-tailed effect shape and the effect settings were all adopted after seeing 2021–2025 results, so the diagnostics are optimistically biased by construction. Two evidence-only checks quantify the gap:

- **Effect shape, selected on evidence**: choosing Gaussian / Student-t / mixture per target using only earlier seasons never selects the Student-t; it scores 2.67 CRPS over 2021–2025 versus 2.63 for the fixed adopted choice.
- **Family, selected on evidence**: pooling all twelve candidate families (all-history, coaches, season-form, Student-t/mixture variants, context, form, direct PL, tier 1, CBA/recent/weighted ablations) and selecting (family, τ, σ) on evidence alone picks `ranking_tier1` for every target and scores **2.65 CRPS, 1/5 favourites, 19.3% winner probability** — the closest thing to an honest procedure-level bound, and evidence that the candidate differences are within selection noise.

With the historical player effects applied at the shape-consistent selection (50/2/0), the effect-adjusted diagnostic scores **CRPS 2.55, 25.9% winner probability, 2/5 favourites and 96% ninety-interval coverage**, with 2023 at 2.12 (versus 2.37 under the Gaussian effects-free protocol). The production configuration is **frozen** at the configuration described in §10 until the 2026 count; 2026 is the only untouched target.

### Approach comparison: legacy vs current vs ensemble

The first-generation design was reconstructed faithfully and rolled with the same protocol (100 feature-bagged XGBoost regressors on the original 120 features, independent Normal season counts from the ensemble spread; `legacy.py`). The stacking ensemble combines three members — the LambdaMART+coaches model, the direct Plackett–Luce model, and a random-forest regressor with coaches' votes — by globally standardised weighted utilities, with weights, τ and σ selected on earlier out-of-sample seasons by the same predeclared criterion. All three are scored on 2021–2025 with suspensions applied:

| Approach | Contender CRPS | 90% coverage | 50% coverage | Favourite won | Winner prior P | Match NLL | Rank-sum |
|---|---|---|---|---|---|---|---|
| Legacy (100-model Normal draws) | 4.77 | 0.20 | 0.09 | 0/5 | 0.000 | n/a | 12 |
| **Current production** (`ranking_season`, Student-t(4) effects) | **2.63** | **0.91** | **0.52** | **2/5** | **0.232** | **5.11** | **5** |
| Stacking ensemble | 2.65 | 0.93 | 0.57 | 1/5 | 0.186 | 5.26 | 7 |

The legacy baseline is overconfident in exactly the way the redesign diagnosed: a fifth of its 90% intervals covered and none of its five favourites won. Both redesign-era approaches are in a different class. The season-form production model edges the ensemble on CRPS and match NLL as well as the top-of-count signal; the ensemble's coverage is a touch closer at 90% (the comparison was run under Gaussian effects; the adopted Student-t(4) shape moves the production row by about 0.01 CRPS and does not change the ordering). A hindsight oracle that picks the best ensemble weights per season reaches CRPS 2.37, so **selection quality, not the member pool, is the binding constraint** — worth revisiting once more medal outcomes accumulate.

## 9. Modelling weak points

1. **Context-free utilities.** `s_{p,g}` depends on the player's own features (plus team shares); it does not see the other players in the match. The teammate competition is enforced at the allocation layer but not learned conditionally. Feeding teammate/opponent utility summaries into the score model (two-stage) or a conditional-logit design is the principled extension.
2. **Compositional errors.** Because each match allocates exactly 6 votes, errors are zero-sum within the match: overrating one player underrates another. The simulation also draws player effects independently, ignoring shared within-match model error.
3. **Extreme seasons still beat the centre.** Historical player effects are learned from prior-season residuals and improved 6 of 8 rolling seasons under the adopted shape (7 of 8 under the Gaussian), and heavy-tailed effects (adopted in §6) lifted 2023's CRPS from 2.37 to 2.05, but the model remains conservative: 2024 Cripps (45 votes) was ranked 1st before the count yet 2024's 50% coverage fell to 0.13 because the *magnitude* of the inflation was outside every specification. Strength-dependent overdispersion or a volatility-aware prior would target this directly, and 2025 Rowell (rank 6 by win probability) is the same failure in a different direction.
4. **Joint calibration removed most of the temperature tension.** With effect-integrated match probabilities, both the match-NLL and season-CRPS optima prefer τ ≈ 0.6–0.7 and differ only in σ (0.6 versus 0.3). The residual disagreement is genuine evidence about how much season-level spread is real rather than a tuning defect.
5. **No model-parameter uncertainty.** One trained model; no bootstrap ensemble; common bias and temporal drift are probed only via sensitivity analysis.
6. **Plackett–Luce assumptions.** IIA/restrictive substitution and no context effects. It is a clean, budget-exact baseline; alternatives (Bradley–Terry variants, conditional logit, dependent-error rank models) are testable upgrades.
7. **Label and identity noise.** Voting is judgmental; genuine zeros depend on crosswalk completeness. Unresolved rows are quarantined, but systematic mismatches would bias a match's allocation.
8. **Non-stationarity.** The feature→vote relationship drifts 2012–2026; era-missing features (CBA) can act as era proxies, and the within-season τ optimum drifted toward 1.0 in 2023–2025 while the rolling selection stayed stable at 0.7–0.8. Recency windows and recency weighting were tested and did not outperform all-history training.
9. **Missing covariates.** No opponent strength, venue/travel, umpire crews, or quarter-level splits (fourth-quarter disposals are not publicly available — see `data/README.md`).
10. **Award-stage eligibility edge cases.** In-season suspensions are applied, but the tribunal rules also cover finals and pre-season carry-overs, and the tracker is a live document that must be refreshed after each sitting. The committed 2026 list is one snapshot.
11. **Evaluation power.** Five rolling seasons with eligibility applied, and only a handful of independent medal outcomes; winner-probability calibration can be proxied, not proven. 2026 is the only live target.
12. **Post-season information boundary.** The production forecast observes match statistics **and** coaches' votes, so it is a count-night forecast rather than a pre-season or live mid-season one. Coaches' votes are published weekly, so a live variant can be updated round-by-round with a deliberately smaller information set.

## 10. Production parameters (2026 forecast)

| Parameter | Value |
|---|---|
| Score generator | match-grouped LambdaMART + AFLCA coaches' votes + leave-one-game-out season aggregates (139 features), trained 2012–2025 on 123,087 player-games |
| Temperature τ | 0.8 (joint calibration on 2015–2025 out-of-sample seasons) |
| Persistent effect scale σ | 0.4 (joint optimum; integrated-NLL optimum 0.6, contender-CRPS optimum 0.3) |
| Persistent effect shape | variance-normalised Student-t with 4 degrees of freedom (10%/3× mixture statistically equivalent) |
| Historical player effects | shrinkage 50 games, mapping 2 utility units per vote, no recency decay (selected under Student-t(4) on 2015–2025) |
| Freeze | production configuration frozen until the 2026 count; 2026 is the only target not used for any selection |
| Eligibility | applied: 32 players suspended during the 2026 home-and-away season excluded from win probabilities |
| Simulations | 10,000 Plackett–Luce counts |
| Data boundary | post-round-24: 2026 match statistics and AFLCA coaches' votes observed, only hidden umpire votes simulated |

Outputs include expected and median votes, 50% and 90% intervals, outright/joint/first-or-joint/top-5 probabilities, per-round `P(1)/P(2)/P(3)` for the interactive heatmap, and a specification sensitivity range. The interactive version is in [`web/`](../web) and at <https://jay-stein.github.io/brownlow-medal-predictor/>.

## Attribution

Data sources and credits are listed in [`NOTICE.md`](../NOTICE.md). The code is released under the [MIT License](../LICENSE).
