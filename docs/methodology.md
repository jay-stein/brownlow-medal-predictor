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

With the production temperature `τ = 0.845`, consider a match where player X has utility 2.0 and compare an elite teammate (utility 1.9) with quiet teammates:

| Scenario | P(X gets 3) | E[X] | E[others] |
|---|---|---|---|
| Elite teammate (1.9) | 46.5% | 2.30 | 2.22 / 0.95 / 0.54 |
| Quiet teammates (0.3 / 0.2) | **74.3%** | **2.69** | 1.24 / 1.14 / 0.94 |

The elite teammate costs X ~0.4 expected votes and ~28 percentage points of P(3) *per game* — roughly 9 votes across a season. It is why a slightly lesser player who is the sole focal point can out-poll a superstar sharing a midfield.

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

## 5. Score generator

A **LambdaMART ranker** (`rank:ndcg`, XGBoost) produces one real-valued utility `s_{p,g}` per player-game, with each match as a ranking group (`qid`) and relevance labels 0–3 votes. It replaced a pseudo-Huber / MAE regression baseline because it optimises **within-match ordering** — the quantity the 3-2-1 allocation actually depends on. Held out chronologically, it beat the baseline on every probability metric.

Training protocol:

- For a target season `Y`, train only on labelled seasons `< Y`.
- Inside the training window, `GroupKFold` over matches selects the number of boosting rounds (mean best iteration across folds); early stopping uses seeded folds.
- Preprocessing is fit on training seasons only: numeric imputation means, categorical encodings, and the >5% missing-column drop are all train-derived. Match grouping means no match ever spans a fold.
- The score's absolute scale is arbitrary (ranking is invariant to monotone transforms). A **positive rescaling** `s → s/τ` is absorbed by the temperature; a general monotone transform is not, because the allocation layer depends on the shape of the gaps between scores, not only their scale.

## 6. Plackett–Luce allocation and temperature

Given utilities, the match distribution is the sequential Plackett–Luce model in §1. Marginal probabilities of receiving 0/1/2/3 votes are available in closed form by direct summation over the first two placements (`O(n²)` per match). The sequential denominator depends on the preceding picks, so symmetric-polynomial shortcuts from other weighted-subset models are **not** valid here; the implementation is audited against exhaustive enumeration on small synthetic matches and against Gumbel-max Monte Carlo simulation on full-size matches, and satisfies the exact invariants (each vote slot sums to one across players; expected match votes sum to six).

The **temperature τ** controls sharpness. It is fitted by maximum likelihood on the observed ordered triples:

```text
τ_Y = argmin_τ  −Σ_matches log P(observed 3-2-1 | τ)
```

fitting is **leak-free per season**: for season `Y`, τ is fitted on out-of-sample scores from seasons `< Y` only. Fitted values are stable at roughly 0.75–0.85 (2026 production fit: **0.845** on 2,555 out-of-sample matches).

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
- **Calibrated temperature — but the gain is the score generator, not τ.** On the production ranking scores, allocation log-loss is already **4.4–6.0 nats at τ=1** against 11.3–11.4 for a uniform allocation. Fitting τ (leak-free values 0.75–0.83) changes that by only −0.20 to +0.29 nats per season — within the ~0.5–0.9 nats bound on what positive rescaling alone can achieve — and in 2023–2025 the leak-free τ slightly *worsens* NLL because the within-season optimum has drifted towards 1.0. The larger reduction from 8.5–9.6 nats (τ=1) to 7.5–7.6 belongs to the earlier pseudo-Huber regression baseline, whose score scale differs; that improvement is attributable to replacing the baseline with the ranking score generator. Brier scores follow the same pattern (ranking 0.073–0.089 versus baseline ~0.106). Temperature is a one-parameter scaling correction, not a substitute for score quality.
- **Persistent player-season effects.** Each simulation draws one deviation `b_p^(j) ~ N(0, σ²)` per player and holds it for **all** of their matches: `u_{p,g} = s_{p,g} + b_p^(j)`. Drawing a fresh effect per match would average the season-level uncertainty away — exactly the original failure mode. The scale σ is not assumed: it is chosen by a grid against historical season-total distributions (contender CRPS), selected at **0.4** on 2015–2023.
- **Simulation, then validation by coverage.** Season totals are built from 10,000 simulated counts; the claim "the intervals are honest" is then tested on held-out seasons rather than asserted. Over 2015–2025:
  - contender 90% intervals covered **88.5%** of observed season totals (nominal 90%)
  - contender 50% intervals covered **48.6%** (nominal 50%)
  - simulated-versus-observed rank correlation 0.73–0.78 in every season
- **Honest award probabilities.** Across the same seasons the model's favourite won **5 of 11** times, and the eventual winner carried an average **36%** pre-count probability. The output is deliberately not a single confident name.
- **Sensitivity, not a single number.** Every forecast is reported across a small set of historically defensible specifications (effect scale 0–0.4). For 2026, Nick Daicos's first-or-joint probability is 79.9% at the calibrated scale but spans 80–92% across specifications — the range is part of the result.
- **Quantiles, not extrema.** Reports use median, 50% and 90% intervals; simulated min/max were removed because they are unstable and widen with simulation count.

### What is still not modelled

Model-parameter uncertainty (uncertainty about the fitted score function itself) is approximated only through the persistent effect and specification sensitivity. A match-block bootstrap ensemble of score generators (planned Phase 3b) would capture it more directly. The persistent effect is also Gaussian and zero-mean, so it carries no learned direction for any particular player. It is not neutral for expected votes — the nonlinear allocation means zero-mean utility noise changes vote means — but those shifts are symmetric in expectation; partially pooled historical player intercepts would give the effect a learned direction.

## 8. Validation protocol and results

- **Folds**: development seasons 2015–2024 (train on all earlier labels), final untouched 2025, production fit 2012–2025 → 2026 forecast.
- **Metrics**: match allocation log-loss (NLL of the observed triple), multiclass Brier for `P(0/1/2/3)`, top-1 hit rate, top-3 slot overlap, reliability of `P(polls)`, season CRPS, interval coverage, and award-level hit rates.

Held-out results for the production (`ranking`) score generator:

| Season | Allocation NLL | Skill vs uniform | Top-1 hit | Top-3 slot overlap |
|--------|----------------|------------------|-----------|--------------------|
| 2015 | 5.08 | 0.55 | 0.59 | 0.66 |
| 2016 | 4.64 | 0.59 | 0.63 | 0.69 |
| 2017 | 4.58 | 0.59 | 0.60 | 0.69 |
| 2018 | 4.83 | 0.57 | 0.55 | 0.69 |
| 2019 | 4.53 | 0.60 | 0.61 | 0.70 |
| 2020 | 5.62 | 0.50 | 0.57 | 0.64 |
| 2021 | 4.79 | 0.58 | 0.67 | 0.69 |
| 2022 | 4.36 | 0.62 | 0.62 | 0.70 |
| 2023 | 6.16 | 0.46 | 0.51 | 0.60 |
| 2024 | 5.80 | 0.49 | 0.52 | 0.61 |
| 2025 (untouched) | 5.50 | 0.52 | 0.51 | 0.64 |

Per-match top-1 accuracy is 51–67%. The presented results do **not** establish an irreducible ceiling for umpire voting; the design goal is calibrated probabilities, not clairvoyance.

## 9. Modelling weak points

1. **Context-free utilities.** `s_{p,g}` depends on the player's own features (plus team shares); it does not see the other players in the match. The teammate competition is enforced at the allocation layer but not learned conditionally. Feeding teammate/opponent utility summaries into the score model (two-stage) or a conditional-logit design is the principled extension.
2. **Compositional errors.** Because each match allocates exactly 6 votes, errors are zero-sum within the match: overrating one player underrates another. The simulation also draws player effects independently, ignoring shared within-match model error.
3. **Persistent effects are zero-mean and temporally unstructured.** Extreme seasons (2024 Cripps 45, 2025 Rowell 39) were ranked 2nd and 9th; a symmetric effect widens the distribution without adding player-specific direction. Note that zero-mean utility effects are **not** neutral for expected votes — the nonlinear allocation can shift means — but they contain no learned correction for a particular player. Partially pooled player intercepts (historical polling priors) or strength-dependent overdispersion target this directly.
4. **Temperature only sets scale.** Match-likelihood and season-CRPS optima for τ diverge (sharper τ improves season CRPS but worsens match NLL) — evidence of missing concentration structure rather than a tuning problem.
5. **No model-parameter uncertainty.** One trained model; no bootstrap ensemble; common bias and temporal drift are probed only via sensitivity analysis.
6. **Plackett–Luce assumptions.** IIA/restrictive substitution and no context effects. It is a clean, budget-exact baseline; alternatives (Bradley–Terry variants, conditional logit, dependent-error rank models) are testable upgrades.
7. **Label and identity noise.** Voting is judgmental; genuine zeros depend on crosswalk completeness. Unresolved rows are quarantined, but systematic mismatches would bias a match's allocation.
8. **Non-stationarity.** The feature→vote relationship drifts 2012–2026; era-missing features (CBA) can act as era proxies; feature coverage varies by era.
9. **Missing covariates.** No opponent strength, venue/travel, umpire crews, or quarter-level splits (fourth-quarter disposals are not publicly available — see `data/README.md`).
10. **Award-stage gaps.** Suspension eligibility is not applied; joint-first and top-5 ties are simulated but the medal's exact eligibility rules would need an in-season source.
11. **Evaluation power.** Eleven seasons, one untouched fold (2025), and only a handful of independent medal outcomes; winner-probability calibration can be proxied, not proven.

## 10. Production parameters (2026 forecast)

| Parameter | Value |
|---|---|
| Score generator | match-grouped LambdaMART (`rank:ndcg`), trained 2012–2025 on 123,087 player-games |
| Temperature τ | 0.845 (leak-free, 2,555 out-of-sample matches) |
| Persistent effect scale | 0.4 (contender-CRPS grid, tuned 2015–2023) |
| Simulations | 10,000 Plackett–Luce counts |
| Data boundary | post-round-24: 2026 match statistics observed, only hidden votes simulated |

Outputs include expected and median votes, 50% and 90% intervals, outright/joint/first-or-joint/top-5 probabilities, per-round `P(1)/P(2)/P(3)` for the interactive heatmap, and a specification sensitivity range. The interactive version is in [`web/`](../web) and at <https://jay-stein.github.io/brownlow-medal-predictor/>.

## Attribution

Data sources and credits are listed in [`NOTICE.md`](../NOTICE.md). The code is released under the [MIT License](../LICENSE).
