"""Joint temperature / effect-scale calibration and fully rolling validation.

The protocol implemented here:

- a shared grid of temperature ``tau`` and persistent-effect scale ``sigma``
  is scored on chronological out-of-sample seasons with two metrics:
  effect-integrated match NLL (the observed ordered triple, marginalising over
  the same persistent player-season effects the simulator draws) and contender
  CRPS of the simulated season totals;
- the two metrics are standardised across the grid and combined with
  predeclared equal weights; the pair with the lowest combined score wins,
  with lower match NLL breaking ties;
- for rolling validation each target season reproduces the entire procedure
  using only earlier seasons as evidence, then forecasts the target once.

Contenders are defined by simulated mean at forecast time, never by the
observed leaderboard, so the rolling records contain no hindsight.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import history as history_effects
from . import simulate

DEFAULT_TAU_GRID: tuple[float, ...] = (0.6, 0.7, 0.8, 0.9, 1.0, 1.1)
DEFAULT_SCALE_GRID: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
MIN_EVIDENCE_SEASONS = 3
NLL_WEIGHT = 0.5
CRPS_WEIGHT = 0.5

ROLLING_COLUMNS = [
    "season",
    "tau_selected",
    "scale_selected",
    "evidence_seasons",
    "mean_crps_contenders",
    "coverage_50_contenders",
    "coverage_90_contenders",
    "width_50_contenders",
    "width_90_contenders",
    "match_nll",
    "favorite_won",
    "winner_probability",
    "winner_mean_rank",
]


def joint_calibration_grid(
    scores_by_season: dict[int, pd.DataFrame],
    seasons: list[int],
    tau_values: tuple[float, ...] | list[float] = DEFAULT_TAU_GRID,
    scales: tuple[float, ...] | list[float] = DEFAULT_SCALE_GRID,
    *,
    n_sims: int = 2000,
    n_draws: int = 256,
    contender_count: int = simulate.DEFAULT_CONTENDERS,
    seed: int = 42,
    effect_distribution: str = "normal",
    effect_t_df: float = 4.0,
    effect_mixture_prob: float = simulate.MIXTURE_PROB,
    effect_mixture_multiplier: float = simulate.MIXTURE_MULTIPLIER,
) -> pd.DataFrame:
    """Score the joint (tau, scale) grid on the requested seasons.

    All candidates for a season share the same effect draws and simulation
    seed, so differences are due to the parameters rather than Monte Carlo
    noise. The persistent-effect distribution shape is held fixed and recorded
    on every row.
    """
    rows: list[dict] = []
    for tau in tau_values:
        for scale in scales:
            for season in seasons:
                if season not in scores_by_season:
                    continue
                frame = scores_by_season[season]
                match_nll = simulate.integrated_match_log_loss(
                    frame,
                    float(tau),
                    float(scale),
                    n_draws=n_draws,
                    seed=seed,
                    effect_distribution=effect_distribution,
                    effect_t_df=effect_t_df,
                    effect_mixture_prob=effect_mixture_prob,
                    effect_mixture_multiplier=effect_mixture_multiplier,
                )
                simulation = simulate.simulate_season(
                    frame,
                    float(tau),
                    effect_scale=float(scale),
                    n_sims=n_sims,
                    seed=seed,
                    effect_distribution=effect_distribution,
                    effect_t_df=effect_t_df,
                    effect_mixture_prob=effect_mixture_prob,
                    effect_mixture_multiplier=effect_mixture_multiplier,
                )
                metrics = simulate.season_metrics(simulation, contender_count=contender_count)
                rows.append(
                    {
                        "tau": float(tau),
                        "effect_scale": float(scale),
                        "season": int(season),
                        "effect_distribution": effect_distribution,
                        "effect_t_df": float(effect_t_df),
                        "effect_mixture_prob": float(effect_mixture_prob),
                        "effect_mixture_multiplier": float(effect_mixture_multiplier),
                        "match_nll": match_nll,
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


@dataclass
class JointSelection:
    tau: float
    effect_scale: float
    summary: pd.DataFrame
    best_match_nll_tau: float
    best_match_nll_scale: float
    best_crps_tau: float
    best_crps_scale: float


def _standardise(summary: pd.DataFrame, nll_weight: float, crps_weight: float) -> pd.DataFrame:
    """Standardise both metrics across candidates and combine them."""
    for column in ("match_nll", "crps"):
        std = float(summary[column].std(ddof=0))
        summary[f"z_{column}"] = (summary[column] - summary[column].mean()) / (
            std if std > 0.0 else 1.0
        )
    summary["combined"] = nll_weight * summary["z_match_nll"] + crps_weight * summary["z_crps"]
    return summary


def select_joint_params(
    grid: pd.DataFrame,
    evidence_seasons: list[int],
    *,
    nll_weight: float = NLL_WEIGHT,
    crps_weight: float = CRPS_WEIGHT,
) -> JointSelection:
    """Select (tau, scale) from the grid using only the evidence seasons.

    Both metrics are standardised across the candidate grid before combining,
    so the weights are on comparable scales. Equal weights are predeclared;
    the raw optima are retained to report the match/season tension.
    """
    subset = grid[grid["season"].isin(evidence_seasons)]
    if subset.empty:
        raise ValueError("no grid rows for the evidence seasons")
    summary = (
        subset.groupby(["tau", "effect_scale"], as_index=False)
        .agg(match_nll=("match_nll", "mean"), crps=("mean_crps_contenders", "mean"))
    )
    summary = _standardise(summary, nll_weight, crps_weight)
    best = summary.sort_values(["combined", "match_nll"]).iloc[0]
    best_nll = summary.sort_values(["match_nll", "crps"]).iloc[0]
    best_crps = summary.sort_values(["crps", "match_nll"]).iloc[0]
    return JointSelection(
        tau=float(best["tau"]),
        effect_scale=float(best["effect_scale"]),
        summary=summary,
        best_match_nll_tau=float(best_nll["tau"]),
        best_match_nll_scale=float(best_nll["effect_scale"]),
        best_crps_tau=float(best_crps["tau"]),
        best_crps_scale=float(best_crps["effect_scale"]),
    )


def rolling_validation(
    grid: pd.DataFrame,
    *,
    report_seasons: list[int],
    min_evidence: int = MIN_EVIDENCE_SEASONS,
) -> pd.DataFrame:
    """Reproduce the full selection procedure for each target season.

    Season Y's (tau, scale) is selected on grid rows from seasons strictly
    earlier than Y only; the record for Y is then read off the grid at those
    parameters. Seasons with fewer than ``min_evidence`` prior seasons are
    skipped rather than given a hindsight-informed setting.
    """
    all_seasons = sorted(int(season) for season in grid["season"].unique())
    rows: list[dict] = []
    for season in sorted(report_seasons):
        evidence = [candidate for candidate in all_seasons if candidate < season]
        if len(evidence) < min_evidence:
            continue
        selection = select_joint_params(grid, evidence)
        match = grid[
            (grid["season"] == season)
            & (grid["tau"] == selection.tau)
            & (grid["effect_scale"] == selection.effect_scale)
        ]
        if match.empty:
            continue
        record = match.iloc[0].to_dict()
        record.update(
            {
                "tau_selected": selection.tau,
                "scale_selected": selection.effect_scale,
                "evidence_seasons": len(evidence),
                "evidence_first": min(evidence),
                "evidence_last": max(evidence),
                "nll_opt_tau": selection.best_match_nll_tau,
                "nll_opt_scale": selection.best_match_nll_scale,
                "crps_opt_tau": selection.best_crps_tau,
                "crps_opt_scale": selection.best_crps_scale,
                "optimiser_tension": float(
                    np.hypot(
                        selection.best_match_nll_tau - selection.best_crps_tau,
                        selection.best_match_nll_scale - selection.best_crps_scale,
                    )
                ),
            }
        )
        rows.append(record)
    return pd.DataFrame(rows)


PLAYER_EFFECT_COLUMNS = [
    "season",
    "shrinkage_selected",
    "mapping_selected",
    "half_life_selected",
    "evidence_seasons",
    "mean_crps_contenders",
    "baseline_crps_contenders",
    "delta_crps_contenders",
    "coverage_90_contenders",
    "baseline_coverage_90_contenders",
    "match_nll",
    "baseline_match_nll",
    "favorite_won",
    "winner_mean_rank",
]


@dataclass
class PlayerEffectSelection:
    shrinkage: float
    mapping: float
    half_life: float
    summary: pd.DataFrame


def player_effect_grid(
    scores_by_season: dict[int, pd.DataFrame],
    residual_history: pd.DataFrame,
    seasons: list[int],
    *,
    tau: float,
    effect_scale: float,
    candidates: list[tuple[float, float, float]],
    n_sims: int = 2000,
    n_draws: int = 256,
    contender_count: int = simulate.DEFAULT_CONTENDERS,
    seed: int = 42,
) -> pd.DataFrame:
    """Score historical-effect candidates on the requested seasons.

    ``candidates`` are ``(shrinkage, mapping, half_life)`` triples; mapping
    zero is the unadjusted baseline. Each candidate's adjustment for a season
    is built only from seasons before it.
    """
    rows: list[dict] = []
    for shrinkage, mapping, half_life in candidates:
        for season in seasons:
            if season not in scores_by_season:
                continue
            adjusted = history_effects.attach_player_effects(
                scores_by_season[season],
                residual_history,
                season,
                shrinkage=shrinkage,
                mapping=mapping,
                half_life=half_life,
            )
            match_nll = simulate.integrated_match_log_loss(
                adjusted, tau, effect_scale, n_draws=n_draws, seed=seed
            )
            simulation = simulate.simulate_season(
                adjusted, tau, effect_scale=effect_scale, n_sims=n_sims, seed=seed
            )
            metrics = simulate.season_metrics(simulation, contender_count=contender_count)
            rows.append(
                {
                    "shrinkage": float(shrinkage),
                    "mapping": float(mapping),
                    "half_life": float(half_life),
                    "season": int(season),
                    "match_nll": match_nll,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def select_player_effects(
    grid: pd.DataFrame,
    evidence_seasons: list[int],
    *,
    nll_weight: float = NLL_WEIGHT,
    crps_weight: float = CRPS_WEIGHT,
) -> PlayerEffectSelection:
    """Select historical-effect settings using only the evidence seasons."""
    subset = grid[grid["season"].isin(evidence_seasons)]
    if subset.empty:
        raise ValueError("no grid rows for the evidence seasons")
    summary = subset.groupby(["shrinkage", "mapping", "half_life"], as_index=False).agg(
        match_nll=("match_nll", "mean"), crps=("mean_crps_contenders", "mean")
    )
    summary = _standardise(summary, nll_weight, crps_weight)
    best = summary.sort_values(["combined", "match_nll"]).iloc[0]
    return PlayerEffectSelection(
        shrinkage=float(best["shrinkage"]),
        mapping=float(best["mapping"]),
        half_life=float(best["half_life"]),
        summary=summary,
    )


def rolling_player_effect_validation(
    grid: pd.DataFrame,
    *,
    report_seasons: list[int],
    min_evidence: int = MIN_EVIDENCE_SEASONS,
    baseline_mapping: float = 0.0,
) -> pd.DataFrame:
    """Rolling selection of historical effects with a paired baseline per season.

    The baseline for each season is the unadjusted candidate (mapping zero),
    evaluated on the same season with the same seed, so the difference is the
    effect of the adjustment rather than simulation noise.
    """
    all_seasons = sorted(int(season) for season in grid["season"].unique())
    rows: list[dict] = []
    for season in sorted(report_seasons):
        evidence = [candidate for candidate in all_seasons if candidate < season]
        if len(evidence) < min_evidence:
            continue
        selection = select_player_effects(grid, evidence)
        adjusted = grid[
            (grid["season"] == season)
            & (grid["shrinkage"] == selection.shrinkage)
            & (grid["mapping"] == selection.mapping)
            & (grid["half_life"] == selection.half_life)
        ]
        baseline = grid[(grid["season"] == season) & (grid["mapping"] == baseline_mapping)]
        if adjusted.empty or baseline.empty:
            continue
        record = adjusted.iloc[0].to_dict()
        base = baseline.iloc[0].to_dict()
        record.update(
            {
                "shrinkage_selected": selection.shrinkage,
                "mapping_selected": selection.mapping,
                "half_life_selected": selection.half_life,
                "evidence_seasons": len(evidence),
                "baseline_crps_contenders": base["mean_crps_contenders"],
                "baseline_coverage_90_contenders": base["coverage_90_contenders"],
                "baseline_match_nll": base["match_nll"],
                "delta_crps_contenders": (
                    record["mean_crps_contenders"] - base["mean_crps_contenders"]
                ),
            }
        )
        rows.append(record)
    return pd.DataFrame(rows)
