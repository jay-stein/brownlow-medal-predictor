"""Monte Carlo simulation of a season's Brownlow count.

Each simulation draws one persistent player-season effect per player and keeps
it for all of that player's matches. Drawing a fresh effect for every match
would average the season-level uncertainty away, which is exactly the failure
mode the redesign targets. Conditional on the resulting utilities, every match
receives a sequential Plackett-Luce allocation of 3, 2 and 1 votes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import evaluate
from .scores import UTILITY_COLUMN

VOTE_COLUMN = "BROWNLOW_VOTES_AUDITED"
MATCH_COLUMN = "PROVIDERID"
PLAYER_COLUMN = "PLAYER_PLAYER_PLAYER_PLAYERID"
SEASON_COLUMN = "ROUND_YEAR"

QUANTILES = (0.05, 0.25, 0.75, 0.95)
DEFAULT_CONTENDERS = 15


@dataclass
class SeasonSimulation:
    season: int
    players: pd.DataFrame
    totals: np.ndarray
    tau: float
    effect_scale: float


def prepare_season(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[np.ndarray, np.ndarray]]]:
    """Return the player table and per-match ``(player indices, utilities)``."""
    work = frame.copy()
    work = work[work[PLAYER_COLUMN].notna()].copy()
    work["PLAYER_KEY"] = work[PLAYER_COLUMN].astype(str)
    players = (
        work.groupby("PLAYER_KEY", as_index=False)
        .agg(FULL_NAME=("FULL_NAME", "last"), TEAM_NAME=("TEAM_NAME", "last"))
    )
    observed = work.groupby("PLAYER_KEY")[VOTE_COLUMN].sum(min_count=1).rename("observed_votes")
    players = players.merge(observed, on="PLAYER_KEY")
    players = players.rename(columns={"PLAYER_KEY": PLAYER_COLUMN})

    positions = {key: index for index, key in enumerate(players[PLAYER_COLUMN])}
    matches: list[tuple[np.ndarray, np.ndarray]] = []
    for _, group in work.groupby(MATCH_COLUMN, sort=False):
        indices = np.array([positions[key] for key in group["PLAYER_KEY"]], dtype=int)
        utilities = group[UTILITY_COLUMN].to_numpy(dtype=float)
        matches.append((indices, utilities))
    return players, matches


def simulate_season(
    frame: pd.DataFrame,
    tau: float,
    effect_scale: float = 0.0,
    n_sims: int = 1000,
    seed: int = 42,
) -> SeasonSimulation:
    """Simulate one season's count with a persistent player-season effect."""
    players, matches = prepare_season(frame)
    n_players = len(players)
    rng = np.random.default_rng(seed)
    if effect_scale > 0:
        effects = rng.normal(0.0, effect_scale, size=(n_players, n_sims))
    else:
        effects = np.zeros((n_players, n_sims))

    totals = np.zeros((n_players, n_sims), dtype=np.int32)
    simulation_index = np.arange(n_sims)

    for indices, utilities in matches:
        scores = utilities[:, None] + effects[indices]
        weights = np.exp((scores - scores.max(axis=0, keepdims=True)) / tau)
        for points in (3, 2, 1):
            cumulative = np.cumsum(weights, axis=0)
            draws = rng.random(n_sims) * cumulative[-1]
            picks = (cumulative < draws).sum(axis=0)
            np.add.at(totals, (indices[picks], simulation_index), points)
            weights[picks, simulation_index] = 0.0

    summary = players.copy()
    summary["sim_mean"] = totals.mean(axis=1)
    summary["sim_median"] = np.median(totals, axis=1)
    for quantile in QUANTILES:
        summary[f"sim_q{int(quantile * 100):02d}"] = np.quantile(totals, quantile, axis=1)
    leaders = totals == totals.max(axis=0, keepdims=True)
    summary["p_outright_first"] = (leaders & (leaders.sum(axis=0, keepdims=True) == 1)).mean(axis=1)
    summary["p_first_or_joint"] = leaders.mean(axis=1)
    top_rank = min(5, totals.shape[0])
    top5_threshold = np.sort(totals, axis=0)[-top_rank, :]
    summary["p_top5"] = (totals >= top5_threshold[None, :]).mean(axis=1)

    return SeasonSimulation(
        season=int(frame[SEASON_COLUMN].iloc[0]),
        players=summary,
        totals=totals,
        tau=float(tau),
        effect_scale=float(effect_scale),
    )


def season_metrics(
    simulation: SeasonSimulation,
    contender_count: int = DEFAULT_CONTENDERS,
) -> dict:
    """CRPS and interval coverage for season vote totals, overall and for contenders."""
    totals = simulation.totals
    observed = simulation.players["observed_votes"].to_numpy(dtype=float)
    labelled = ~np.isnan(observed)
    contender_mask = labelled & np.isin(
        np.arange(len(observed)),
        np.argsort(-totals.mean(axis=1))[:contender_count],
    )

    def mean_crps(mask: np.ndarray) -> float:
        indices = np.flatnonzero(mask)
        if len(indices) == 0:
            return float("nan")
        return float(np.mean([evaluate.crps_ensemble(totals[i], observed[i]) for i in indices]))

    result = {
        "n_players": int(labelled.sum()),
        "mean_crps": mean_crps(labelled),
        "mean_crps_contenders": mean_crps(contender_mask),
    }
    for level in (0.5, 0.9):
        label = int(level * 100)
        for suffix, mask in (("", labelled), ("_contenders", contender_mask)):
            indices = np.flatnonzero(mask)
            if len(indices) == 0:
                result[f"coverage_{label}{suffix}"] = float("nan")
                result[f"width_{label}{suffix}"] = float("nan")
                continue
            pairs = [evaluate.interval_coverage(totals[i], observed[i], level) for i in indices]
            result[f"coverage_{label}{suffix}"] = float(np.mean([pair[0] for pair in pairs]))
            result[f"width_{label}{suffix}"] = float(np.mean([pair[1] for pair in pairs]))
    return result


def calibrate_effect_scales(
    scores_by_season: dict[int, pd.DataFrame],
    taus: dict[int, float],
    scales: list[float],
    seasons: list[int],
    n_sims: int = 1000,
    seed: int = 42,
    contender_count: int = DEFAULT_CONTENDERS,
) -> pd.DataFrame:
    """Evaluate a grid of player-season effect scales on historical seasons."""
    rows: list[dict] = []
    for scale in scales:
        for season in seasons:
            if season not in scores_by_season or season not in taus:
                continue
            simulation = simulate_season(
                scores_by_season[season],
                taus[season],
                effect_scale=scale,
                n_sims=n_sims,
                seed=seed,
            )
            metrics = season_metrics(simulation, contender_count=contender_count)
            rows.append(
                {"effect_scale": float(scale), "season": season, "tau": taus[season], **metrics}
            )
    return pd.DataFrame(rows)
