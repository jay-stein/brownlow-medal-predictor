"""Monte Carlo simulation of a season's Brownlow count.

Each simulation draws one persistent player-season effect per player and keeps
it for all of that player's matches. Drawing a fresh effect for every match
would average the season-level uncertainty away, which is exactly the failure
mode the redesign targets. Conditional on the resulting utilities, every match
receives a sequential Plackett-Luce allocation of 3, 2 and 1 votes.

When ``track_rounds`` is enabled the simulator also records cumulative vote
paths per round: running mean and quantiles, per-round increments, and a small
sample of full simulation paths for fan-chart visualisation.
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
ROUND_COLUMN = "ROUND_ROUNDNUMBER"

QUANTILES = (0.05, 0.25, 0.75, 0.95)
ROUND_QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)
DEFAULT_CONTENDERS = 15
DEFAULT_PATH_COUNT = 50


@dataclass
class SeasonSimulation:
    season: int
    players: pd.DataFrame
    totals: np.ndarray
    tau: float
    effect_scale: float
    rounds: list[int] | None = None
    cumulative_mean: np.ndarray | None = None
    cumulative_quantiles: np.ndarray | None = None
    increment_mean: np.ndarray | None = None
    increment_quantiles: np.ndarray | None = None
    path_totals: np.ndarray | None = None


def prepare_season(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, list[tuple[int, np.ndarray, np.ndarray]]]:
    """Return the player table and per-match ``(round, player indices, utilities)``."""
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
    matches: list[tuple[int, np.ndarray, np.ndarray]] = []
    for _, group in work.groupby(MATCH_COLUMN, sort=False):
        indices = np.array([positions[key] for key in group["PLAYER_KEY"]], dtype=int)
        utilities = group[UTILITY_COLUMN].to_numpy(dtype=float)
        round_number = int(group[ROUND_COLUMN].iloc[0])
        matches.append((round_number, indices, utilities))
    return players, matches


def _allocate_match(
    rng: np.random.Generator,
    indices: np.ndarray,
    utilities: np.ndarray,
    effects: np.ndarray,
    tau: float,
    totals: np.ndarray,
    n_sims: int,
) -> None:
    """Draw one ordered 3-2-1 allocation per simulation and add to ``totals``."""
    scores = utilities[:, None] + effects[indices]
    weights = np.exp((scores - scores.max(axis=0, keepdims=True)) / tau)
    simulation_index = np.arange(n_sims)
    for points in (3, 2, 1):
        cumulative = np.cumsum(weights, axis=0)
        draws = rng.random(n_sims) * cumulative[-1]
        picks = (cumulative < draws).sum(axis=0)
        np.add.at(totals, (indices[picks], simulation_index), points)
        weights[picks, simulation_index] = 0.0


def simulate_season(
    frame: pd.DataFrame,
    tau: float,
    effect_scale: float = 0.0,
    n_sims: int = 1000,
    seed: int = 42,
    track_rounds: bool = False,
    path_count: int = DEFAULT_PATH_COUNT,
) -> SeasonSimulation:
    """Simulate one season's count with a persistent player-season effect."""
    players, matches = prepare_season(frame)
    n_players = len(players)
    rng = np.random.default_rng(seed)
    if effect_scale > 0:
        effects = rng.normal(0.0, effect_scale, size=(n_players, n_sims))
    else:
        effects = np.zeros((n_players, n_sims))

    rounds = None
    cumulative_mean = None
    cumulative_quantiles = None
    increment_mean = None
    increment_quantiles = None
    path_totals = None

    if track_rounds:
        rounds = sorted({round_number for round_number, _, _ in matches})
        round_position = {round_number: index for index, round_number in enumerate(rounds)}
        totals = np.zeros((n_players, n_sims), dtype=np.int32)
        cumulative_mean = np.zeros((n_players, len(rounds)))
        cumulative_quantiles = np.zeros((n_players, len(rounds), len(ROUND_QUANTILES)))
        increment_mean = np.zeros((n_players, len(rounds)))
        increment_quantiles = np.zeros((n_players, len(rounds), len(ROUND_QUANTILES)))
        previous = np.zeros((n_players, n_sims), dtype=np.int32)
        increment = np.zeros((n_players, n_sims), dtype=np.int32)
        path_rng = np.random.default_rng(seed + 1)
        path_index = path_rng.choice(n_sims, size=min(path_count, n_sims), replace=False)
        path_totals = np.zeros((n_players, len(rounds), len(path_index)), dtype=np.int16)

        ordered = sorted(matches, key=lambda entry: round_position[entry[0]])
        pointer = 0
        for round_index, round_number in enumerate(rounds):
            while pointer < len(ordered) and ordered[pointer][0] == round_number:
                _, indices, utilities = ordered[pointer]
                _allocate_match(rng, indices, utilities, effects, tau, totals, n_sims)
                pointer += 1
            path_totals[:, round_index, :] = totals[:, path_index]
            cumulative_mean[:, round_index] = totals.mean(axis=1)
            cumulative_quantiles[:, round_index, :] = np.quantile(
                totals, ROUND_QUANTILES, axis=1
            ).T
            np.subtract(totals, previous, out=increment)
            increment_mean[:, round_index] = increment.mean(axis=1)
            increment_quantiles[:, round_index, :] = np.quantile(
                increment, ROUND_QUANTILES, axis=1
            ).T
            previous[:] = totals
    else:
        totals = np.zeros((n_players, n_sims), dtype=np.int32)
        for _, indices, utilities in matches:
            _allocate_match(rng, indices, utilities, effects, tau, totals, n_sims)

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
        rounds=rounds,
        cumulative_mean=cumulative_mean,
        cumulative_quantiles=cumulative_quantiles,
        increment_mean=increment_mean,
        increment_quantiles=increment_quantiles,
        path_totals=path_totals,
    )


def _rounded(value, digits: int = 3):
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value):
        return None
    return round(value, digits)


def forecast_export(
    simulation: SeasonSimulation,
    players: pd.DataFrame,
    *,
    top: int = 50,
    metadata: dict | None = None,
) -> dict:
    """Build a compact JSON-ready payload for the interactive web visualisation."""
    if simulation.rounds is None or simulation.cumulative_quantiles is None:
        raise ValueError("run simulate_season(track_rounds=True) before exporting")
    if simulation.path_totals is None:
        raise ValueError("simulation is missing sampled paths")

    position_of = {player_id: index for index, player_id in enumerate(simulation.players[PLAYER_COLUMN])}
    exported_players = []
    for _, row in players.head(top).iterrows():
        position = position_of.get(row[PLAYER_COLUMN])
        if position is None:
            continue
        cumulative = simulation.cumulative_quantiles[position]
        increment = simulation.increment_quantiles[position]
        paths = simulation.path_totals[position]
        exported_players.append(
            {
                "id": str(row[PLAYER_COLUMN]),
                "name": str(row["FULL_NAME"]).title(),
                "team": str(row["TEAM_NAME"]),
                "expectedVotes": _rounded(row.get("sim_mean")),
                "medianVotes": _rounded(row.get("sim_median")),
                "q05": _rounded(row.get("sim_q05")),
                "q25": _rounded(row.get("sim_q25")),
                "q75": _rounded(row.get("sim_q75")),
                "q95": _rounded(row.get("sim_q95")),
                "pOutright": _rounded(row.get("p_outright_first"), 4),
                "pFirstOrJoint": _rounded(row.get("p_first_or_joint"), 4),
                "pTop5": _rounded(row.get("p_top5"), 4),
                "winLow": _rounded(row.get("p_first_or_joint_min"), 4),
                "winHigh": _rounded(row.get("p_first_or_joint_max"), 4),
                "rounds": {
                    "cumMean": [_rounded(value) for value in simulation.cumulative_mean[position]],
                    "cumQ05": [_rounded(value) for value in cumulative[:, 0]],
                    "cumQ25": [_rounded(value) for value in cumulative[:, 1]],
                    "cumMedian": [_rounded(value) for value in cumulative[:, 2]],
                    "cumQ75": [_rounded(value) for value in cumulative[:, 3]],
                    "cumQ95": [_rounded(value) for value in cumulative[:, 4]],
                    "incMean": [_rounded(value) for value in simulation.increment_mean[position]],
                    "incQ05": [_rounded(value) for value in increment[:, 0]],
                    "incQ25": [_rounded(value) for value in increment[:, 1]],
                    "incMedian": [_rounded(value) for value in increment[:, 2]],
                    "incQ75": [_rounded(value) for value in increment[:, 3]],
                    "incQ95": [_rounded(value) for value in increment[:, 4]],
                    "paths": [
                        [int(value) for value in paths[:, index]]
                        for index in range(paths.shape[1])
                    ],
                },
            }
        )

    return {
        "season": simulation.season,
        "meta": metadata or {},
        "rounds": [
            {"number": int(number), "label": "OR" if number == 0 else f"R{int(number)}"}
            for number in simulation.rounds
        ],
        "players": exported_players,
    }


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
