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
from scipy.special import logsumexp

from . import evaluate, pl
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
TRIPLE_CANDIDATES = 40


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
    round_p1: np.ndarray | None = None
    round_p2: np.ndarray | None = None
    round_p3: np.ndarray | None = None
    match_groups: list[MatchGroup] | None = None
    match_slot_counts: list[np.ndarray] | None = None
    match_triple_counts: list[dict[int, int]] | None = None
    player_count: int | None = None
    team_names: list[str] | None = None
    team_cumulative_mean: np.ndarray | None = None
    team_cumulative_quantiles: np.ndarray | None = None
    team_increment_mean: np.ndarray | None = None
    team_increment_quantiles: np.ndarray | None = None
    team_path_totals: np.ndarray | None = None


@dataclass
class MatchGroup:
    match_id: str
    round_number: int
    indices: np.ndarray
    utilities: np.ndarray
    triple: tuple[int, int, int] | None
    home_team: str | None = None
    away_team: str | None = None
    home_score: float | None = None
    away_score: float | None = None
    game_date: str | None = None
    venue: str | None = None


def _first_value(group: pd.DataFrame, column: str):
    if column not in group.columns:
        return None
    value = group[column].iloc[0]
    if pd.isna(value):
        return None
    return value


def prepare_season(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, list[MatchGroup]]:
    """Return the player table and per-match groups.

    ``indices`` are positions into the player table, ``utilities`` are the
    match's scored player-games, and ``triple`` is the observed ordered 3-2-1
    recipient positions when the match has an evaluable label (``None``
    otherwise).
    """
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
    matches: list[MatchGroup] = []
    for match_id, group in work.groupby(MATCH_COLUMN, sort=False):
        indices = np.array([positions[key] for key in group["PLAYER_KEY"]], dtype=int)
        utilities = group[UTILITY_COLUMN].to_numpy(dtype=float)
        round_number = int(group[ROUND_COLUMN].iloc[0])
        votes = group[VOTE_COLUMN].to_numpy(dtype=float)
        recipients = np.flatnonzero(np.nan_to_num(votes) > 0.0)
        triple = None
        if len(recipients) == 3 and sorted(votes[recipients].tolist()) == [1.0, 2.0, 3.0]:
            order = recipients[np.argsort(-votes[recipients], kind="stable")]
            triple = tuple(int(index) for index in order)
        matches.append(
            MatchGroup(
                match_id=str(match_id),
                round_number=round_number,
                indices=indices,
                utilities=utilities,
                triple=triple,
                home_team=_first_value(group, "HOME_TEAM_NAME"),
                away_team=_first_value(group, "AWAY_TEAM_NAME"),
                home_score=_first_value(group, "HOMETEAMSCORE_MATCHSCORE_TOTALSCORE"),
                away_score=_first_value(group, "AWAYTEAMSCORE_MATCHSCORE_TOTALSCORE"),
                game_date=_first_value(group, "GAME_DATE"),
                venue=_first_value(group, "VENUE_NAME"),
            )
        )
    return players, matches


def _allocate_match(
    rng: np.random.Generator,
    indices: np.ndarray,
    utilities: np.ndarray,
    effects: np.ndarray,
    tau: float,
    totals: np.ndarray,
    n_sims: int,
    picks_out: list[np.ndarray] | None = None,
) -> None:
    """Draw one ordered 3-2-1 allocation per simulation and add to ``totals``.

    When ``picks_out`` is supplied, the local player positions chosen for each
    vote slot are appended to it.
    """
    scores = utilities[:, None] + effects[indices]
    weights = np.exp((scores - scores.max(axis=0, keepdims=True)) / tau)
    simulation_index = np.arange(n_sims)
    for points in (3, 2, 1):
        cumulative = np.cumsum(weights, axis=0)
        draws = rng.random(n_sims) * cumulative[-1]
        picks = (cumulative < draws).sum(axis=0)
        np.add.at(totals, (indices[picks], simulation_index), points)
        if picks_out is not None:
            picks_out.append(picks)
        weights[picks, simulation_index] = 0.0


def summarise_totals(
    players: pd.DataFrame,
    totals: np.ndarray,
    eligible: np.ndarray | None = None,
) -> pd.DataFrame:
    """Season summary columns shared by every simulator.

    ``totals`` is ``(n_players, n_sims)``. Award probabilities are computed
    among eligible players only when ``eligible`` is supplied.
    """
    summary = players.copy()
    summary["sim_mean"] = totals.mean(axis=1)
    summary["sim_median"] = np.median(totals, axis=1)
    for quantile in QUANTILES:
        summary[f"sim_q{int(quantile * 100):02d}"] = np.quantile(totals, quantile, axis=1)
    if eligible is None:
        eligible = np.ones(totals.shape[0], dtype=bool)
    award_totals = np.where(eligible[:, None], totals, -1)
    leaders = award_totals == award_totals.max(axis=0, keepdims=True)
    summary["p_outright_first"] = (leaders & (leaders.sum(axis=0, keepdims=True) == 1)).mean(axis=1)
    summary["p_first_or_joint"] = leaders.mean(axis=1)
    summary["ineligible"] = ~eligible
    return summary


def eligible_mask(players: pd.DataFrame, ineligible: set[str] | None) -> np.ndarray:
    """Boolean eligibility mask over a players frame."""
    if not ineligible:
        return np.ones(len(players), dtype=bool)
    player_ids = players[PLAYER_COLUMN].astype(str).to_numpy()
    return ~np.isin(player_ids, list(ineligible))


def _record_match(
    slot_counts: np.ndarray,
    triple_counts: dict[int, int],
    picks: list[np.ndarray],
    n_players: int,
) -> None:
    """Accumulate per-player slot counts and the top ordered triples."""
    for slot in range(3):
        np.add.at(slot_counts, (picks[slot], slot), 1)
    encoded = (picks[0].astype(np.int64) * n_players + picks[1]) * n_players + picks[2]
    values, counts = np.unique(encoded, return_counts=True)
    for index in np.argsort(-counts)[:TRIPLE_CANDIDATES]:
        triple_counts[int(values[index])] = int(counts[index])


def simulate_season(
    frame: pd.DataFrame,
    tau: float,
    effect_scale: float = 0.0,
    n_sims: int = 1000,
    seed: int = 42,
    track_rounds: bool = False,
    path_count: int = DEFAULT_PATH_COUNT,
    ineligible: set[str] | None = None,
    track_matches: bool = False,
) -> SeasonSimulation:
    """Simulate one season's count with a persistent player-season effect.

    ``ineligible`` lists player ids suspended during the season: they keep
    their simulated votes but award probabilities are computed among eligible
    players only, matching the medal's fairest-and-best rule.
    """
    players, matches = prepare_season(frame)
    n_players = len(players)
    eligible = eligible_mask(players, ineligible)
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
    round_p1 = None
    round_p2 = None
    round_p3 = None
    team_names = None
    team_cumulative_mean = None
    team_cumulative_quantiles = None
    team_increment_mean = None
    team_increment_quantiles = None
    team_path_totals = None
    match_groups = None
    match_slot_counts = None
    match_triple_counts = None
    player_count = None
    match_index_of: dict[str, int] = {}
    if track_matches:
        match_groups = matches
        match_slot_counts = [np.zeros((len(match.indices), 3), dtype=np.int64) for match in matches]
        match_triple_counts = [{} for _ in matches]
        player_count = n_players
        match_index_of = {match.match_id: index for index, match in enumerate(matches)}

    if track_rounds:
        rounds = sorted({match.round_number for match in matches})
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
        round_p1 = np.zeros((n_players, len(rounds)))
        round_p2 = np.zeros((n_players, len(rounds)))
        round_p3 = np.zeros((n_players, len(rounds)))

        team_names = sorted(players["TEAM_NAME"].dropna().unique().tolist())
        team_player_indices = [
            np.flatnonzero(players["TEAM_NAME"].to_numpy() == name) for name in team_names
        ]
        n_teams = len(team_names)
        team_cumulative_mean = np.zeros((n_teams, len(rounds)))
        team_cumulative_quantiles = np.zeros((n_teams, len(rounds), len(ROUND_QUANTILES)))
        team_increment_mean = np.zeros((n_teams, len(rounds)))
        team_increment_quantiles = np.zeros((n_teams, len(rounds), len(ROUND_QUANTILES)))
        team_previous = np.zeros((n_teams, n_sims))

        ordered = sorted(matches, key=lambda match: round_position[match.round_number])
        pointer = 0
        for round_index, round_number in enumerate(rounds):
            while pointer < len(ordered) and ordered[pointer].round_number == round_number:
                match = ordered[pointer]
                picks: list[np.ndarray] = []
                _allocate_match(
                    rng,
                    match.indices,
                    match.utilities,
                    effects,
                    tau,
                    totals,
                    n_sims,
                    picks_out=picks,
                )
                if track_matches:
                    index = match_index_of[match.match_id]
                    _record_match(
                        match_slot_counts[index], match_triple_counts[index], picks, n_players
                    )
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
            round_p1[:, round_index] = (increment == 1).mean(axis=1)
            round_p2[:, round_index] = (increment == 2).mean(axis=1)
            round_p3[:, round_index] = (increment == 3).mean(axis=1)
            previous[:] = totals

            team_totals = np.stack([totals[idx].sum(axis=0) for idx in team_player_indices])
            team_cumulative_mean[:, round_index] = team_totals.mean(axis=1)
            team_cumulative_quantiles[:, round_index, :] = np.quantile(
                team_totals, ROUND_QUANTILES, axis=1
            ).T
            team_increment = team_totals - team_previous
            team_increment_mean[:, round_index] = team_increment.mean(axis=1)
            team_increment_quantiles[:, round_index, :] = np.quantile(
                team_increment, ROUND_QUANTILES, axis=1
            ).T
            team_previous[:] = team_totals

        team_path_totals = np.stack(
            [path_totals[idx].sum(axis=0) for idx in team_player_indices]
        )
    else:
        totals = np.zeros((n_players, n_sims), dtype=np.int32)
        for index, match in enumerate(matches):
            picks = []
            _allocate_match(
                rng,
                match.indices,
                match.utilities,
                effects,
                tau,
                totals,
                n_sims,
                picks_out=picks,
            )
            if track_matches:
                _record_match(match_slot_counts[index], match_triple_counts[index], picks, n_players)

    summary = summarise_totals(players, totals, eligible)
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
        round_p1=round_p1,
        round_p2=round_p2,
        round_p3=round_p3,
        match_groups=match_groups,
        match_slot_counts=match_slot_counts,
        match_triple_counts=match_triple_counts,
        player_count=player_count,
        team_names=team_names,
        team_cumulative_mean=team_cumulative_mean,
        team_cumulative_quantiles=team_cumulative_quantiles,
        team_increment_mean=team_increment_mean,
        team_increment_quantiles=team_increment_quantiles,
        team_path_totals=team_path_totals,
    )


def integrated_match_log_loss(
    frame: pd.DataFrame,
    tau: float,
    effect_scale: float = 0.0,
    n_draws: int = 256,
    seed: int = 42,
) -> float:
    """Mean NLL of observed triples, integrating over persistent player effects.

    Each match probability is averaged over draws of the same persistent
    player-season effects the season simulator uses, so match-level and
    season-level calibration refer to the same predictive distribution.
    """
    players, matches = prepare_season(frame)
    evaluable = [match for match in matches if match.triple is not None]
    if not evaluable:
        return float("nan")
    rng = np.random.default_rng(seed)
    if effect_scale > 0.0:
        effects = rng.normal(0.0, effect_scale, size=(len(players), n_draws))
    else:
        effects = np.zeros((len(players), n_draws))
    losses = []
    for match in evaluable:
        scores = match.utilities[:, None] + effects[match.indices]
        log_prob = pl.match_log_likelihood_samples(scores, match.triple, tau)
        losses.append(float(np.log(n_draws) - logsumexp(log_prob)))
    return float(np.mean(losses))


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
    reports: dict[str, dict] | None = None,
    match_stats: dict[tuple[str, str], dict] | None = None,
) -> dict:
    """Build a compact JSON-ready payload for the interactive web visualisation."""
    if simulation.rounds is None or simulation.cumulative_quantiles is None:
        raise ValueError("run simulate_season(track_rounds=True) before exporting")
    if simulation.path_totals is None or simulation.round_p3 is None:
        raise ValueError("simulation is missing sampled paths or round probabilities")

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
                "ineligible": bool(row.get("ineligible", False)),
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
                    "p1": [_rounded(value, 4) for value in simulation.round_p1[position]],
                    "p2": [_rounded(value, 4) for value in simulation.round_p2[position]],
                    "p3": [_rounded(value, 4) for value in simulation.round_p3[position]],
                    "paths": [
                        [int(value) for value in paths[:, index]]
                        for index in range(paths.shape[1])
                    ],
                },
            }
        )

    n_sims = simulation.totals.shape[1]
    team_rounds: dict[str, dict] = {}
    if (
        simulation.team_names is not None
        and simulation.team_cumulative_mean is not None
        and simulation.team_cumulative_quantiles is not None
        and simulation.team_increment_mean is not None
        and simulation.team_increment_quantiles is not None
    ):
        for index, name in enumerate(simulation.team_names):
            cumulative = simulation.team_cumulative_quantiles[index]
            increment = simulation.team_increment_quantiles[index]
            paths: list[list[int]] = []
            if simulation.team_path_totals is not None:
                team_paths = simulation.team_path_totals[index]
                paths = [
                    [int(value) for value in team_paths[:, path]]
                    for path in range(min(team_paths.shape[1], 40))
                ]
            team_rounds[str(name)] = {
                "cumMean": [_rounded(value) for value in simulation.team_cumulative_mean[index]],
                "cumQ05": [_rounded(value) for value in cumulative[:, 0]],
                "cumQ25": [_rounded(value) for value in cumulative[:, 1]],
                "cumMedian": [_rounded(value) for value in cumulative[:, 2]],
                "cumQ75": [_rounded(value) for value in cumulative[:, 3]],
                "cumQ95": [_rounded(value) for value in cumulative[:, 4]],
                "incMean": [_rounded(value) for value in simulation.team_increment_mean[index]],
                "incQ05": [_rounded(value) for value in increment[:, 0]],
                "incQ25": [_rounded(value) for value in increment[:, 1]],
                "incMedian": [_rounded(value) for value in increment[:, 2]],
                "incQ75": [_rounded(value) for value in increment[:, 3]],
                "incQ95": [_rounded(value) for value in increment[:, 4]],
                "paths": paths,
            }

    team_payload = []
    for team, group in simulation.players.groupby("TEAM_NAME"):
        positions = group.index.to_numpy()
        team_totals = simulation.totals[positions].sum(axis=0)
        top_players = group.sort_values("sim_mean", ascending=False).head(4)
        team_payload.append(
            {
                "name": str(team),
                "expected": _rounded(team_totals.mean()),
                "q05": _rounded(np.quantile(team_totals, 0.05)),
                "q95": _rounded(np.quantile(team_totals, 0.95)),
                "nPlayers": len(group),
                "players": [
                    {
                        "id": str(row[PLAYER_COLUMN]),
                        "name": str(row["FULL_NAME"]).title(),
                        "expected": _rounded(row["sim_mean"]),
                    }
                    for _, row in top_players.iterrows()
                ],
                "rounds": team_rounds.get(str(team)),
            }
        )
    team_payload.sort(key=lambda row: row["expected"], reverse=True)

    match_payload = []
    if (
        simulation.match_groups is not None
        and simulation.match_slot_counts is not None
        and simulation.match_triple_counts is not None
        and simulation.player_count
    ):
        point_values = np.array([3.0, 2.0, 1.0])
        for index, match in enumerate(simulation.match_groups):
            counts = simulation.match_slot_counts[index]
            probabilities = counts / n_sims
            expected_points = probabilities @ point_values
            leaders = np.argsort(-expected_points)[:8]
            vote_rows = []
            for local in leaders:
                row = simulation.players.iloc[match.indices[local]]
                stats = (match_stats or {}).get(
                    (match.match_id, str(row[PLAYER_COLUMN])), {}
                )
                vote_rows.append(
                    {
                        "id": str(row[PLAYER_COLUMN]),
                        "name": str(row["FULL_NAME"]).title(),
                        "team": str(row["TEAM_NAME"]),
                        "p3": _rounded(probabilities[local, 0], 4),
                        "p2": _rounded(probabilities[local, 1], 4),
                        "p1": _rounded(probabilities[local, 2], 4),
                        "disposals": _rounded(stats.get("DISPOSALS"), 0),
                        "goals": _rounded(stats.get("GOALS"), 0),
                        "coachVotes": _rounded(stats.get("COACH_VOTES"), 0),
                        "ratingPoints": _rounded(stats.get("RATINGPOINTS"), 1),
                    }
                )
            triples = sorted(
                simulation.match_triple_counts[index].items(), key=lambda item: -item[1]
            )[:3]
            triple_rows = []
            for encoded, count in triples:
                third = encoded % simulation.player_count
                second = (encoded // simulation.player_count) % simulation.player_count
                first = encoded // (simulation.player_count**2)
                names = [
                    str(simulation.players.iloc[match.indices[local]]["FULL_NAME"]).title()
                    for local in (first, second, third)
                ]
                triple_rows.append({"players": names, "p": _rounded(count / n_sims, 4)})
            record = {
                "id": match.match_id,
                "round": int(match.round_number),
                "date": str(match.game_date) if match.game_date is not None else None,
                "home": match.home_team,
                "away": match.away_team,
                "venue": match.venue,
                "homeScore": int(match.home_score) if match.home_score is not None else None,
                "awayScore": int(match.away_score) if match.away_score is not None else None,
                "votes": vote_rows,
                "triples": triple_rows,
            }
            if reports and match.match_id in reports:
                record["report"] = reports[match.match_id]
            match_payload.append(record)

    return {
        "season": simulation.season,
        "meta": metadata or {},
        "rounds": [
            {"number": int(number), "label": "OR" if number == 0 else f"R{int(number)}"}
            for number in simulation.rounds
        ],
        "players": exported_players,
        "teams": team_payload,
        "matches": match_payload,
    }


def season_metrics(
    simulation: SeasonSimulation,
    contender_count: int = DEFAULT_CONTENDERS,
    ineligible: set[str] | None = None,
) -> dict:
    """CRPS, interval coverage and award outcomes for season vote totals.

    Contenders are the ``contender_count`` players with the highest simulated
    mean, a definition that is available at forecast time; no hindsight from
    the observed leaderboard enters the mask. ``ineligible`` removes suspended
    players from the favourite/winner determination but not from the votes.
    """
    totals = simulation.totals
    observed = simulation.players["observed_votes"].to_numpy(dtype=float)
    labelled = ~np.isnan(observed)
    mean_votes = totals.mean(axis=1)
    contender_mask = labelled & np.isin(
        np.arange(len(observed)),
        np.argsort(-mean_votes)[:contender_count],
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

    if labelled.any():
        n_players = len(observed)
        if ineligible:
            player_ids = simulation.players[PLAYER_COLUMN].astype(str).to_numpy()
            eligible = ~np.isin(player_ids, list(ineligible))
        else:
            eligible = np.ones(n_players, dtype=bool)
        eligible_mean = np.where(eligible, mean_votes, -np.inf)
        favorite = int(np.argmax(eligible_mean))
        observed_eligible = labelled & eligible
        observed_max = float(np.nanmax(np.where(observed_eligible, observed, np.nan)))
        joint_winners = set(
            np.flatnonzero(observed_eligible & (observed == observed_max)).tolist()
        )
        winner = int(np.nanargmax(np.where(observed_eligible, observed, np.nan)))
        result["favorite_won"] = bool(favorite in joint_winners)
        result["favorite_probability"] = float(simulation.players["p_first_or_joint"].iloc[favorite])
        result["winner_probability"] = float(simulation.players["p_first_or_joint"].iloc[winner])
        result["winner_mean_rank"] = int(1 + (mean_votes > mean_votes[winner]).sum())
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
