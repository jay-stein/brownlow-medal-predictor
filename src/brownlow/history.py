"""Partially pooled historical player effects.

The statistical score can systematically under- or over-rate particular
players (for example, players whose polling style the feature set does not
capture). This module turns prior seasons' out-of-sample residuals into a
shrunken per-player adjustment:

1. For every prior season's out-of-sample scores, compute each player's
   expected votes from the Plackett-Luce marginals and compare with the
   observed votes, giving a residual per game.
2. Average those residuals with optional recency weighting, shrink strongly
   toward zero when the player has little evidence, and map the vote-unit
   effect into utility units through a single scalar.

The adjustment is *conditional on the statistical score*: it is built from
residuals, never from raw vote totals, so past performance is not counted
twice. Whether it helps is decided by rolling out-of-sample evaluation; the
baseline mapping of zero is always part of the candidate grid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import pl, simulate

PLAYER_COLUMN = simulate.PLAYER_COLUMN
VOTE_POINTS = np.array([0.0, 1.0, 2.0, 3.0])


def player_residual_history(
    scores_by_season: dict[int, pd.DataFrame],
    tau: float,
) -> pd.DataFrame:
    """Per player-season expected, observed and residual votes.

    Expected votes come from the Plackett-Luce marginals of each match's
    out-of-sample utilities, so the residual measures what the score model
    missed rather than what the player scored in raw terms.
    """
    records: list[dict] = []
    for season in sorted(scores_by_season):
        frame = scores_by_season[season]
        players, matches = simulate.prepare_season(frame)
        n_players = len(players)
        expected = np.zeros(n_players)
        observed = np.zeros(n_players)
        games = np.zeros(n_players, dtype=int)
        for match in matches:
            if match.triple is None:
                continue
            marginals = pl.pl_marginals(match.utilities, tau)
            expected_here = marginals @ VOTE_POINTS
            np.add.at(expected, match.indices, expected_here)
            for slot, points in ((0, 3.0), (1, 2.0), (2, 1.0)):
                observed[match.indices[match.triple[slot]]] += points
            np.add.at(games, match.indices, 1)
        for index, player_id in enumerate(players[PLAYER_COLUMN]):
            if games[index] == 0:
                continue
            records.append(
                {
                    "player_id": str(player_id),
                    "season": int(season),
                    "games": int(games[index]),
                    "expected_votes": float(expected[index]),
                    "observed_votes": float(observed[index]),
                    "residual": float(observed[index] - expected[index]),
                    "residual_per_game": float(
                        (observed[index] - expected[index]) / games[index]
                    ),
                }
            )
    return pd.DataFrame(records)


def attach_player_effects(
    frame: pd.DataFrame,
    history: pd.DataFrame,
    target_season: int,
    *,
    shrinkage: float = 100.0,
    mapping: float = 1.0,
    half_life: float = 0.0,
) -> pd.DataFrame:
    """Add shrunken historical player effects to a season's utilities.

    ``shrinkage`` is in games: ``alpha = residual_per_game * n_eff /
    (n_eff + shrinkage)``. ``half_life`` is in seasons (0 disables decay).
    ``mapping`` converts the vote-unit effect to utility units; ``mapping=0``
    reproduces the unadjusted baseline exactly.
    """
    work = frame.copy()
    work["PLAYER_EFFECT"] = 0.0
    prior = history[history["season"] < target_season]
    if prior.empty or mapping == 0.0:
        return work

    if half_life and half_life > 0.0:
        age = target_season - 1 - prior["season"]
        weight = 0.5 ** (age / half_life)
    else:
        weight = pd.Series(1.0, index=prior.index)
    prior = prior.assign(
        weighted_games=weight * prior["games"],
        weighted_residual=weight * prior["residual"],
    )
    grouped = prior.groupby("player_id", as_index=False).agg(
        weighted_games=("weighted_games", "sum"),
        weighted_residual=("weighted_residual", "sum"),
    )
    evidence = grouped["weighted_games"].to_numpy(dtype=float)
    residual_per_game = grouped["weighted_residual"].to_numpy(dtype=float) / np.maximum(
        evidence, 1e-9
    )
    grouped["alpha"] = residual_per_game * evidence / (evidence + shrinkage)
    effect_by_player = dict(zip(grouped["player_id"], grouped["alpha"] * mapping))

    work["PLAYER_EFFECT"] = work[PLAYER_COLUMN].astype(str).map(effect_by_player).fillna(0.0)
    work[simulate.UTILITY_COLUMN] = work[simulate.UTILITY_COLUMN] + work["PLAYER_EFFECT"]
    return work
