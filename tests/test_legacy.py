import numpy as np
import pandas as pd

from brownlow import features, legacy

PLAYER_COLUMN = "PLAYER_PLAYER_PLAYER_PLAYERID"


def test_legacy_feature_columns_drop_cba_and_height():
    columns = features.FEATURE_LIST + legacy.LEGACY_DROP
    frame = pd.DataFrame({column: [0.0] for column in columns})
    selected = legacy.legacy_feature_columns(frame)
    assert selected.shape[1] == len(features.FEATURE_LIST) - len(legacy.LEGACY_DROP)
    for dropped in legacy.LEGACY_DROP:
        assert dropped not in selected.columns


def test_simulate_legacy_season_sums_independent_draws():
    frame = pd.DataFrame(
        {
            "ROUND_YEAR": [2024] * 8,
            "PROVIDERID": ["m1"] * 4 + ["m2"] * 4,
            "ROUND_ROUNDNUMBER": [0] * 8,
            PLAYER_COLUMN: ["p1", "p2", "p3", "p4"] * 2,
            "FULL_NAME": ["A", "B", "C", "D"] * 2,
            "TEAM_NAME": ["T"] * 8,
            "pred_mean": [2.0, 1.0, 0.5, 0.1] * 2,
            "pred_std": [0.4, 0.4, 0.4, 0.4] * 2,
            "BROWNLOW_VOTES_AUDITED": [3.0, 2.0, 1.0, 0.0] * 2,
        }
    )
    simulation = legacy.simulate_legacy_season(frame, n_sims=400, seed=1)
    assert simulation.totals.shape == (4, 400)
    # No match budget: season totals are not constrained to six per match.
    assert simulation.players["sim_mean"].sum() > 0
    assert simulation.players["p_first_or_joint"].iloc[0] > 0.5


def test_legacy_simulation_respects_eligibility():
    frame = pd.DataFrame(
        {
            "ROUND_YEAR": [2024] * 4,
            "PROVIDERID": ["m1"] * 4,
            "ROUND_ROUNDNUMBER": [0] * 4,
            PLAYER_COLUMN: ["p1", "p2", "p3", "p4"],
            "FULL_NAME": ["A", "B", "C", "D"],
            "TEAM_NAME": ["T"] * 4,
            "pred_mean": [2.5, 1.2, 0.3, 0.2],
            "pred_std": [0.2, 0.2, 0.2, 0.2],
            "BROWNLOW_VOTES_AUDITED": [3.0, 2.0, 1.0, 0.0],
        }
    )
    simulation = legacy.simulate_legacy_season(frame, n_sims=300, seed=2, ineligible={"p1"})
    players = simulation.players.set_index(PLAYER_COLUMN)
    assert players.loc["p1", "p_first_or_joint"] == 0.0
    assert players.loc["p1", "ineligible"]
    assert players.loc["p2", "p_first_or_joint"] > 0.6
    assert np.isfinite(simulation.totals).all()
