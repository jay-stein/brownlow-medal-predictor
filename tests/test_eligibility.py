import numpy as np
import pandas as pd
import pytest

from brownlow import eligibility, simulate

PLAYER_COLUMN = "PLAYER_PLAYER_PLAYER_PLAYERID"


def _season_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ROUND_YEAR": [2024] * 4,
            "PROVIDERID": ["m1"] * 4,
            "ROUND_ROUNDNUMBER": [0] * 4,
            PLAYER_COLUMN: ["p1", "p2", "p3", "p4"],
            "FULL_NAME": ["A", "B", "C", "D"],
            "TEAM_NAME": ["T"] * 4,
            "UTILITY": [3.0, 2.0, 1.0, 0.0],
            "BROWNLOW_VOTES_AUDITED": [3.0, 2.0, 1.0, 0.0],
        }
    )


def test_load_ineligible_reads_seasonal_file(tmp_path):
    pd.DataFrame(
        {
            "season": [2024, 2024],
            "player_id": ["p1", "p2"],
            "player_name": ["A", "B"],
        }
    ).to_csv(tmp_path / "suspensions_2024.csv", index=False)
    assert eligibility.ineligible_ids(2024, directory=tmp_path) == {"p1", "p2"}
    assert eligibility.ineligible_ids(2023, directory=tmp_path) == set()


def test_ineligible_player_cannot_be_awarded_the_medal():
    simulation = simulate.simulate_season(
        _season_frame(), tau=1.0, n_sims=500, seed=4, ineligible={"p1"}
    )
    players = simulation.players.set_index(PLAYER_COLUMN)
    assert players.loc["p1", "p_first_or_joint"] == 0.0
    assert players.loc["p2", "p_first_or_joint"] > 0.9
    assert bool(players.loc["p1", "ineligible"]) is True
    assert bool(players.loc["p2", "ineligible"]) is False


def test_eligibility_leaves_vote_totals_unchanged():
    without = simulate.simulate_season(_season_frame(), tau=1.0, n_sims=300, seed=7)
    with_rules = simulate.simulate_season(
        _season_frame(), tau=1.0, n_sims=300, seed=7, ineligible={"p1"}
    )
    np.testing.assert_array_equal(without.totals, with_rules.totals)


def test_historical_curated_file_covers_known_cases():
    table = eligibility.load_ineligible(2014)
    names = set(table["player_name"])
    assert {"NAT FYFE", "STEVE JOHNSON"} <= names
    assert "PATRICK DANGERFIELD" in set(eligibility.load_ineligible(2017)["player_name"])


def test_2026_suspension_file_is_populated():
    table = eligibility.load_ineligible(2026)
    assert len(table) >= 20
    assert "HARLEY REID" in set(table["player_name"])
