import numpy as np
import pandas as pd
import pytest

from brownlow import history

PLAYERS = ["p1", "p2", "p3", "p4"]
PLAYER_COLUMN = "PLAYER_PLAYER_PLAYER_PLAYERID"


def _frame(year: int, match_specs: list[tuple[list[float], list[float]]]) -> pd.DataFrame:
    rows = []
    for match_index, (utilities, votes) in enumerate(match_specs):
        for player, utility, vote in zip(PLAYERS, utilities, votes):
            rows.append(
                {
                    "ROUND_YEAR": year,
                    "PROVIDERID": f"m{year}-{match_index}",
                    "ROUND_ROUNDNUMBER": match_index,
                    PLAYER_COLUMN: player,
                    "FULL_NAME": player.upper(),
                    "TEAM_NAME": "T",
                    "UTILITY": float(utility),
                    "BROWNLOW_VOTES_AUDITED": float(vote),
                }
            )
    return pd.DataFrame(rows)


def _history_row(player: str, season: int, residual: float, games: int = 10) -> dict:
    return {
        "player_id": player,
        "season": season,
        "games": games,
        "expected_votes": 0.0,
        "observed_votes": residual,
        "residual": residual,
        "residual_per_game": residual / games,
    }


def test_player_residual_history_uses_expected_not_observed():
    frame = _frame(
        2020,
        [
            ([3.0, 2.0, 1.0, 0.0], [3.0, 2.0, 1.0, 0.0]),
            ([0.0, 1.0, 2.0, 3.0], [3.0, 2.0, 1.0, 0.0]),
        ],
    )
    table = history.player_residual_history({2020: frame}, tau=1.0)
    row = table[table["player_id"] == "p1"].iloc[0]
    assert row["games"] == 2
    assert row["residual"] > 0.0
    other = table[table["player_id"] == "p4"].iloc[0]
    assert other["residual"] < 0.0


def test_attach_player_effects_zero_mapping_is_exact_baseline():
    frame = _frame(2021, [([3.0, 2.0, 1.0, 0.0], [3.0, 2.0, 1.0, 0.0])])
    table = pd.DataFrame([_history_row("p1", 2020, 5.0)])
    adjusted = history.attach_player_effects(frame, table, 2021, shrinkage=0.0, mapping=0.0)
    np.testing.assert_allclose(adjusted["UTILITY"], frame["UTILITY"])
    assert (adjusted["PLAYER_EFFECT"] == 0.0).all()


def test_attach_player_effects_maps_and_shrinks():
    frame = _frame(2021, [([3.0, 2.0, 1.0, 0.0], [3.0, 2.0, 1.0, 0.0])])
    table = pd.DataFrame([_history_row("p1", 2020, 2.0, games=10)])
    full = history.attach_player_effects(frame, table, 2021, shrinkage=0.0, mapping=2.0)
    shrunk = history.attach_player_effects(frame, table, 2021, shrinkage=90.0, mapping=2.0)
    full_effect = full.loc[full[PLAYER_COLUMN] == "p1", "PLAYER_EFFECT"].iloc[0]
    shrunk_effect = shrunk.loc[shrunk[PLAYER_COLUMN] == "p1", "PLAYER_EFFECT"].iloc[0]
    assert full_effect == pytest.approx(0.4)
    assert 0.0 < shrunk_effect < full_effect
    assert shrunk.loc[shrunk[PLAYER_COLUMN] == "p2", "PLAYER_EFFECT"].iloc[0] == 0.0


def test_attach_player_effects_recency_weighting():
    frame = _frame(2021, [([3.0, 2.0, 1.0, 0.0], [3.0, 2.0, 1.0, 0.0])])
    table = pd.DataFrame(
        [_history_row("p1", 2019, -5.0), _history_row("p1", 2020, 5.0)]
    )
    flat = history.attach_player_effects(frame, table, 2021, shrinkage=0.0, half_life=0.0)
    recent = history.attach_player_effects(frame, table, 2021, shrinkage=0.0, half_life=1.0)
    flat_effect = flat.loc[flat[PLAYER_COLUMN] == "p1", "PLAYER_EFFECT"].iloc[0]
    recent_effect = recent.loc[recent[PLAYER_COLUMN] == "p1", "PLAYER_EFFECT"].iloc[0]
    assert flat_effect == pytest.approx(0.0)
    assert recent_effect > 0.0


def test_attach_player_effects_ignores_target_and_future_seasons():
    frame = _frame(2021, [([3.0, 2.0, 1.0, 0.0], [3.0, 2.0, 1.0, 0.0])])
    table = pd.DataFrame(
        [_history_row("p1", 2021, 9.0), _history_row("p1", 2022, 9.0)]
    )
    adjusted = history.attach_player_effects(frame, table, 2021, shrinkage=0.0, mapping=1.0)
    assert (adjusted["PLAYER_EFFECT"] == 0.0).all()
