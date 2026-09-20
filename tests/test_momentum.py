import pandas as pd
import pytest

from brownlow import momentum

PROVIDER = "CD_M1"


def _events() -> pd.DataFrame:
    rows = [
        # period, seconds, type, value, home_away, agg_home, agg_away, player
        (1, 81, "GOAL", 6, "AWAY", 0, 6, "B"),
        (1, 200, "GOAL", 6, "HOME", 6, 6, "A"),
        (1, 300, "BEHIND", 1, "HOME", 7, 6, "A"),
        (2, 400, "GOAL", 6, "AWAY", 7, 12, "C"),
        (2, 500, "GOAL", 6, "HOME", 13, 12, "D"),
        (3, 1100, "GOAL", 6, "AWAY", 13, 18, "B"),
        (3, 1300, "GOAL", 6, "AWAY", 13, 24, "B"),
        (3, 1500, "GOAL", 6, "HOME", 19, 24, "A"),
        (4, 100, "GOAL", 6, "HOME", 25, 24, "A"),
        (4, 1300, "GOAL", 6, "HOME", 31, 24, "A"),
        (4, 1500, "GOAL", 6, "AWAY", 31, 30, "C"),
    ]
    return pd.DataFrame(
        [
            {
                "SEASON": 2026,
                "PROVIDERID": PROVIDER,
                "PERIOD": period,
                "PERIOD_SECONDS": seconds,
                "SCORE_TYPE": kind,
                "SCORE_VALUE": value,
                "HOME_AWAY": side,
                "TEAM_ID": "CD_T1" if side == "HOME" else "CD_T2",
                "TEAM_NAME": "Home" if side == "HOME" else "Away",
                "PLAYER_ID": player,
                "PLAYER_NAME": player,
                "AGG_HOME": home,
                "AGG_AWAY": away,
            }
            for period, seconds, kind, value, side, home, away, player in rows
        ]
    )


def test_match_features_capture_swings_and_comes_back():
    matches = momentum.match_features(_events()).iloc[0]
    assert matches["M_Q4_SWING"] == 6  # -5 at three-quarter time to +1
    assert matches["M_LATE_SWING"] == 6
    assert matches["M_COMEBACK_HOME"] == 1
    assert matches["M_BLOWN_HOME"] == 0
    assert matches["M_RUN_Q4_HOME"] == 2
    assert matches["M_RUN_Q4_AWAY"] == 1
    assert matches["M_LEAD_Q4_HOME"] == 7
    assert matches["M_CLOSE_Q4"] == 1
    assert matches["M_LAST10_CLOSE"] == 1
    assert matches["M_Q4_SCORE_HOME"] == 12
    assert matches["M_Q4_SCORE_AWAY"] == 6


def test_player_features_score_when_it_mattered():
    players = momentum.player_features(_events()).set_index("PLAYER_ID")
    a = players.loc["A"]
    assert a["Q4_GOALS"] == 2
    assert a["Q4_SCORES"] == 2
    assert a["LATE_GOALS"] == 3  # Q3 1500s plus both Q4 goals
    assert a["CLUTCH_SCORES"] == 3  # late-window margins -5, +1 and +7 (all within 12)
    assert a["TEAM_Q4_SHARE"] == pytest.approx(1.0)
    assert a["FIRST_GOAL"] == 0
    b = players.loc["B"]
    assert b["FIRST_GOAL"] == 1
    assert b["CLUTCH_SCORES"] == 1  # only the 1300s Q3 goal is in the late window
    assert players.loc["C", "LAST_GOAL"] == 1


def test_attach_momentum_flips_perspective_and_zero_fills():
    table = pd.DataFrame(
        {
            "PROVIDERID": [PROVIDER, PROVIDER, PROVIDER, "CD_OTHER"],
            momentum.PLAYER_COLUMN: ["A", "B", "Z", "Q"],
            "AT_HOME": [1, 0, 1, 1],
        }
    )
    out = momentum.attach_momentum(table, _events())
    home = out.iloc[0]
    away = out.iloc[1]
    non_scorer = out.iloc[2]
    missing = out.iloc[3]

    assert home["TEAM_Q4_SWING"] == 6
    assert away["TEAM_Q4_SWING"] == -6
    assert home["TEAM_COMEBACK"] == 1
    assert away["TEAM_COMEBACK"] == 0
    assert home["TEAM_RUN_Q4"] == 2
    assert away["TEAM_RUN_Q4"] == 1
    assert home["TEAM_LEAD_Q4"] == 7
    assert away["TEAM_LEAD_Q4"] == -1

    assert non_scorer["Q4_GOALS"] == 0
    assert non_scorer["LATE_SCORES"] == 0
    assert non_scorer["CLOSE_Q4"] == 1

    for column in momentum.MOMENTUM_FEATURES:
        assert pd.isna(missing[column]), column


def test_attach_momentum_without_events_is_all_nan():
    table = pd.DataFrame(
        {
            "PROVIDERID": [PROVIDER],
            momentum.PLAYER_COLUMN: ["A"],
            "AT_HOME": [1],
        }
    )
    out = momentum.attach_momentum(table, pd.DataFrame(columns=["PROVIDERID"]))
    assert out[momentum.MOMENTUM_FEATURES].isna().all().all()
