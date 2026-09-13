import numpy as np
import pandas as pd
import pytest

from brownlow.features import (
    FEATURE_LIST,
    NATIVE_MISSING_FEATURES,
    PROPORTION_FEATURES,
    RAW_STAT_FEATURES,
    FeaturePreprocessor,
    add_height_features,
    add_match_context,
    add_season_aggregates,
    compute_team_elo,
)

PLAYER_COLUMN = "PLAYER_PLAYER_PLAYER_PLAYERID"


def _team_stats() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "matchId": ["m1", "m2"],
            "match.date": ["2024-03-01", "2024-03-08"],
            "SEASON": [2024, 2024],
            "match.homeTeam.name": ["Fremantle", "Sydney Swans"],
            "match.awayTeam.name": ["Sydney Swans", "Fremantle"],
            "homeTeamScore.matchScore.totalScore": [100, 90],
            "awayTeamScore.matchScore.totalScore": [80, 70],
        }
    )


def test_feature_list_shape():
    assert len(RAW_STAT_FEATURES) == 58
    assert len(PROPORTION_FEATURES) == 56
    assert len(FEATURE_LIST) == 124
    assert len(set(FEATURE_LIST)) == len(FEATURE_LIST)
    assert "EXTENDEDSTATS_CENTREBOUNCEATTENDANCES" in FEATURE_LIST
    assert "PLAYER_HEIGHT" in FEATURE_LIST
    assert "HEIGHT_VS_POSITION" in FEATURE_LIST


def test_add_height_features_relative_to_position():
    frame = pd.DataFrame(
        {
            "PROVIDERID": ["m1", "m1", "m1"],
            "GAME_DATE": ["2024-03-01", "2024-03-01", "2024-03-01"],
            "PLAYER_PLAYER_PLAYER_PLAYERID": ["p1", "p2", "p3"],
        }
    )
    details = pd.DataFrame(
        {
            "providerId": ["p1", "p2", "p3"],
            "season": [2024, 2024, 2024],
            "position": ["MIDFIELDER", "MIDFIELDER", "MIDFIELDER"],
            "heightInCm": [190, 180, 185],
        }
    )
    enriched = add_height_features(frame, details)
    assert enriched["PLAYER_HEIGHT"].tolist() == [190, 180, 185]
    # peers of p1 are 180 and 185 -> mean 182.5
    assert enriched["HEIGHT_VS_POSITION"].iloc[0] == pytest.approx(7.5)
    assert enriched["HEIGHT_VS_POSITION"].iloc[1] == pytest.approx(180 - 187.5)


def test_feature_preprocessor_fills_with_train_means():
    train = pd.DataFrame(
        {
            "GOALS": [1.0, 3.0, 2.0],
            "DISPOSALS": [10.0, 20.0, 30.0],
            "PLAYER_PLAYER_POSITION": pd.Categorical(["MID", "FWD", "MID"]),
            "PLAYERTEAM_OUTCOME": pd.Categorical(["WIN", "LOSS", "WIN"]),
        }
    )
    test = pd.DataFrame(
        {
            "GOALS": [np.nan],
            "DISPOSALS": [40.0],
            "PLAYER_PLAYER_POSITION": pd.Categorical(["MID"]),
            "PLAYERTEAM_OUTCOME": pd.Categorical(["WIN"]),
        }
    )
    preprocessor = FeaturePreprocessor().fit(train)
    transformed = preprocessor.transform(test)
    assert transformed["GOALS"].iloc[0] == 2.0
    assert transformed["DISPOSALS"].iloc[0] == 40.0


def test_feature_preprocessor_leaves_native_missing_features():
    assert "EXTENDEDSTATS_CENTREBOUNCEATTENDANCES" in NATIVE_MISSING_FEATURES
    train = pd.DataFrame(
        {
            "GOALS": [1.0, 3.0],
            "EXTENDEDSTATS_CENTREBOUNCEATTENDANCES": [np.nan, 2.0],
            "PLAYER_HEIGHT": [188.0, np.nan],
            "PLAYER_PLAYER_POSITION": pd.Categorical(["MID", "MID"]),
            "PLAYERTEAM_OUTCOME": pd.Categorical(["WIN", "WIN"]),
        }
    )
    test = pd.DataFrame(
        {
            "GOALS": [np.nan],
            "EXTENDEDSTATS_CENTREBOUNCEATTENDANCES": [np.nan],
            "PLAYER_HEIGHT": [np.nan],
            "PLAYER_PLAYER_POSITION": pd.Categorical(["MID"]),
            "PLAYERTEAM_OUTCOME": pd.Categorical(["WIN"]),
        }
    )
    transformed = FeaturePreprocessor().fit(train).transform(test)
    assert transformed["GOALS"].iloc[0] == 2.0
    assert np.isnan(transformed["EXTENDEDSTATS_CENTREBOUNCEATTENDANCES"].iloc[0])
    assert np.isnan(transformed["PLAYER_HEIGHT"].iloc[0])


def test_feature_preprocessor_encodes_unseen_categories_as_missing():
    train = pd.DataFrame(
        {
            "GOALS": [1.0, 3.0],
            "DISPOSALS": [10.0, 20.0],
            "PLAYER_PLAYER_POSITION": pd.Categorical(["MID", "FWD"]),
            "PLAYERTEAM_OUTCOME": pd.Categorical(["WIN", "LOSS"]),
        }
    )
    test = pd.DataFrame(
        {
            "GOALS": [2.0],
            "DISPOSALS": [15.0],
            "PLAYER_PLAYER_POSITION": pd.Categorical(["RUCK"]),
            "PLAYERTEAM_OUTCOME": pd.Categorical(["DRAW"]),
        }
    )
    transformed = FeaturePreprocessor().fit(train).transform(test)
    assert transformed["PLAYER_PLAYER_POSITION"].isna().all()
    assert transformed["PLAYERTEAM_OUTCOME"].isna().all()


def test_compute_team_elo_records_pre_match_ratings():
    elo = compute_team_elo(_team_stats()).set_index("MATCHID")
    assert elo.loc["m1", "HOME_ELO"] == pytest.approx(1500.0)
    assert elo.loc["m1", "AWAY_ELO"] == pytest.approx(1500.0)
    # Fremantle won the first match, so it is above 1500 for the second.
    assert elo.loc["m2", "AWAY_ELO"] > 1500.0
    assert elo.loc["m2", "HOME_ELO"] < 1500.0


def test_add_match_context_flags_close_games_and_travel():
    frame = pd.DataFrame(
        {
            "MATCHID": ["m1"],
            "TEAM_NAME": ["Fremantle"],
            "AT_HOME": [1],
            "PLAYERTEAM_MARGIN": [6],
            "VENUE_STATE": ["VIC"],
        }
    )
    out = add_match_context(frame, _team_stats())
    assert out["ABS_MARGIN"].iloc[0] == 6
    assert out["CLOSE_FINISH"].iloc[0] == 1
    assert out["INTERSTATE"].iloc[0] == 1.0
    assert out["OPPONENT_ELO"].iloc[0] == pytest.approx(1500.0)


def test_add_season_aggregates_leave_one_game_out():
    frame = pd.DataFrame(
        {
            PLAYER_COLUMN: ["p1", "p1", "p2", "p2", "p3"],
            "ROUND_YEAR": [2024] * 5,
            "PROVIDERID": ["m1", "m2", "m1", "m2", "m1"],
            "DISPOSALS": [10.0, 20.0, 5.0, 15.0, 8.0],
            "COACH_VOTES": [0.0, 8.0, 0.0, 0.0, 0.0],
            "PLAYERTEAM_OUTCOME": ["WIN", "LOSS", "WIN", "LOSS", "WIN"],
        }
    )
    out = add_season_aggregates(frame)
    p1 = out[out[PLAYER_COLUMN] == "p1"].sort_values("PROVIDERID")
    assert p1["SEASON_DISPOSALS_PG"].tolist() == pytest.approx([20.0, 10.0])
    assert p1["SEASON_COACH_VOTES_TOTAL"].tolist() == pytest.approx([8.0, 0.0])
    assert p1["SEASON_COACH_VOTES_RANK"].tolist() == pytest.approx([1.0, 2.0])
    p3 = out[out[PLAYER_COLUMN] == "p3"].iloc[0]
    assert np.isnan(p3["SEASON_DISPOSALS_PG"])
    assert p3["SEASON_GAMES"] == 1
