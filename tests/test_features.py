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
