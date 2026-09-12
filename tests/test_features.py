import numpy as np
import pandas as pd

from brownlow.features import (
    FEATURE_LIST,
    PROPORTION_FEATURES,
    RAW_STAT_FEATURES,
    FeaturePreprocessor,
)


def test_feature_list_shape():
    assert len(RAW_STAT_FEATURES) == 57
    assert len(PROPORTION_FEATURES) == 55
    assert len(FEATURE_LIST) == 120
    assert len(set(FEATURE_LIST)) == len(FEATURE_LIST)


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
