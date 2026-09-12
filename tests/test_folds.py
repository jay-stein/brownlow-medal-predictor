import pandas as pd

from brownlow.folds import (
    DEV_FOLDS,
    FINAL_FOLD,
    match_grouped_cv_indices,
    split_by_season,
)


def test_dev_folds_are_chronological():
    for fold in DEV_FOLDS:
        assert max(fold.train_seasons) < fold.eval_season
    assert max(FINAL_FOLD.train_seasons) < FINAL_FOLD.eval_season


def test_split_by_season_keeps_whole_matches():
    df = pd.DataFrame(
        {
            "ROUND_YEAR": [2020, 2020, 2021, 2021],
            "PROVIDERID": ["m1", "m1", "m2", "m3"],
            "value": [1, 2, 3, 4],
        }
    )
    train, evaluation = split_by_season(df, (2020,), 2021)
    assert set(train["PROVIDERID"]) == {"m1"}
    assert set(evaluation["PROVIDERID"]) == {"m2", "m3"}
    assert len(train) + len(evaluation) == len(df)


def test_match_grouped_cv_never_splits_a_match():
    match_ids = pd.Series([f"m{index // 4}" for index in range(40)])
    folds = match_grouped_cv_indices(match_ids, n_splits=5)
    assert len(folds) == 5
    for train_idx, valid_idx in folds:
        assert not set(match_ids.iloc[train_idx]) & set(match_ids.iloc[valid_idx])
        assert len(valid_idx) > 0
