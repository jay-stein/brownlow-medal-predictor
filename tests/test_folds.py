import pandas as pd

from brownlow.folds import (
    DEV_FOLDS,
    FINAL_FOLD,
    season_cv_indices,
    split_by_season,
    time_ordered_cv_indices,
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


def test_season_cv_is_forward_chaining():
    seasons = pd.Series([2020] * 4 + [2021] * 4 + [2022] * 4)
    folds = season_cv_indices(seasons, n_splits=2)
    assert [sorted(set(seasons.iloc[valid])) for _, valid in folds] == [[2021], [2022]]
    for train_idx, valid_idx in folds:
        assert seasons.iloc[train_idx].max() < seasons.iloc[valid_idx].min()
        assert len(valid_idx) > 0


def test_season_cv_never_validates_on_unsupported_early_seasons():
    seasons = pd.Series([2020] * 2 + [2021] * 2)
    folds = season_cv_indices(seasons, n_splits=5)
    assert [sorted(set(seasons.iloc[valid])) for _, valid in folds] == [[2021]]


def test_time_ordered_cv_falls_back_to_rounds_for_one_season():
    seasons = pd.Series([2020] * 6)
    rounds = pd.Series([1, 1, 2, 2, 3, 3])
    folds = time_ordered_cv_indices(seasons, rounds, n_splits=2)
    assert [sorted(set(rounds.iloc[valid])) for _, valid in folds] == [[2], [3]]
    for train_idx, valid_idx in folds:
        assert rounds.iloc[train_idx].max() < rounds.iloc[valid_idx].min()
