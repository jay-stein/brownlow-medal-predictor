"""Chronological, match-grouped evaluation folds.

The schedule follows the redesign plan: development folds train on earlier
seasons and evaluate on the next; the final fold is used once, untouched; the
production fit trains on every labelled season ahead of the forecast season.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    name: str
    train_seasons: tuple[int, ...]
    eval_season: int


FIRST_LABEL_SEASON = 2012

# Development folds: evaluate on the next season after all earlier labels.
DEV_FOLDS: tuple[Fold, ...] = tuple(
    Fold(f"dev_{year}", tuple(range(FIRST_LABEL_SEASON, year)), year)
    for year in range(2015, 2025)
)

FINAL_FOLD = Fold("final_2025", tuple(range(FIRST_LABEL_SEASON, 2025)), 2025)

# The production fit includes the final evaluation season once it has been
# used, so future forecasts train on every available label season.
PRODUCTION_SEASONS: tuple[int, ...] = tuple(range(FIRST_LABEL_SEASON, 2026))
FORECAST_SEASON = 2026


def season_mask(df: pd.DataFrame, seasons: tuple[int, ...], year_col: str = "ROUND_YEAR") -> pd.Series:
    """Rows whose season is in ``seasons``."""
    return df[year_col].isin(list(seasons))


def split_by_season(
    df: pd.DataFrame,
    train_seasons: tuple[int, ...],
    eval_season: int,
    year_col: str = "ROUND_YEAR",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split: whole seasons only, never a random row split."""
    train = df[season_mask(df, train_seasons, year_col)].copy()
    evaluation = df[df[year_col] == eval_season].copy()
    return train, evaluation


def season_cv_indices(
    season_ids: pd.Series | np.ndarray,
    n_splits: int = 3,
    min_train_seasons: int = 1,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Forward-chaining folds over whole seasons.

    The last ``n_splits`` seasons are each validated once, trained only on the
    seasons before them. Every validation fold is therefore a genuine forecast
    into a later era, unlike a random match-grouped split.
    """
    seasons = np.asarray(season_ids)
    unique = np.sort(np.unique(seasons))
    start = max(min_train_seasons, len(unique) - n_splits)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for index in range(start, len(unique)):
        valid_season = unique[index]
        folds.append(
            (np.flatnonzero(seasons < valid_season), np.flatnonzero(seasons == valid_season))
        )
    if not folds:
        raise ValueError("not enough distinct seasons for a time-ordered split")
    return folds


def time_ordered_cv_indices(
    season_ids: pd.Series | np.ndarray,
    round_ids: pd.Series | np.ndarray,
    n_splits: int = 3,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Time-ordered folds, falling back to rounds when only one season exists."""
    seasons = np.asarray(season_ids)
    if len(np.unique(seasons)) >= 2:
        return season_cv_indices(seasons, n_splits=n_splits)
    rounds = np.asarray(round_ids)
    unique = np.sort(np.unique(rounds))
    start = max(1, len(unique) - n_splits)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for index in range(start, len(unique)):
        valid_round = unique[index]
        folds.append(
            (np.flatnonzero(rounds < valid_round), np.flatnonzero(rounds == valid_round))
        )
    if not folds:
        raise ValueError("not enough distinct rounds for a time-ordered split")
    return folds
