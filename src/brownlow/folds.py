"""Chronological, match-grouped evaluation folds.

The schedule follows the redesign plan: development folds train on earlier
seasons and evaluate on the next; the final fold is used once, untouched; the
production fit trains on every labelled season ahead of the forecast season.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


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


def match_grouped_cv_indices(
    match_ids: pd.Series | np.ndarray,
    n_splits: int = 10,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """GroupKFold indices where every match stays in a single partition."""
    groups = np.asarray(match_ids)
    splitter = GroupKFold(n_splits=n_splits)
    dummy = np.zeros((len(groups), 1))
    return list(splitter.split(dummy, groups=groups))
