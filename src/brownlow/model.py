"""Baseline XGBoost utility model.

Round selection uses match-grouped cross-validation and explicitly takes the
mean best iteration across folds, rather than the length of a truncated CV
history. Evaluation uses whole held-out seasons (see :mod:`brownlow.folds`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from .folds import match_grouped_cv_indices

DEFAULT_PARAMS: dict = {
    "objective": "reg:pseudohubererror",
    "eval_metric": "mae",
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "verbosity": 0,
}


def select_best_round(
    X: pd.DataFrame,
    y: pd.Series,
    match_ids: pd.Series,
    params: dict | None = None,
    num_boost_round: int = 3000,
    early_stopping_rounds: int = 100,
    n_splits: int = 10,
    seed: int = 42,
) -> dict:
    """Match-grouped CV to choose the number of boosting rounds.

    Returns the mean best iteration across folds plus per-fold diagnostics.
    """
    fold_params = {**(params or DEFAULT_PARAMS)}
    X = X.reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    match_ids = pd.Series(match_ids).reset_index(drop=True)

    best_iterations: list[int] = []
    fold_mae: list[float] = []
    for fold, (train_idx, valid_idx) in enumerate(
        match_grouped_cv_indices(match_ids, n_splits=n_splits)
    ):
        dtrain = xgb.DMatrix(X.iloc[train_idx], label=y.iloc[train_idx], enable_categorical=True)
        dvalid = xgb.DMatrix(X.iloc[valid_idx], label=y.iloc[valid_idx], enable_categorical=True)
        booster = xgb.train(
            {**fold_params, "seed": seed + fold},
            dtrain,
            num_boost_round=num_boost_round,
            evals=[(dvalid, "validation")],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False,
        )
        best_iterations.append(int(booster.best_iteration) + 1)
        fold_mae.append(float(booster.best_score))

    return {
        "best_round": int(np.ceil(np.mean(best_iterations))),
        "fold_best_iterations": best_iterations,
        "fold_mae": fold_mae,
    }


def fit_utilities(
    X: pd.DataFrame,
    y: pd.Series,
    num_boost_round: int,
    params: dict | None = None,
    seed: int = 42,
) -> xgb.Booster:
    """Train the baseline utility model on all supplied rows."""
    dtrain = xgb.DMatrix(X, label=y, enable_categorical=True)
    return xgb.train(
        {**(params or DEFAULT_PARAMS), "seed": seed},
        dtrain,
        num_boost_round=num_boost_round,
        verbose_eval=False,
    )


def predict_utilities(booster: xgb.Booster, X: pd.DataFrame) -> np.ndarray:
    return booster.predict(xgb.DMatrix(X, enable_categorical=True))
