"""XGBoost utility models.

Two objectives are supported:

* ``regression`` - robust pseudo-Huber regression on per-game votes (baseline).
* ``ranking``    - match-grouped LambdaMART (``rank:ndcg``) that optimises the
  within-match ordering directly, which is the quantity the 3-2-1 allocation
  depends on.

Round selection uses match-grouped cross-validation and explicitly takes the
mean best iteration across folds, rather than the length of a truncated CV
history. Evaluation uses whole held-out seasons (see :mod:`brownlow.folds`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from .folds import match_grouped_cv_indices

REGRESSION = "regression"
RANKING = "ranking"

REGRESSION_PARAMS: dict = {
    "objective": "reg:pseudohubererror",
    "eval_metric": "mae",
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "verbosity": 0,
}

RANKING_PARAMS: dict = {
    "objective": "rank:ndcg",
    "eval_metric": "ndcg@3",
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "verbosity": 0,
}

MODEL_PARAMS: dict[str, dict] = {
    REGRESSION: REGRESSION_PARAMS,
    RANKING: RANKING_PARAMS,
}

# Kept for backwards compatibility with earlier call sites.
DEFAULT_PARAMS = REGRESSION_PARAMS


def objective_params(model_key: str) -> dict:
    """Return a copy of the default parameters for a registered model."""
    if model_key not in MODEL_PARAMS:
        raise ValueError(f"unknown model: {model_key!r}")
    return dict(MODEL_PARAMS[model_key])


def is_ranking(params: dict) -> bool:
    return str(params.get("objective", "")).startswith("rank:")


def _sorted_groups(match_ids: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Order rows by match and build contiguous group ids for ranking."""
    order = np.argsort(match_ids.to_numpy(), kind="stable")
    qid = pd.factorize(match_ids.to_numpy()[order], sort=False)[0]
    return order, qid


def build_dmatrix(
    X: pd.DataFrame,
    y: pd.Series | None = None,
    match_ids: pd.Series | None = None,
    ranking: bool = False,
) -> xgb.DMatrix:
    """Build a DMatrix, adding ``qid`` groups for ranking objectives."""
    if ranking:
        if match_ids is None:
            raise ValueError("match_ids are required for ranking objectives")
        match_ids = pd.Series(match_ids).reset_index(drop=True)
        order, qid = _sorted_groups(match_ids)
        labels = None if y is None else pd.Series(y).reset_index(drop=True).iloc[order]
        return xgb.DMatrix(X.iloc[order], label=labels, qid=qid, enable_categorical=True)
    return xgb.DMatrix(X, label=y, enable_categorical=True)


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
    ranking = is_ranking(fold_params)
    X = X.reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    match_ids = pd.Series(match_ids).reset_index(drop=True)

    best_iterations: list[int] = []
    fold_scores: list[float] = []
    for fold, (train_idx, valid_idx) in enumerate(
        match_grouped_cv_indices(match_ids, n_splits=n_splits)
    ):
        dtrain = build_dmatrix(
            X.iloc[train_idx], y.iloc[train_idx], match_ids.iloc[train_idx], ranking=ranking
        )
        dvalid = build_dmatrix(
            X.iloc[valid_idx], y.iloc[valid_idx], match_ids.iloc[valid_idx], ranking=ranking
        )
        booster = xgb.train(
            {**fold_params, "seed": seed + fold},
            dtrain,
            num_boost_round=num_boost_round,
            evals=[(dvalid, "validation")],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False,
        )
        best_iterations.append(int(booster.best_iteration) + 1)
        fold_scores.append(float(booster.best_score))

    return {
        "best_round": int(np.ceil(np.mean(best_iterations))),
        "fold_best_iterations": best_iterations,
        "fold_scores": fold_scores,
        "objective": fold_params.get("objective"),
        "eval_metric": fold_params.get("eval_metric"),
    }


def fit_utilities(
    X: pd.DataFrame,
    y: pd.Series,
    num_boost_round: int,
    params: dict | None = None,
    match_ids: pd.Series | None = None,
    seed: int = 42,
) -> xgb.Booster:
    """Train the utility model on all supplied rows."""
    fold_params = {**(params or DEFAULT_PARAMS)}
    dtrain = build_dmatrix(X, y, match_ids, ranking=is_ranking(fold_params))
    return xgb.train(
        {**fold_params, "seed": seed},
        dtrain,
        num_boost_round=num_boost_round,
        verbose_eval=False,
    )


def predict_utilities(booster: xgb.Booster, X: pd.DataFrame) -> np.ndarray:
    return booster.predict(xgb.DMatrix(X, enable_categorical=True))
