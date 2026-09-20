"""XGBoost utility models.

Two objectives are supported:

* ``regression`` - robust pseudo-Huber regression on per-game votes (baseline).
* ``ranking``    - match-grouped LambdaMART (``rank:ndcg``) that optimises the
  within-match ordering directly, which is the quantity the 3-2-1 allocation
  depends on.

Round selection uses time-ordered season validation (falling back to rounds
when only one training season exists), so early stopping mimics forecasting
into a later era rather than a random match split. Evaluation uses whole
held-out seasons (see :mod:`brownlow.folds`).
"""

from __future__ import annotations

import os
import subprocess

import numpy as np
import pandas as pd
import xgboost as xgb

from . import features, momentum, pl
from .folds import time_ordered_cv_indices

REGRESSION = "regression"
RANKING = "ranking"
RANKING_RECENT = "ranking_recent"
RANKING_WEIGHTED = "ranking_weighted"
RANKING_NOCBA = "ranking_nocba"
RANKING_PL = "ranking_pl"
RANKING_COACHES = "ranking_coaches"
RANKING_FORM = "ranking_form"
RANKING_SEASON = "ranking_season"
RANKING_CONTEXT = "ranking_context"
RANKING_TIER1 = "ranking_tier1"
RANKING_MOMENTUM = "ranking_momentum"

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

# Direct optimisation of the ordered-triple (Plackett-Luce) likelihood: the
# objective is supplied to XGBoost rather than a built-in rank metric.
PL_PARAMS: dict = {
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
    RANKING_RECENT: RANKING_PARAMS,
    RANKING_WEIGHTED: RANKING_PARAMS,
    RANKING_NOCBA: RANKING_PARAMS,
    RANKING_PL: PL_PARAMS,
    RANKING_COACHES: RANKING_PARAMS,
    RANKING_FORM: RANKING_PARAMS,
    RANKING_SEASON: RANKING_PARAMS,
    RANKING_CONTEXT: RANKING_PARAMS,
    RANKING_TIER1: RANKING_PARAMS,
    RANKING_MOMENTUM: RANKING_PARAMS,
}

CBA_FEATURES = [
    "EXTENDEDSTATS_CENTREBOUNCEATTENDANCES",
    "EXTENDEDSTATS_CENTREBOUNCEATTENDANCES_prop",
]

COACH_FEATURES = ["COACH_VOTES", "COACH_VOTES_SHARE"]

MODEL_OPTIONS: dict[str, dict] = {
    RANKING_RECENT: {"base": RANKING, "train_window": 6},
    RANKING_WEIGHTED: {"base": RANKING, "recency_half_life": 4.0},
    RANKING_NOCBA: {"base": RANKING, "drop_features": CBA_FEATURES},
    RANKING_PL: {"objective_kind": "pl"},
    RANKING_COACHES: {"base": RANKING, "extra_features": COACH_FEATURES},
    RANKING_FORM: {
        "base": RANKING,
        "extra_features": COACH_FEATURES + features.FORM_FEATURES,
    },
    RANKING_SEASON: {
        "base": RANKING,
        "extra_features": COACH_FEATURES + features.SEASON_AGGREGATE_FEATURES,
    },
    RANKING_CONTEXT: {
        "base": RANKING,
        "extra_features": COACH_FEATURES + features.CONTEXT_FEATURES,
    },
    RANKING_TIER1: {
        "base": RANKING,
        "extra_features": (
            COACH_FEATURES
            + features.SEASON_AGGREGATE_FEATURES
            + features.TIER1_FEATURES
        ),
    },
    RANKING_MOMENTUM: {
        "base": RANKING,
        "extra_features": (
            COACH_FEATURES
            + features.SEASON_AGGREGATE_FEATURES
            + momentum.MOMENTUM_FEATURES
        ),
    },
}


def model_options(model_key: str) -> dict:
    """Training options for a registered model key."""
    if model_key not in MODEL_PARAMS:
        raise ValueError(f"unknown model: {model_key!r}")
    return dict(MODEL_OPTIONS.get(model_key, {}))


def is_pl(model_key: str) -> bool:
    return MODEL_OPTIONS.get(model_key, {}).get("objective_kind") == "pl"

# Kept for backwards compatibility with earlier call sites.
DEFAULT_PARAMS = REGRESSION_PARAMS


def objective_params(model_key: str) -> dict:
    """Return a copy of the default parameters for a registered model."""
    if model_key not in MODEL_PARAMS:
        raise ValueError(f"unknown model: {model_key!r}")
    params = dict(MODEL_PARAMS[model_key])
    params.setdefault("device", default_device())
    return params


_DEVICE: str | None = None


def cuda_available() -> bool:
    """Whether XGBoost has CUDA support and an NVIDIA GPU is present."""
    if not xgb.build_info().get("USE_CUDA", False):
        return False
    try:
        subprocess.run(["nvidia-smi"], capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return False
    return True


def default_device() -> str:
    """Training device, read once from ``BROWNLOW_XGB_DEVICE`` (default cpu).

    GPU hist was benchmarked at parity with CPU on this workload (small
    tabular data with thousands of small ranking groups), so CPU stays the
    default for exact reproducibility with the cached score files. Set
    ``BROWNLOW_XGB_DEVICE=cuda`` to opt in.
    """
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = os.environ.get("BROWNLOW_XGB_DEVICE", "").strip().lower() or "cpu"
    return _DEVICE


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
    weight: pd.Series | None = None,
) -> xgb.DMatrix:
    """Build a DMatrix, adding ``qid`` groups for ranking objectives.

    For ranking objectives XGBoost expects one weight per query (match), so
    instance weights are averaged within each match; for other objectives the
    weights are passed through unchanged.
    """
    if ranking:
        if match_ids is None:
            raise ValueError("match_ids are required for ranking objectives")
        match_ids = pd.Series(match_ids).reset_index(drop=True)
        order, qid = _sorted_groups(match_ids)
        labels = None if y is None else pd.Series(y).reset_index(drop=True).iloc[order]
        if weight is not None:
            sorted_weight = pd.Series(weight).reset_index(drop=True).iloc[order].to_numpy(dtype=float)
            group_weight = pd.Series(sorted_weight).groupby(qid).mean().to_numpy(dtype=float)
        else:
            group_weight = None
        return xgb.DMatrix(
            X.iloc[order], label=labels, qid=qid, weight=group_weight, enable_categorical=True
        )
    return xgb.DMatrix(X, label=y, weight=weight, enable_categorical=True)


def _pl_grad_hess(scores: np.ndarray, order: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray]:
    """Gradient and diagonal Hessian of the Plackett-Luce NLL for one match.

    ``order`` is the observed finishing order (3-vote player first). The
    gradient uses the fact that a removed player's weight cancels from later
    denominators, so only the remaining set contributes at each step.
    """
    weights = np.exp((scores - scores.max()) / tau)
    gradient = np.zeros_like(weights)
    hessian = np.zeros_like(weights)
    remaining = np.ones(len(weights), dtype=bool)
    for pick in order:
        total = weights[remaining].sum()
        probabilities = weights / total
        gradient[remaining] += probabilities[remaining]
        gradient[pick] -= 1.0
        hessian[remaining] += probabilities[remaining] * (1.0 - probabilities[remaining]) / tau**2
        remaining[pick] = False
    return gradient, hessian


def pl_objective(tau: float = 1.0):
    """Custom XGBoost objective: negative log-likelihood of observed triples."""

    def objective(preds: np.ndarray, dtrain: xgb.DMatrix) -> tuple[np.ndarray, np.ndarray]:
        labels = dtrain.get_label()
        group_ptr = dtrain.get_uint_info("group_ptr")
        gradient = np.zeros_like(preds)
        hessian = np.zeros_like(preds)
        for group in range(len(group_ptr) - 1):
            lo, hi = int(group_ptr[group]), int(group_ptr[group + 1])
            group_labels = labels[lo:hi]
            recipients = np.flatnonzero(group_labels > 0)
            if len(recipients) != 3 or sorted(group_labels[recipients].tolist()) != [1.0, 2.0, 3.0]:
                continue
            order = recipients[np.argsort(-group_labels[recipients])]
            grad, hess = _pl_grad_hess(preds[lo:hi], order, tau)
            gradient[lo:hi] = grad
            hessian[lo:hi] = hess
        return gradient, hessian

    return objective


def pl_metric(tau: float = 1.0):
    """Custom evaluation metric: mean Plackett-Luce NLL (lower is better)."""

    def metric(preds: np.ndarray, dtrain: xgb.DMatrix) -> tuple[str, float]:
        labels = dtrain.get_label()
        group_ptr = dtrain.get_uint_info("group_ptr")
        total = 0.0
        count = 0
        for group in range(len(group_ptr) - 1):
            lo, hi = int(group_ptr[group]), int(group_ptr[group + 1])
            group_labels = labels[lo:hi]
            recipients = np.flatnonzero(group_labels > 0)
            if len(recipients) != 3 or sorted(group_labels[recipients].tolist()) != [1.0, 2.0, 3.0]:
                continue
            order = recipients[np.argsort(-group_labels[recipients])]
            total += pl.match_log_likelihood(preds[lo:hi], tuple(int(i) for i in order), tau)
            count += 1
        return "pl_nll", -(total / count) if count else 0.0

    return metric


def select_best_round(
    X: pd.DataFrame,
    y: pd.Series,
    match_ids: pd.Series,
    season_ids: pd.Series,
    round_ids: pd.Series,
    params: dict | None = None,
    num_boost_round: int = 3000,
    early_stopping_rounds: int = 100,
    n_splits: int = 3,
    seed: int = 42,
    sample_weight: pd.Series | None = None,
    use_pl: bool = False,
) -> dict:
    """Time-ordered CV to choose the number of boosting rounds.

    Validation folds are the latest seasons of the training window (or the
    latest rounds when only one season is available), each model trained only
    on earlier rows. Returns the mean best iteration across folds plus per-fold
    diagnostics.
    """
    fold_params = {**(params or DEFAULT_PARAMS)}
    grouping = is_ranking(fold_params) or use_pl
    X = X.reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    match_ids = pd.Series(match_ids).reset_index(drop=True)
    season_ids = pd.Series(season_ids).reset_index(drop=True)
    round_ids = pd.Series(round_ids).reset_index(drop=True)
    if sample_weight is not None:
        sample_weight = pd.Series(sample_weight).reset_index(drop=True)

    best_iterations: list[int] = []
    fold_scores: list[float] = []
    fold_seasons: list[list[int]] = []
    for fold, (train_idx, valid_idx) in enumerate(
        time_ordered_cv_indices(season_ids, round_ids, n_splits=n_splits)
    ):
        dtrain = build_dmatrix(
            X.iloc[train_idx],
            y.iloc[train_idx],
            match_ids.iloc[train_idx],
            ranking=grouping,
            weight=None if sample_weight is None else sample_weight.iloc[train_idx],
        )
        dvalid = build_dmatrix(
            X.iloc[valid_idx],
            y.iloc[valid_idx],
            match_ids.iloc[valid_idx],
            ranking=grouping,
            weight=None if sample_weight is None else sample_weight.iloc[valid_idx],
        )
        train_kwargs: dict = {}
        if use_pl:
            train_kwargs = {"obj": pl_objective(), "custom_metric": pl_metric(), "maximize": False}
        booster = xgb.train(
            {**fold_params, "seed": seed + fold},
            dtrain,
            num_boost_round=num_boost_round,
            evals=[(dvalid, "validation")],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False,
            **train_kwargs,
        )
        best_iterations.append(int(booster.best_iteration) + 1)
        fold_scores.append(float(booster.best_score))
        fold_seasons.append(sorted(int(value) for value in season_ids.iloc[valid_idx].unique()))

    return {
        "best_round": int(np.ceil(np.mean(best_iterations))),
        "fold_best_iterations": best_iterations,
        "fold_scores": fold_scores,
        "fold_validation_seasons": fold_seasons,
        "cv_scheme": "time_ordered_seasons",
        "objective": fold_params.get("objective", "pl"),
        "eval_metric": fold_params.get("eval_metric", "pl_nll"),
    }


def fit_utilities(
    X: pd.DataFrame,
    y: pd.Series,
    num_boost_round: int,
    params: dict | None = None,
    match_ids: pd.Series | None = None,
    seed: int = 42,
    sample_weight: pd.Series | None = None,
    use_pl: bool = False,
) -> xgb.Booster:
    """Train the utility model on all supplied rows."""
    fold_params = {**(params or DEFAULT_PARAMS)}
    grouping = is_ranking(fold_params) or use_pl
    dtrain = build_dmatrix(X, y, match_ids, ranking=grouping, weight=sample_weight)
    train_kwargs: dict = {}
    if use_pl:
        train_kwargs = {"obj": pl_objective()}
    return xgb.train(
        {**fold_params, "seed": seed},
        dtrain,
        num_boost_round=num_boost_round,
        verbose_eval=False,
        **train_kwargs,
    )


def predict_utilities(booster: xgb.Booster, X: pd.DataFrame) -> np.ndarray:
    return booster.predict(xgb.DMatrix(X, enable_categorical=True))
