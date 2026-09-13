"""Legacy baseline: feature-bagged XGBoost plus independent Normal counts.

Faithful reconstruction of the first-generation pipeline, recovered from the
original notebooks (`xgboost_predict_brownlow_v2.ipynb` and
`montecarlo_predict_brownlow_v1.ipynb`):

- 100 pseudo-Huber XGBoost regressors, each trained on a random 70% of the
  120 match-stat features (no coaches' votes, no centre-bounce attendances,
  no height);
- per player-game, the mean and spread across the 100 models are used as
  ``Normal(mean, std)`` draws that are summed to season totals, with **no
  match budget** and the raw (unclipped) draws;
- round selection uses the same time-ordered season validation as the current
  pipeline, so the comparison isolates the modelling method.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from . import features, model, paths, scores, simulate

LEGACY_MODELS = 100
LEGACY_FEATURE_FRACTION = 0.7
LEGACY_ROUNDS = 100  # the original cross-validation selected 98
EPS = 1e-6

LEGACY_DROP = [
    "EXTENDEDSTATS_CENTREBOUNCEATTENDANCES",
    "EXTENDEDSTATS_CENTREBOUNCEATTENDANCES_prop",
    "PLAYER_HEIGHT",
    "HEIGHT_VS_POSITION",
]

LEGACY_DIR = paths.PROCESSED_DIR / "legacy"


def legacy_cached(season: int) -> Path:
    return LEGACY_DIR / f"scores_{season}.parquet"


def load_or_train_legacy(
    labelled: pd.DataFrame,
    target_season: int,
    *,
    force: bool = False,
    n_models: int = LEGACY_MODELS,
    feature_fraction: float = LEGACY_FEATURE_FRACTION,
    num_boost_round: int = LEGACY_ROUNDS,
    seed: int = 42,
) -> pd.DataFrame:
    """Cached per-season legacy predictions (100-model training is expensive)."""
    path = legacy_cached(target_season)
    if path.exists() and not force:
        return pd.read_parquet(path)
    frame, metadata = train_legacy_ensemble(
        labelled,
        target_season,
        n_models=n_models,
        feature_fraction=feature_fraction,
        num_boost_round=num_boost_round,
        seed=seed,
    )
    LEGACY_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    (LEGACY_DIR / f"scores_{target_season}.json").write_text(json.dumps(metadata, indent=2))
    return frame


def legacy_feature_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """The original 120-feature set: base features minus CBA and height."""
    columns = features.feature_frame(frame)
    drop = [column for column in LEGACY_DROP if column in columns.columns]
    return columns.drop(columns=drop)


def train_legacy_ensemble(
    labelled: pd.DataFrame,
    target_season: int,
    *,
    n_models: int = LEGACY_MODELS,
    feature_fraction: float = LEGACY_FEATURE_FRACTION,
    num_boost_round: int = LEGACY_ROUNDS,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Train the feature-bagged ensemble on earlier labels and score one season."""
    train = scores.labelled_rows(labelled)
    train = train[train["ROUND_YEAR"] < target_season].reset_index(drop=True)
    evaluation = labelled[labelled["ROUND_YEAR"] == target_season].reset_index(drop=True)
    if train.empty or evaluation.empty:
        raise ValueError(f"missing rows for legacy season {target_season}")

    preprocessor = features.FeaturePreprocessor().fit(legacy_feature_columns(train))
    X_train = preprocessor.transform(legacy_feature_columns(train))
    X_eval = preprocessor.transform(legacy_feature_columns(evaluation))
    y_train = train["BROWNLOW_VOTES_AUDITED"].astype(float)

    rng = np.random.default_rng(seed)
    columns = list(X_train.columns)
    k_features = max(1, int(round(feature_fraction * len(columns))))
    params = {**model.REGRESSION_PARAMS, "device": model.default_device()}
    predictions = []
    for index in range(n_models):
        subset = sorted(rng.choice(columns, size=k_features, replace=False))
        dtrain = xgb.DMatrix(X_train[subset], label=y_train, enable_categorical=True)
        booster = xgb.train(
            {**params, "seed": 1000 + index},
            dtrain,
            num_boost_round=num_boost_round,
            verbose_eval=False,
        )
        dtest = xgb.DMatrix(X_eval[subset], enable_categorical=True)
        predictions.append(booster.predict(dtest))

    stack = np.vstack(predictions)
    frame = evaluation.loc[:, scores.KEY_COLUMNS].copy()
    frame["pred_mean"] = stack.mean(axis=0)
    frame["pred_std"] = stack.std(axis=0)
    metadata = {
        "season": int(target_season),
        "model_key": "legacy_ensemble",
        "n_models": n_models,
        "feature_fraction": feature_fraction,
        "k_features": k_features,
        "num_boost_round": num_boost_round,
        "train_seasons": sorted(int(s) for s in train["ROUND_YEAR"].unique()),
        "n_train_rows": len(train),
        "n_eval_rows": len(evaluation),
        "seed": int(seed),
    }
    return frame, metadata


def simulate_legacy_season(
    frame: pd.DataFrame,
    n_sims: int = 1000,
    seed: int = 42,
    ineligible: set[str] | None = None,
) -> simulate.SeasonSimulation:
    """Sum independent Normal draws per player-game into season totals."""
    work = frame[frame[simulate.PLAYER_COLUMN].notna()].copy()
    work["PLAYER_KEY"] = work[simulate.PLAYER_COLUMN].astype(str)
    players = (
        work.groupby("PLAYER_KEY", as_index=False)
        .agg(FULL_NAME=("FULL_NAME", "last"), TEAM_NAME=("TEAM_NAME", "last"))
    )
    observed = work.groupby("PLAYER_KEY")[simulate.VOTE_COLUMN].sum(min_count=1).rename("observed_votes")
    players = players.merge(observed, on="PLAYER_KEY").rename(columns={"PLAYER_KEY": simulate.PLAYER_COLUMN})

    positions = {key: index for index, key in enumerate(players[simulate.PLAYER_COLUMN])}
    indices = np.array([positions[key] for key in work["PLAYER_KEY"]], dtype=int)
    means = work["pred_mean"].to_numpy(dtype=float)
    stds = work["pred_std"].to_numpy(dtype=float)
    stds = np.where(np.isfinite(stds) & (stds > EPS), stds, EPS)

    rng = np.random.default_rng(seed)
    draws = rng.normal(loc=means[:, None], scale=stds[:, None], size=(len(work), n_sims))
    totals = np.zeros((len(players), n_sims))
    np.add.at(totals, indices, draws)

    eligible = simulate.eligible_mask(players, ineligible)
    summary = simulate.summarise_totals(players, totals, eligible)
    return simulate.SeasonSimulation(
        season=frame[simulate.SEASON_COLUMN].iloc[0],
        players=summary,
        totals=totals,
        tau=float("nan"),
        effect_scale=float("nan"),
    )
