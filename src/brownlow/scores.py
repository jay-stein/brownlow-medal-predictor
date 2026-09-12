"""Per-season score generation with caching.

Each season's scores come from a model trained only on earlier labelled
seasons, so the resulting chronological backtests are leak-free. Scores are
cached under ``data/processed/scores/<model_key>/`` and reused unless
``force=True``.

Two score generators are registered in :mod:`brownlow.model`: the pseudo-Huber
regression baseline and the match-grouped LambdaMART ranking model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pandas as pd

from . import features, model, paths

LABELLED_STATUSES = ("voted", "zero")

KEY_COLUMNS = [
    "PROVIDERID",
    "ROUND_YEAR",
    "PLAYER_PLAYER_PLAYER_PLAYERID",
    "FULL_NAME",
    "TEAM_NAME",
    "ROUND_ROUNDNUMBER",
    "GAME_DATE",
    "LABEL_STATUS",
    "BROWNLOW_VOTES_AUDITED",
]

UTILITY_COLUMN = "UTILITY"


@dataclass
class SeasonScores:
    season: int
    frame: pd.DataFrame
    metadata: dict
    model_key: str = field(default=model.REGRESSION)


def scores_dir(model_key: str = model.REGRESSION):
    directory = paths.PROCESSED_DIR / "scores" / model_key
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def labelled_rows(labelled: pd.DataFrame) -> pd.DataFrame:
    """Rows with a trustworthy label (voted or genuine zero)."""
    return labelled[labelled["LABEL_STATUS"].isin(LABELLED_STATUSES)].copy()


def train_season_scores(
    labelled: pd.DataFrame,
    target_season: int,
    *,
    model_key: str = model.REGRESSION,
    params: dict | None = None,
    n_splits: int = 5,
    num_boost_round: int = 3000,
    early_stopping_rounds: int = 100,
    seed: int = 42,
) -> SeasonScores:
    """Train on all labelled seasons before ``target_season`` and score it."""
    labelled_seasons = sorted(int(s) for s in labelled["ROUND_YEAR"].unique())
    train_seasons = [s for s in labelled_seasons if s < target_season]
    if not train_seasons:
        raise ValueError(f"no labelled seasons before {target_season}")

    fold_params = params if params is not None else model.objective_params(model_key)

    train = labelled_rows(labelled)
    train = train[train["ROUND_YEAR"].isin(train_seasons)].reset_index(drop=True)
    evaluation = labelled[labelled["ROUND_YEAR"] == target_season].reset_index(drop=True)
    if evaluation.empty:
        raise ValueError(f"no rows for season {target_season}")

    preprocessor = features.FeaturePreprocessor().fit(features.feature_frame(train))
    X_train = preprocessor.transform(features.feature_frame(train))
    y_train = train["BROWNLOW_VOTES_AUDITED"].astype(float)

    cv = model.select_best_round(
        X_train,
        y_train,
        train["PROVIDERID"],
        params=fold_params,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
        n_splits=n_splits,
        seed=seed,
    )
    booster = model.fit_utilities(
        X_train,
        y_train,
        cv["best_round"],
        params=fold_params,
        match_ids=train["PROVIDERID"],
        seed=seed,
    )

    X_eval = preprocessor.transform(features.feature_frame(evaluation))
    frame = evaluation.loc[:, KEY_COLUMNS].copy()
    frame[UTILITY_COLUMN] = model.predict_utilities(booster, X_eval)

    metadata = {
        "season": int(target_season),
        "model_key": model_key,
        "objective": fold_params.get("objective"),
        "eval_metric": fold_params.get("eval_metric"),
        "train_seasons": train_seasons,
        "n_train_rows": len(train),
        "n_train_matches": int(train["PROVIDERID"].nunique()),
        "n_eval_rows": len(evaluation),
        "n_eval_matches": int(evaluation["PROVIDERID"].nunique()),
        "best_round": int(cv["best_round"]),
        "fold_best_iterations": cv["fold_best_iterations"],
        "fold_scores": [float(value) for value in cv["fold_scores"]],
        "n_splits": n_splits,
        "num_boost_round": num_boost_round,
        "early_stopping_rounds": int(early_stopping_rounds),
        "seed": int(seed),
    }
    return SeasonScores(int(target_season), frame, metadata, model_key)


def save_scores(scores: SeasonScores) -> None:
    directory = scores_dir(scores.model_key)
    scores.frame.to_parquet(directory / f"scores_{scores.season}.parquet", index=False)
    (directory / f"scores_{scores.season}.json").write_text(json.dumps(scores.metadata, indent=2))


def cached_seasons(model_key: str = model.REGRESSION) -> list[int]:
    """Seasons with cached score files, in ascending order."""
    seasons: list[int] = []
    for path in scores_dir(model_key).glob("scores_*.parquet"):
        try:
            seasons.append(int(path.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return sorted(seasons)


def load_scores(season: int, model_key: str = model.REGRESSION) -> SeasonScores | None:
    directory = scores_dir(model_key)
    parquet = directory / f"scores_{season}.parquet"
    metadata_path = directory / f"scores_{season}.json"
    if not parquet.exists() or not metadata_path.exists():
        return None
    return SeasonScores(
        int(season),
        pd.read_parquet(parquet),
        json.loads(metadata_path.read_text()),
        model_key,
    )
