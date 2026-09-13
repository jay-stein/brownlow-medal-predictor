"""Loss-based stacking ensemble over diverse score generators.

Members are utilities from different model families and objectives:

- ``ranking_coaches`` — production LambdaMART with match statistics and
  coaches' votes;
- ``ranking_pl`` — direct Plackett-Luce likelihood optimisation (no coaches);
- ``rf_coaches`` — a random-forest regressor with coaches' votes (a practical
  stand-in for the conditional-inference forests used by MoS, which are only
  available in R).

Within each match, every member's utilities are z-scored and combined with
weights on the simplex. Weights, temperature and effect scale are selected on
earlier out-of-sample seasons by the same predeclared criterion used for the
production model: equal-weight z-combination of effect-integrated match NLL
and contender CRPS. A pure zero-weights member is allowed, so the ensemble can
fall back to any single member.
"""

from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from . import features, model, paths, scores, simulate, validation

UTILITY_COLUMN = scores.UTILITY_COLUMN
KEY_COLUMNS = scores.KEY_COLUMNS
MATCH_COLUMN = simulate.MATCH_COLUMN

RF_DIR = paths.PROCESSED_DIR / "ensemble" / "rf_coaches"
WEIGHT_STEPS = (0.0, 0.25, 0.5, 0.75, 1.0)


def rf_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    base = features.feature_frame(frame)
    extra = [column for column in model.COACH_FEATURES if column in frame.columns]
    if extra:
        base = pd.concat([base, frame[extra]], axis=1)
    return base


def train_rf_scores(
    labelled: pd.DataFrame,
    target_season: int,
    *,
    n_estimators: int = 400,
    max_features: float = 0.3,
    min_samples_leaf: int = 5,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Random-forest member: predict votes from features plus coaches' votes."""
    train = scores.labelled_rows(labelled)
    train = train[train["ROUND_YEAR"] < target_season].reset_index(drop=True)
    evaluation = labelled[labelled["ROUND_YEAR"] == target_season].reset_index(drop=True)
    if train.empty or evaluation.empty:
        raise ValueError(f"missing rows for RF season {target_season}")

    preprocessor = features.FeaturePreprocessor().fit(rf_feature_frame(train))
    X_train = preprocessor.transform(rf_feature_frame(train)).copy()
    X_eval = preprocessor.transform(rf_feature_frame(evaluation)).copy()
    for frame in (X_train, X_eval):
        for column in features.CATEGORICAL_FEATURES:
            if column in frame.columns:
                frame[column] = pd.Categorical(frame[column]).codes

    forest = RandomForestRegressor(
        n_estimators=n_estimators,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        n_jobs=-1,
        random_state=seed,
    )
    forest.fit(X_train, train["BROWNLOW_VOTES_AUDITED"].astype(float))
    result = evaluation.loc[:, KEY_COLUMNS].copy()
    result[UTILITY_COLUMN] = forest.predict(X_eval)
    metadata = {
        "season": int(target_season),
        "model_key": "rf_coaches",
        "n_estimators": n_estimators,
        "max_features": max_features,
        "min_samples_leaf": min_samples_leaf,
        "n_train_rows": len(train),
        "n_eval_rows": len(evaluation),
        "seed": int(seed),
    }
    return result, metadata


def load_or_train_rf(labelled: pd.DataFrame, season: int, force: bool = False) -> pd.DataFrame:
    path = RF_DIR / f"scores_{season}.parquet"
    if path.exists() and not force:
        return pd.read_parquet(path)
    frame, metadata = train_rf_scores(labelled, season)
    RF_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    (RF_DIR / f"scores_{season}.json").write_text(json.dumps(metadata, indent=2))
    return frame


def member_frame(
    member: str, labelled: pd.DataFrame, season: int, force: bool = False
) -> pd.DataFrame:
    """Utilities for one ensemble member and season, training if necessary."""
    if member == "rf_coaches":
        return load_or_train_rf(labelled, season, force=force)
    cached = scores.load_scores(season, member)
    if cached is None:
        raise ValueError(
            f"no cached scores for member {member} season {season}: "
            f"run `baseline --model {member} --seasons {season}` first"
        )
    return cached.frame


def combine_scores(
    member_frames: dict[str, pd.DataFrame],
    weights: dict[str, float],
) -> pd.DataFrame:
    """Weighted sum of globally standardised member utilities.

    Each member is z-scored over the whole season (not within matches), which
    preserves the relative spread of utilities inside a match — the quantity
    the Plackett-Luce allocation depends on. A pure member therefore
    reproduces its raw model up to a global scale absorbed by the temperature.
    """
    names = list(member_frames)
    base = member_frames[names[0]].reset_index(drop=True)
    keys = base[KEY_COLUMNS].reset_index(drop=True)
    for name in names[1:]:
        other = member_frames[name].reset_index(drop=True)
        if len(other) != len(base) or not other[KEY_COLUMNS].equals(keys):
            raise ValueError(f"member {name} is not row-aligned with {names[0]}")

    combined = base[KEY_COLUMNS].copy()
    combined[UTILITY_COLUMN] = 0.0
    for name in names:
        weight = float(weights.get(name, 0.0))
        if weight == 0.0:
            continue
        utilities = member_frames[name][UTILITY_COLUMN].to_numpy(dtype=float)
        spread = float(np.std(utilities))
        scale = spread if np.isfinite(spread) and spread > 0.0 else 1.0
        z = (utilities - float(np.mean(utilities))) / scale
        combined[UTILITY_COLUMN] += weight * z
    return combined


def weight_candidates(names: list[str]) -> list[dict[str, float]]:
    """Simplex grid over member weights (equal steps, summing to one)."""
    candidates = []
    for values in itertools.product(WEIGHT_STEPS, repeat=len(names)):
        if abs(sum(values) - 1.0) < 1e-9:
            candidates.append(dict(zip(names, values)))
    return candidates


def _weight_label(weights: dict[str, float]) -> str:
    return ",".join(f"{name}:{value:g}" for name, value in weights.items())


def ensemble_grid(
    members_by_season: dict[int, dict[str, pd.DataFrame]],
    seasons: list[int],
    weight_sets: list[dict[str, float]],
    tau_values: list[float],
    scales: list[float],
    *,
    n_sims: int = 1000,
    n_draws: int = 256,
    contender_count: int = simulate.DEFAULT_CONTENDERS,
    seed: int = 42,
) -> pd.DataFrame:
    """Score every (weights, tau, scale) combination on the requested seasons."""
    combined_cache: dict[tuple[str, int], pd.DataFrame] = {}
    for weights in weight_sets:
        label = _weight_label(weights)
        for season in seasons:
            combined_cache[(label, season)] = combine_scores(members_by_season[season], weights)

    rows: list[dict] = []
    for weights in weight_sets:
        label = _weight_label(weights)
        weight_columns = {f"w_{name}": float(value) for name, value in weights.items()}
        for season in seasons:
            combined = combined_cache[(label, season)]
            for tau in tau_values:
                for scale in scales:
                    match_nll = simulate.integrated_match_log_loss(
                        combined, float(tau), float(scale), n_draws=n_draws, seed=seed
                    )
                    simulation = simulate.simulate_season(
                        combined, float(tau), effect_scale=float(scale), n_sims=n_sims, seed=seed
                    )
                    metrics = simulate.season_metrics(simulation, contender_count=contender_count)
                    rows.append(
                        {
                            "weights": label,
                            **weight_columns,
                            "tau": float(tau),
                            "scale": float(scale),
                            "season": int(season),
                            "match_nll": match_nll,
                            **metrics,
                        }
                    )
    return pd.DataFrame(rows)


def rolling_ensemble(
    grid: pd.DataFrame,
    *,
    report_seasons: list[int],
    min_evidence: int = validation.MIN_EVIDENCE_SEASONS,
) -> pd.DataFrame:
    """Fully rolling ensemble records: weights and parameters from prior seasons."""
    weight_columns = [column for column in grid.columns if column.startswith("w_")]
    group_columns = weight_columns + ["tau", "scale"]
    all_seasons = sorted(int(season) for season in grid["season"].unique())
    rows: list[dict] = []
    for season in sorted(report_seasons):
        evidence = [candidate for candidate in all_seasons if candidate < season]
        if len(evidence) < min_evidence:
            continue
        subset = grid[grid["season"].isin(evidence)]
        summary = subset.groupby(group_columns, as_index=False).agg(
            match_nll=("match_nll", "mean"), crps=("mean_crps_contenders", "mean")
        )
        summary = validation._standardise(summary, validation.NLL_WEIGHT, validation.CRPS_WEIGHT)
        best = summary.sort_values(["combined", "match_nll"]).iloc[0]
        mask = pd.Series(True, index=grid.index)
        for column in group_columns:
            mask &= grid[column] == best[column]
        match = grid[mask & (grid["season"] == season)]
        if match.empty:
            continue
        record = match.iloc[0].to_dict()
        selected_weights = ",".join(
            f"{column[2:]}:{best[column]:g}" for column in weight_columns if best[column] > 0.0
        )
        record.update(
            {
                "evidence_seasons": len(evidence),
                "weights_selected": selected_weights,
                "tau_selected": float(best["tau"]),
                "scale_selected": float(best["scale"]),
                "selection_combined": float(best["combined"]),
            }
        )
        rows.append(record)
    return pd.DataFrame(rows)
