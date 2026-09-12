"""Feature engineering ported from ``notebooks/xgboost_predict_brownlow_v2.ipynb``.

The feature list is intentionally identical to the notebook's 120-feature set so
the frozen baseline scores remain comparable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NAN_THRESHOLD = 0.05

POSITION_FEATURE = "PLAYER_PLAYER_POSITION"

RAW_STAT_FEATURES = [
    "GOALS",
    "BEHINDS",
    "KICKS",
    "HANDBALLS",
    "DISPOSALS",
    "MARKS",
    "BOUNCES",
    "TACKLES",
    "CONTESTEDPOSSESSIONS",
    "UNCONTESTEDPOSSESSIONS",
    "TOTALPOSSESSIONS",
    "INSIDE50S",
    "MARKSINSIDE50",
    "CONTESTEDMARKS",
    "HITOUTS",
    "ONEPERCENTERS",
    "DISPOSALEFFICIENCY",
    "CLANGERS",
    "FREESFOR",
    "FREESAGAINST",
    "DREAMTEAMPOINTS",
    "REBOUND50S",
    "GOALASSISTS",
    "GOALACCURACY",
    "RATINGPOINTS",
    "TURNOVERS",
    "INTERCEPTS",
    "TACKLESINSIDE50",
    "SHOTSATGOAL",
    "SCOREINVOLVEMENTS",
    "METRESGAINED",
    "CLEARANCES_CENTRECLEARANCES",
    "CLEARANCES_STOPPAGECLEARANCES",
    "CLEARANCES_TOTALCLEARANCES",
    "EXTENDEDSTATS_EFFECTIVEKICKS",
    "EXTENDEDSTATS_KICKEFFICIENCY",
    "EXTENDEDSTATS_KICKTOHANDBALLRATIO",
    "EXTENDEDSTATS_EFFECTIVEDISPOSALS",
    "EXTENDEDSTATS_MARKSONLEAD",
    "EXTENDEDSTATS_INTERCEPTMARKS",
    "EXTENDEDSTATS_CONTESTEDPOSSESSIONRATE",
    "EXTENDEDSTATS_HITOUTSTOADVANTAGE",
    "EXTENDEDSTATS_HITOUTWINPERCENTAGE",
    "EXTENDEDSTATS_HITOUTTOADVANTAGERATE",
    "EXTENDEDSTATS_GROUNDBALLGETS",
    "EXTENDEDSTATS_F50GROUNDBALLGETS",
    "EXTENDEDSTATS_SCORELAUNCHES",
    "EXTENDEDSTATS_PRESSUREACTS",
    "EXTENDEDSTATS_DEFHALFPRESSUREACTS",
    "EXTENDEDSTATS_SPOILS",
    "EXTENDEDSTATS_RUCKCONTESTS",
    "EXTENDEDSTATS_CONTESTDEFONEONONES",
    "EXTENDEDSTATS_CONTESTDEFLOSSES",
    "EXTENDEDSTATS_CONTESTDEFLOSSPERCENTAGE",
    "EXTENDEDSTATS_CONTESTOFFONEONONES",
    "EXTENDEDSTATS_CONTESTOFFWINS",
    "EXTENDEDSTATS_CONTESTOFFWINSPERCENTAGE",
]

# The two kick-efficiency stats are absent from the notebook's proportion list.
PROPORTION_FEATURES = [
    col
    for col in RAW_STAT_FEATURES
    if col not in {"EXTENDEDSTATS_KICKEFFICIENCY", "EXTENDEDSTATS_KICKTOHANDBALLRATIO"}
]

ENGINEERED_FEATURES = [
    "PLAYER_POINTS",
    "PLAYER_CAPTAIN",
    "AT_HOME",
    "PLAYERTEAMSCORE",
    "PLAYER_MINUTESINFRONT",
    "PLAYERTEAM_MARGIN",
    "PLAYERTEAM_OUTCOME",
]

FEATURE_LIST = (
    [POSITION_FEATURE]
    + RAW_STAT_FEATURES
    + ENGINEERED_FEATURES
    + [f"{col}_prop" for col in PROPORTION_FEATURES]
)

CATEGORICAL_FEATURES = [POSITION_FEATURE, "PLAYERTEAM_OUTCOME"]

TEAM_COLS_TO_KEEP = [
    "MATCHID",
    "ROUND_YEAR",
    "HOMETEAMSCORE_MINUTESINFRONT",
    "AWAYTEAMSCORE_MINUTESINFRONT",
    "HOMETEAMSCORE_MATCHSCORE_TOTALSCORE",
    "AWAYTEAMSCORE_MATCHSCORE_TOTALSCORE",
]


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Upper-case column names and replace dots (literal, not regex)."""
    out = df.copy()
    out.columns = out.columns.str.upper().str.replace(".", "_", regex=False)
    return out


def derive_game_date(df: pd.DataFrame) -> pd.Series:
    """Local game date from UTC start time (UTC+10, as in the original notebook)."""
    timeseries_utc = pd.to_datetime(df["UTCSTARTTIME"], utc=True)
    return (timeseries_utc + pd.Timedelta(hours=10)).dt.date


def add_full_name(df: pd.DataFrame) -> pd.Series:
    return (df["PLAYER_GIVENNAME"].fillna("") + " " + df["PLAYER_SURNAME"].fillna("")).str.upper().str.strip()


def nan_fraction_columns(df: pd.DataFrame, threshold: float = NAN_THRESHOLD) -> list[str]:
    fractions = df.isna().mean().sort_values(ascending=False)
    return fractions[fractions > threshold].index.tolist()


def add_team_context(df: pd.DataFrame) -> pd.DataFrame:
    """Outcome features derived from the merged team statistics."""
    out = df.copy()
    out["AT_HOME"] = (out["HOME_TEAM_NAME"] == out["TEAM_NAME"]).astype(int)
    home_score = out["HOMETEAMSCORE_MATCHSCORE_TOTALSCORE"]
    away_score = out["AWAYTEAMSCORE_MATCHSCORE_TOTALSCORE"]
    out["PLAYERTEAMSCORE"] = np.where(out["AT_HOME"] == 1, home_score, away_score)
    out["PLAYER_MINUTESINFRONT"] = np.where(
        out["AT_HOME"] == 1,
        out["HOMETEAMSCORE_MINUTESINFRONT"],
        out["AWAYTEAMSCORE_MINUTESINFRONT"],
    )
    out["PLAYERTEAM_MARGIN"] = np.where(
        out["AT_HOME"] == 1, home_score - away_score, away_score - home_score
    )
    out["PLAYERTEAM_OUTCOME"] = np.where(
        out["PLAYERTEAM_MARGIN"] > 0,
        "WIN",
        np.where(out["PLAYERTEAM_MARGIN"] == 0, "DRAW", "LOSE"),
    )
    return out


def add_proportion_features(df: pd.DataFrame) -> pd.DataFrame:
    """Within-team share of each stat, per match (``<stat>_prop``)."""
    proportions = df.groupby(["PROVIDERID", "TEAM_NAME"])[PROPORTION_FEATURES].transform(
        lambda x: x / x.sum()
    )
    proportions = proportions.fillna(0)
    proportions.columns = [f"{col}_prop" for col in proportions.columns]
    return pd.concat([df, proportions], axis=1)


def build_feature_table(
    player_stats: pd.DataFrame,
    team_stats: pd.DataFrame,
    train_through: int = 2025,
) -> pd.DataFrame:
    """Build the merged feature table (finals excluded, labels not yet attached).

    ``train_through`` controls which seasons inform the >5% NaN column drop so
    that no evaluation-season information leaks into feature selection.
    """
    df = normalise_columns(player_stats)
    df = df[~df["ROUND_NAME"].astype(str).str.contains("Final", case=False, na=False)].copy()
    df["GAME_DATE"] = derive_game_date(df)
    df["FULL_NAME"] = add_full_name(df)

    seasons = pd.to_datetime(df["GAME_DATE"]).dt.year
    reference = df[seasons <= train_through]
    drop_cols = nan_fraction_columns(reference)
    df = df.drop(columns=drop_cols)

    df["PLAYER_POINTS"] = df["GOALS"] * 6 + df["BEHINDS"]

    teams = normalise_columns(team_stats)
    df = df.merge(
        teams[TEAM_COLS_TO_KEEP],
        how="inner",
        left_on="PROVIDERID",
        right_on="MATCHID",
    )

    df = add_team_context(df)
    df = add_proportion_features(df)
    return df


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Select the model feature columns in the canonical order."""
    return df.loc[:, FEATURE_LIST]


class FeaturePreprocessor:
    """Train-only imputation and categorical encoding.

    Fit on the training seasons, then apply to the evaluation season so that
    learned preprocessing never sees evaluation data.
    """

    def __init__(self) -> None:
        self.numeric_features_: list[str] = []
        self.means_: pd.Series | None = None
        self.categories_: dict[str, pd.Index] = {}

    def fit(self, X: pd.DataFrame) -> FeaturePreprocessor:
        numeric = X.select_dtypes(include=["number"]).columns.tolist()
        self.numeric_features_ = [col for col in numeric if col in FEATURE_LIST]
        self.means_ = X[self.numeric_features_].mean()
        self.categories_ = {
            col: pd.Categorical(X[col]).categories for col in CATEGORICAL_FEATURES if col in X
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.means_ is None:
            raise RuntimeError("FeaturePreprocessor must be fitted before transform()")
        out = X.copy()
        out[self.numeric_features_] = out[self.numeric_features_].fillna(self.means_)
        for col, categories in self.categories_.items():
            dtype = pd.CategoricalDtype(categories=categories)
            values = out[col]
            out[col] = values.where(values.isin(categories)).astype(dtype)
        return out
