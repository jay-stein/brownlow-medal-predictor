"""Feature engineering for the Brownlow pipeline.

The 124-feature set evolved from the original notebook model; the full
definition lives in :data:`FEATURE_LIST`.
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
    "EXTENDEDSTATS_CENTREBOUNCEATTENDANCES",
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

HEIGHT_FEATURES = ["PLAYER_HEIGHT", "HEIGHT_VS_POSITION"]

# Features that are deliberately left missing for XGBoost to handle natively:
# centre bounce attendances only exist from 2021, and heights are unavailable
# for a small number of players.
NATIVE_MISSING_FEATURES = [
    "EXTENDEDSTATS_CENTREBOUNCEATTENDANCES",
    "EXTENDEDSTATS_CENTREBOUNCEATTENDANCES_prop",
    "PLAYER_HEIGHT",
    "HEIGHT_VS_POSITION",
]

FEATURE_LIST = (
    [POSITION_FEATURE]
    + RAW_STAT_FEATURES
    + ENGINEERED_FEATURES
    + HEIGHT_FEATURES
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
    # Keep era-missing stats missing rather than pretending they are zero.
    for column in PROPORTION_FEATURES:
        prop_column = f"{column}_prop"
        if prop_column in NATIVE_MISSING_FEATURES:
            proportions[prop_column] = proportions[prop_column].where(df[column].notna())
    return pd.concat([df, proportions], axis=1)


def add_height_features(df: pd.DataFrame, player_details: pd.DataFrame) -> pd.DataFrame:
    """Attach player height and height relative to same-position peers in the match.

    The squad endpoint reports one row per player-season with ``providerId``
    (Champion Data id), ``position`` and ``heightInCm``.
    """
    details = normalise_columns(player_details)
    details = details.rename(columns={"HEIGHTINCM": "HEIGHT_CM"})
    details = details[["PROVIDERID", "SEASON", "POSITION", "HEIGHT_CM"]].copy()
    details["HEIGHT_CM"] = pd.to_numeric(details["HEIGHT_CM"], errors="coerce")
    details.loc[details["HEIGHT_CM"] <= 0, "HEIGHT_CM"] = pd.NA

    out = df.copy()
    out["SEASON_KEY"] = pd.to_datetime(out["GAME_DATE"]).dt.year
    height_by_player = details.groupby("PROVIDERID")["HEIGHT_CM"].max()
    out["PLAYER_HEIGHT"] = out["PLAYER_PLAYER_PLAYER_PLAYERID"].map(height_by_player)

    positions = details.dropna(subset=["POSITION"]).drop_duplicates(["PROVIDERID", "SEASON"])
    position_by_season = positions.set_index(["PROVIDERID", "SEASON"])["POSITION"]
    player_id = out["PLAYER_PLAYER_PLAYER_PLAYERID"]
    keys = pd.MultiIndex.from_arrays([player_id, out["SEASON_KEY"]])
    out["POSITION_GROUP"] = position_by_season.reindex(keys).to_numpy()

    latest_position = (
        positions.sort_values("SEASON").drop_duplicates("PROVIDERID", keep="last")
        .set_index("PROVIDERID")["POSITION"]
    )
    out["POSITION_GROUP"] = (
        out["POSITION_GROUP"].fillna(player_id.map(latest_position)).fillna("UNKNOWN")
    )

    grouped = out.groupby(["PROVIDERID", "POSITION_GROUP"])["PLAYER_HEIGHT"]
    group_sum = grouped.transform("sum")
    group_count = grouped.transform("count")
    peer_mean = (group_sum - out["PLAYER_HEIGHT"]) / (group_count - 1)
    match_mean = out.groupby("PROVIDERID")["PLAYER_HEIGHT"].transform("mean")
    out["HEIGHT_VS_POSITION"] = out["PLAYER_HEIGHT"] - peer_mean.where(
        group_count > 1, match_mean
    )
    return out.drop(columns=["SEASON_KEY", "POSITION_GROUP"])


def build_feature_table(
    player_stats: pd.DataFrame,
    team_stats: pd.DataFrame,
    player_details: pd.DataFrame | None = None,
    train_through: int = 2025,
) -> pd.DataFrame:
    """Build the merged feature table (finals excluded, labels not yet attached).

    ``train_through`` controls which seasons inform the >5% NaN column drop so
    that no evaluation-season information leaks into feature selection.
    Columns listed in :data:`NATIVE_MISSING_FEATURES` are kept regardless and
    left missing for XGBoost to handle.
    """
    df = normalise_columns(player_stats)
    df = df[~df["ROUND_NAME"].astype(str).str.contains("Final", case=False, na=False)].copy()
    df["GAME_DATE"] = derive_game_date(df)
    df["FULL_NAME"] = add_full_name(df)
    df["PLAYER_CAPTAIN"] = (
        df["PLAYER_CAPTAIN"]
        .map({True: 1, False: 0, "True": 1, "False": 0, "TRUE": 1, "FALSE": 0})
        .astype(float)
    )

    seasons = pd.to_datetime(df["GAME_DATE"]).dt.year
    reference = df[seasons <= train_through]
    drop_cols = [
        column
        for column in nan_fraction_columns(reference)
        if column not in NATIVE_MISSING_FEATURES
    ]
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
    if player_details is not None:
        df = add_height_features(df, player_details)
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
        self.numeric_features_ = [
            col for col in numeric if col in FEATURE_LIST and col not in NATIVE_MISSING_FEATURES
        ]
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
