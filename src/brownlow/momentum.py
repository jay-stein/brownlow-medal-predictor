"""Fourth-quarter and late-game momentum features from the AFL scoring timeline.

The play-by-play extract (see ``afl_cfs``) records every scoring event with its
period, clock second, and running score. This module turns those events into:

- **match features** describing how the game finished: the late-third and
  fourth-quarter swings, comebacks and blown leads, longest goal runs, and
  whether the game was close late; and
- **player features** describing scoring when it mattered: fourth-quarter and
  late-third scoring, scoring in close late-game situations, share of the
  team's final-quarter scoring, and first/last goal of the game.

Match features are stored from the home team's perspective and flipped to the
player's team when attached, so a positive swing always means the player's
team gained ground.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import paths

PLAYBYPLAY_CSV = paths.DATA_DIR / "playbyplay_2012_2026.csv"

LATE_Q3_SECONDS = 1200
LAST10_SECONDS = 600
CLUTCH_MARGIN = 12
CLOSE_MARGIN = 6

PLAYER_COLUMN = "PLAYER_PLAYER_PLAYER_PLAYERID"

MATCH_FEATURE_COLUMNS = [
    "M_Q4_SWING",
    "M_LATE_SWING",
    "M_COMEBACK_HOME",
    "M_BLOWN_HOME",
    "M_RUN_Q4_HOME",
    "M_RUN_Q4_AWAY",
    "M_LEAD_Q4_HOME",
    "M_LEAD_Q4_AWAY",
    "M_CLOSE_Q4",
    "M_LAST10_CLOSE",
    "M_Q4_SCORE_HOME",
    "M_Q4_SCORE_AWAY",
]

PLAYER_FEATURE_COLUMNS = [
    "Q4_GOALS",
    "Q4_SCORES",
    "LATE_GOALS",
    "LATE_SCORES",
    "CLUTCH_SCORES",
    "TEAM_Q4_SHARE",
    "FIRST_GOAL",
    "LAST_GOAL",
]

MOMENTUM_FEATURES = [
    "TEAM_Q4_SWING",
    "TEAM_LATE_SWING",
    "TEAM_COMEBACK",
    "TEAM_BLOWN_LEAD",
    "TEAM_RUN_Q4",
    "TEAM_LEAD_Q4",
    "CLOSE_Q4",
    "LAST10_CLOSE",
    *PLAYER_FEATURE_COLUMNS,
]


def load_events(path: str | Path | None = None) -> pd.DataFrame | None:
    """Load the play-by-play extract, or ``None`` when it has not been scraped."""
    location = Path(path) if path is not None else PLAYBYPLAY_CSV
    if not location.exists():
        return None
    frame = pd.read_csv(location, dtype={"PROVIDERID": str, "PLAYER_ID": str})
    frame["PLAYER_ID"] = frame["PLAYER_ID"].fillna("")
    return frame


def _max_goal_run(period_events: pd.DataFrame, side: str) -> int:
    """Longest streak of consecutive goals for ``side`` within the events."""
    best = 0
    current = 0
    for event in period_events.itertuples(index=False):
        if event.SCORE_TYPE != "GOAL":
            continue
        if event.HOME_AWAY == side:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def match_features(events: pd.DataFrame) -> pd.DataFrame:
    """One row per match with late-game features from the home perspective."""
    rows: list[dict] = []
    for provider_id, group in events.groupby("PROVIDERID", sort=False):
        ordered = group.sort_values(["PERIOD", "PERIOD_SECONDS"], kind="stable")
        margin = (ordered["AGG_HOME"] - ordered["AGG_AWAY"]).to_numpy(dtype=int)
        period = ordered["PERIOD"].to_numpy()
        seconds = ordered["PERIOD_SECONDS"].to_numpy()
        final_margin = int(margin[-1])

        before_four = np.flatnonzero(period <= 3)
        margin_3qtr = int(margin[before_four[-1]]) if len(before_four) else 0
        late3 = np.flatnonzero((period == 3) & (seconds <= LATE_Q3_SECONDS))
        margin_late3 = int(margin[late3[-1]]) if len(late3) else margin_3qtr

        q4 = ordered[period == 4]
        q4_margin = (q4["AGG_HOME"] - q4["AGG_AWAY"]).to_numpy(dtype=int)
        q4_seconds = q4["PERIOD_SECONDS"].to_numpy(dtype=int)
        last10_close = int(
            np.any(np.abs(q4_margin[q4_seconds >= LAST10_SECONDS]) <= CLOSE_MARGIN)
        ) if len(q4_margin) else 0
        rows.append(
            {
                "PROVIDERID": provider_id,
                "M_Q4_SWING": final_margin - margin_3qtr,
                "M_LATE_SWING": final_margin - margin_late3,
                "M_COMEBACK_HOME": int(margin_late3 < 0 and final_margin > 0),
                "M_BLOWN_HOME": int(margin_late3 > 0 and final_margin < 0),
                "M_RUN_Q4_HOME": _max_goal_run(q4, "HOME"),
                "M_RUN_Q4_AWAY": _max_goal_run(q4, "AWAY"),
                "M_LEAD_Q4_HOME": int(q4_margin.max()) if len(q4_margin) else 0,
                "M_LEAD_Q4_AWAY": int(-q4_margin.min()) if len(q4_margin) else 0,
                "M_CLOSE_Q4": int(np.any(np.abs(q4_margin) <= CLOSE_MARGIN)),
                "M_LAST10_CLOSE": last10_close,
                "M_Q4_SCORE_HOME": int(
                    q4.loc[q4["HOME_AWAY"] == "HOME", "SCORE_VALUE"].sum()
                ),
                "M_Q4_SCORE_AWAY": int(
                    q4.loc[q4["HOME_AWAY"] == "AWAY", "SCORE_VALUE"].sum()
                ),
            }
        )
    return pd.DataFrame(rows, columns=["PROVIDERID", *MATCH_FEATURE_COLUMNS])


def player_features(events: pd.DataFrame) -> pd.DataFrame:
    """One row per scoring player-match with late-game scoring features."""
    scorers = events[events["PLAYER_ID"] != ""].copy()
    scorers["MARGIN"] = scorers["AGG_HOME"] - scorers["AGG_AWAY"]
    scorers["LATE_WINDOW"] = (scorers["PERIOD"] == 4) | (
        (scorers["PERIOD"] == 3) & (scorers["PERIOD_SECONDS"] >= LATE_Q3_SECONDS)
    )

    q4_by_team = (
        scorers[scorers["PERIOD"] == 4]
        .groupby(["PROVIDERID", "HOME_AWAY"])["SCORE_VALUE"]
        .sum()
    )

    goals = scorers[scorers["SCORE_TYPE"] == "GOAL"]
    first_goal = goals.sort_values(["PROVIDERID", "PERIOD", "PERIOD_SECONDS"]).groupby(
        "PROVIDERID"
    ).head(1)
    last_goal = goals.sort_values(["PROVIDERID", "PERIOD", "PERIOD_SECONDS"]).groupby(
        "PROVIDERID"
    ).tail(1)
    first_ids = set(zip(first_goal["PROVIDERID"], first_goal["PLAYER_ID"]))
    last_ids = set(zip(last_goal["PROVIDERID"], last_goal["PLAYER_ID"]))

    rows: list[dict] = []
    for (provider_id, player_id), group in scorers.groupby(
        ["PROVIDERID", "PLAYER_ID"], sort=False
    ):
        q4 = group[group["PERIOD"] == 4]
        late = group[group["LATE_WINDOW"]]
        clutch = late[late["MARGIN"].abs() <= CLUTCH_MARGIN]
        team_q4 = float(q4_by_team.get((provider_id, group["HOME_AWAY"].iloc[0]), 0.0))
        rows.append(
            {
                "PROVIDERID": provider_id,
                "PLAYER_ID": player_id,
                "Q4_GOALS": int((q4["SCORE_TYPE"] == "GOAL").sum()),
                "Q4_SCORES": len(q4),
                "LATE_GOALS": int((late["SCORE_TYPE"] == "GOAL").sum()),
                "LATE_SCORES": len(late),
                "CLUTCH_SCORES": len(clutch),
                "TEAM_Q4_SHARE": float(q4["SCORE_VALUE"].sum() / team_q4) if team_q4 else 0.0,
                "FIRST_GOAL": int((provider_id, player_id) in first_ids),
                "LAST_GOAL": int((provider_id, player_id) in last_ids),
            }
        )
    return pd.DataFrame(
        rows, columns=["PROVIDERID", "PLAYER_ID", *PLAYER_FEATURE_COLUMNS]
    )


def attach_momentum(table: pd.DataFrame, events: pd.DataFrame | None = None) -> pd.DataFrame:
    """Attach momentum features to the labelled table.

    Players in matches without play-by-play data receive NaN for every feature;
    players in matches with data who did not score receive zeros. The
    intermediate home-perspective columns are removed.
    """
    out = table.copy()
    out["PROVIDERID"] = out["PROVIDERID"].astype(str)
    if events is None:
        events = load_events()
    if events is None or events.empty:
        for column in MOMENTUM_FEATURES:
            out[column] = np.nan
        return out

    matches = match_features(events)
    players = player_features(events)
    out = out.merge(matches, on="PROVIDERID", how="left")
    player_key = out[PLAYER_COLUMN].astype(str)
    lookup = pd.DataFrame({"PROVIDERID": out["PROVIDERID"].to_numpy(), "PLAYER_ID": player_key})
    lookup = lookup.merge(players, on=["PROVIDERID", "PLAYER_ID"], how="left")

    present = out["PROVIDERID"].isin(set(matches["PROVIDERID"])).to_numpy()
    for column in PLAYER_FEATURE_COLUMNS:
        values = np.nan_to_num(lookup[column].to_numpy(dtype=float), nan=0.0)
        out[column] = np.where(present, values, np.nan)

    at_home = out["AT_HOME"].fillna(0).astype(int).to_numpy() == 1
    sign = np.where(at_home, 1.0, -1.0)
    out["TEAM_Q4_SWING"] = out["M_Q4_SWING"] * sign
    out["TEAM_LATE_SWING"] = out["M_LATE_SWING"] * sign
    out["TEAM_COMEBACK"] = np.where(at_home, out["M_COMEBACK_HOME"], out["M_BLOWN_HOME"])
    out["TEAM_BLOWN_LEAD"] = np.where(at_home, out["M_BLOWN_HOME"], out["M_COMEBACK_HOME"])
    out["TEAM_RUN_Q4"] = np.where(at_home, out["M_RUN_Q4_HOME"], out["M_RUN_Q4_AWAY"])
    out["TEAM_LEAD_Q4"] = np.where(at_home, out["M_LEAD_Q4_HOME"], out["M_LEAD_Q4_AWAY"])
    out["CLOSE_Q4"] = out["M_CLOSE_Q4"]
    out["LAST10_CLOSE"] = out["M_LAST10_CLOSE"]
    return out.drop(columns=MATCH_FEATURE_COLUMNS)
