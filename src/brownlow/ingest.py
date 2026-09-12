"""Raw data loading, player-id crosswalk and audited labels.

The Champion Data stats file and the AFL Tables vote file use different player
id namespaces. The crosswalk is built in two layers:

1. exact compact full-name + game date, with team agreement breaking ties;
2. surname + game date + team, with first-name similarity breaking ties
   (covers nicknames such as ``Josh``/``Joshua`` and optional middle initials).

Unmatched rows are quarantined rather than silently labelled as zero votes,
and a match only establishes genuine zeros if all three vote recipients map to
stats rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import paths
from .features import add_full_name, derive_game_date, normalise_columns
from .naming import compact_key, normalise_team, strip_name_suffix

VOTE_COL = "BROWNLOW_VOTES"
CD_ID_COL = "PLAYER_PLAYER_PLAYER_PLAYERID"

LABEL_VOTED = "voted"
LABEL_ZERO = "zero"
LABEL_UNRESOLVED = "unresolved"

ACCEPTED = "accepted"
AMBIGUOUS = "ambiguous"
UNMATCHED = "unmatched"

METHOD_NAME = "name"
METHOD_SURNAME = "surname_team"
METHOD_NONE = "none"


def load_player_stats(path=None) -> pd.DataFrame:
    return pd.read_csv(path or paths.PLAYER_STATS_CSV, index_col=0)


def load_team_stats(path=None) -> pd.DataFrame:
    return pd.read_csv(path or paths.TEAM_STATS_CSV, index_col=0)


def load_brownlow_votes(path=None) -> pd.DataFrame:
    return pd.read_csv(path or paths.BROWNLOW_VOTES_CSV)


def load_player_details(path=None) -> pd.DataFrame:
    return pd.read_csv(path or paths.PLAYER_DETAILS_CSV)


def _match_key(home_names: pd.Series, away_names: pd.Series) -> pd.Series:
    keys = [
        "|".join(sorted([normalise_team(home), normalise_team(away)]))
        for home, away in zip(home_names, away_names)
    ]
    return pd.Series(keys, index=home_names.index)


def _fixture_key(match_key: pd.Series, game_date: pd.Series) -> pd.Series:
    return match_key.astype(str) + "@" + game_date.astype(str)


def _prepare_stats(player_stats: pd.DataFrame) -> pd.DataFrame:
    df = normalise_columns(player_stats)
    return pd.DataFrame(
        {
            "cd_id": df[CD_ID_COL].astype("string"),
            "game_date": derive_game_date(df),
            "name_key": add_full_name(df).map(compact_key).map(strip_name_suffix),
            "first_key": df["PLAYER_GIVENNAME"].fillna("").map(compact_key),
            "surname_key": df["PLAYER_SURNAME"].fillna("").map(compact_key).map(strip_name_suffix),
            "team_key": df["TEAM_NAME"].map(normalise_team),
            "match_key": _match_key(df["HOME_TEAM_NAME"], df["AWAY_TEAM_NAME"]),
        }
    )


def _prepare_votes(votes: pd.DataFrame) -> pd.DataFrame:
    df = normalise_columns(votes)
    season = df["SEASON"]
    if isinstance(season, pd.DataFrame):
        season = season.iloc[:, 0]
    return pd.DataFrame(
        {
            "vote_id": pd.to_numeric(df["ID"], errors="coerce").astype("Int64"),
            "season": pd.to_numeric(season, errors="coerce").astype("Int64"),
            "game_date": pd.to_datetime(df["DATE"], errors="coerce").dt.date,
            "name_key": [
                strip_name_suffix(compact_key(f"{first} {last}"))
                for first, last in zip(df["FIRST_NAME"].fillna(""), df["SURNAME"].fillna(""))
            ],
            "first_key": df["FIRST_NAME"].fillna("").map(compact_key),
            "surname_key": df["SURNAME"].fillna("").map(compact_key).map(strip_name_suffix),
            "player_name": (df["FIRST_NAME"].astype(str) + " " + df["SURNAME"].astype(str)).str.strip(),
            "team_key": df["PLAYING_FOR"].map(normalise_team),
            "votes": pd.to_numeric(df[VOTE_COL], errors="coerce"),
            "match_key": _match_key(df["HOME_TEAM"], df["AWAY_TEAM"]),
        }
    )


def _first_name_score(vote_first: str, cd_first: str) -> float:
    """Similarity between first names: exact (2), prefix (1), otherwise 0."""
    if not vote_first or not cd_first:
        return 0.0
    if vote_first == cd_first:
        return 2.0
    short, long = sorted([vote_first, cd_first], key=len)
    if len(short) >= 3 and long.startswith(short):
        return 1.0
    if vote_first[0] == cd_first[0]:
        return 0.5
    return 0.0


def _layer1_selection(vote_id: int, candidates: pd.DataFrame) -> dict | None:
    """Accept only when team agreement identifies a single best candidate."""
    ordered = candidates.sort_values(
        ["n_team_agree", "n_games"], ascending=False
    ).reset_index(drop=True)
    top = ordered.iloc[0]
    if top["n_team_agree"] <= 0:
        return None
    if len(ordered) > 1:
        second = ordered.iloc[1]
        strictly_better = top["n_team_agree"] > second["n_team_agree"] or (
            top["n_team_agree"] == second["n_team_agree"] and top["n_games"] > second["n_games"]
        )
        if not strictly_better:
            return None
    return {
        "vote_id": vote_id,
        "cd_id": top["cd_id"],
        "n_games": int(top["n_games"]),
        "n_team_agree": int(top["n_team_agree"]),
        "status": ACCEPTED,
        "match_method": METHOD_NAME,
    }


def _record_from_row(vote_id: int, row: pd.Series, status: str, method: str) -> dict:
    return {
        "vote_id": vote_id,
        "cd_id": row["cd_id"],
        "n_games": int(row["n_games"]),
        "n_team_agree": int(row["n_team_agree"]),
        "status": status,
        "match_method": method,
    }


def _select_surname_candidate(vote_id: int, candidates: pd.DataFrame) -> dict:
    rows = []
    for cd_id, group in candidates.groupby("cd_id"):
        score = max(
            _first_name_score(vote_first, cd_first)
            for vote_first, cd_first in zip(group["first_key_vote"], group["first_key_cd"])
        )
        rows.append({"cd_id": cd_id, "score": score, "n_games": group["game_date"].nunique()})
    table = pd.DataFrame(rows).sort_values(["score", "n_games"], ascending=False).reset_index(drop=True)
    accepted = len(table) == 1 or table.iloc[0]["score"] > table.iloc[1]["score"]
    top = table.iloc[0]
    return {
        "vote_id": vote_id,
        "cd_id": top["cd_id"],
        "n_games": int(top["n_games"]),
        "n_team_agree": int(top["n_games"]),
        "status": ACCEPTED if accepted else AMBIGUOUS,
        "match_method": METHOD_SURNAME,
    }


def build_crosswalk(
    player_stats: pd.DataFrame | None = None,
    votes: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Map AFL Tables player ids to Champion Data player ids with an audit trail."""
    stats = _prepare_stats(player_stats if player_stats is not None else load_player_stats())
    vote_rows = _prepare_votes(votes if votes is not None else load_brownlow_votes())
    vote_ids = sorted(vote_rows["vote_id"].dropna().unique())

    stats_lookup = stats.dropna(subset=["cd_id"]).drop_duplicates()
    records: list[dict] = []
    resolved: set[int] = set()

    pairs = vote_rows[["vote_id", "name_key", "game_date", "team_key"]].merge(
        stats_lookup[["cd_id", "name_key", "game_date", "team_key"]],
        on=["name_key", "game_date"],
        how="inner",
        suffixes=("_vote", "_cd"),
    )
    pairs["team_agree"] = (
        pairs["team_key_vote"].notna() & (pairs["team_key_vote"] == pairs["team_key_cd"])
    ).astype(int)
    aggregated = pairs.groupby(["vote_id", "cd_id"], as_index=False).agg(
        n_games=("game_date", "nunique"),
        n_team_agree=("team_agree", "sum"),
    )
    fallback: dict[int, pd.Series] = {}
    for vote_id, candidates in aggregated.groupby("vote_id", sort=False):
        vid = int(vote_id)
        record = _layer1_selection(vid, candidates)
        if record is not None:
            records.append(record)
            resolved.add(vid)
        else:
            fallback[vid] = candidates.sort_values(
                ["n_team_agree", "n_games"], ascending=False
            ).iloc[0]

    pending = sorted({int(v) for v in vote_ids} - resolved)
    surname_resolved: set[int] = set()
    if pending:
        unresolved = vote_rows[vote_rows["vote_id"].isin(pending)]
        pairs_surname = unresolved[
            ["vote_id", "first_key", "surname_key", "game_date", "team_key"]
        ].merge(
            stats_lookup[["cd_id", "first_key", "surname_key", "game_date", "team_key"]],
            on=["surname_key", "game_date", "team_key"],
            how="inner",
            suffixes=("_vote", "_cd"),
        )
        for vote_id, candidates in pairs_surname.groupby("vote_id", sort=False):
            vid = int(vote_id)
            records.append(_select_surname_candidate(vid, candidates))
            surname_resolved.add(vid)

    for vid in pending:
        if vid in surname_resolved:
            continue
        if vid in fallback:
            records.append(_record_from_row(vid, fallback[vid], AMBIGUOUS, METHOD_NAME))
        else:
            records.append(
                {
                    "vote_id": vid,
                    "cd_id": pd.NA,
                    "n_games": 0,
                    "n_team_agree": 0,
                    "status": UNMATCHED,
                    "match_method": METHOD_NONE,
                }
            )

    if records:
        crosswalk = pd.DataFrame(records)
    else:
        crosswalk = pd.DataFrame(
            columns=["vote_id", "cd_id", "n_games", "n_team_agree", "status", "match_method"]
        )

    # A Champion Data identity can only belong to one AFL Tables player: keep the
    # identity with the strongest team agreement and quarantine the others.
    accepted_mask = crosswalk["status"] == ACCEPTED
    for _, group in crosswalk[accepted_mask].groupby("cd_id"):
        if group["vote_id"].nunique() <= 1:
            continue
        ordered = group.sort_values(["n_team_agree", "n_games"], ascending=False)
        if (
            len(ordered) > 1
            and ordered.iloc[0]["n_team_agree"] > ordered.iloc[1]["n_team_agree"]
        ):
            keep = ordered.index[0]
            crosswalk.loc[group.index.difference([keep]), "status"] = AMBIGUOUS
        else:
            crosswalk.loc[group.index, "status"] = AMBIGUOUS

    identities = (
        vote_rows.dropna(subset=["vote_id"])
        .groupby("vote_id", as_index=False)
        .agg(
            player_name=("player_name", "first"),
            team_key=("team_key", "first"),
            season=("season", "max"),
        )
    )
    return crosswalk.merge(identities, on="vote_id", how="left")


def attach_labels(
    feature_table: pd.DataFrame,
    votes: pd.DataFrame | None = None,
    crosswalk: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Attach audited vote labels with three states.

    ``voted``       - matched to a resolved match record and polled votes
    ``zero``        - matched to a resolved match record and polled no votes
    ``unresolved``  - identity join failed, or the match's recipients did not
                      all map to stats rows; quarantined from training
    """
    vote_rows = _prepare_votes(votes if votes is not None else load_brownlow_votes())
    crosswalk = crosswalk if crosswalk is not None else build_crosswalk(votes=votes)

    accepted = crosswalk.loc[crosswalk["status"] == ACCEPTED, ["vote_id", "cd_id"]]
    rows = vote_rows.merge(accepted, on="vote_id", how="inner")
    rows["fixture_key"] = _fixture_key(rows["match_key"], rows["game_date"])

    stats_keys = feature_table[
        [CD_ID_COL, "GAME_DATE", "PROVIDERID", "HOME_TEAM_NAME", "AWAY_TEAM_NAME"]
    ].copy()
    stats_keys["cd_id"] = stats_keys[CD_ID_COL].astype("string")
    stats_keys["match_key"] = _match_key(stats_keys["HOME_TEAM_NAME"], stats_keys["AWAY_TEAM_NAME"])
    stats_keys["fixture_key"] = _fixture_key(stats_keys["match_key"], stats_keys["GAME_DATE"])
    stats_keys = stats_keys.drop_duplicates(subset=["cd_id", "fixture_key"])

    rows = rows.merge(
        stats_keys[["cd_id", "fixture_key", "PROVIDERID"]],
        on=["cd_id", "fixture_key"],
        how="left",
    )

    recipients = rows[rows["votes"] > 0]
    complete_by_fixture = recipients.groupby("fixture_key")["PROVIDERID"].apply(
        lambda s: bool(len(s) == 3 and s.notna().all() and s.dropna().nunique() == 1)
    )

    labelled = feature_table.copy()
    labelled["cd_id"] = labelled[CD_ID_COL].astype("string")
    labelled["match_key"] = _match_key(labelled["HOME_TEAM_NAME"], labelled["AWAY_TEAM_NAME"])
    labelled["fixture_key"] = _fixture_key(labelled["match_key"], labelled["GAME_DATE"])

    lookup = (
        rows[["cd_id", "fixture_key", "votes"]]
        .dropna(subset=["votes"])
        .drop_duplicates(subset=["cd_id", "fixture_key"])
        .rename(columns={"votes": "vote_value"})
    )
    labelled = labelled.merge(lookup, on=["cd_id", "fixture_key"], how="left")
    labelled["match_complete"] = (
        labelled["fixture_key"].map(complete_by_fixture).fillna(False).astype(bool)
    )

    labelled["LABEL_STATUS"] = np.select(
        [
            labelled["vote_value"].isna(),
            ~labelled["match_complete"],
            labelled["vote_value"] > 0,
        ],
        [LABEL_UNRESOLVED, LABEL_UNRESOLVED, LABEL_VOTED],
        default=LABEL_ZERO,
    )
    labelled["BROWNLOW_VOTES_AUDITED"] = labelled["vote_value"].where(
        labelled["LABEL_STATUS"] != LABEL_UNRESOLVED
    )
    labelled = labelled.drop(columns=["cd_id", "match_key", "fixture_key", "vote_value"])

    audit = summarise_audit(labelled, crosswalk)
    return labelled, audit


def summarise_audit(labelled: pd.DataFrame, crosswalk: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Per-season label coverage plus crosswalk outcome tables."""
    status_counts = labelled.groupby(["ROUND_YEAR", "LABEL_STATUS"]).size().unstack(fill_value=0)
    for column in (LABEL_VOTED, LABEL_ZERO, LABEL_UNRESOLVED):
        if column not in status_counts:
            status_counts[column] = 0
    status_counts = status_counts[[LABEL_VOTED, LABEL_ZERO, LABEL_UNRESOLVED]]

    summary = pd.concat(
        [
            labelled.groupby("ROUND_YEAR")["PROVIDERID"].nunique().rename("matches"),
            labelled.groupby("ROUND_YEAR").size().rename("player_games"),
            status_counts,
        ],
        axis=1,
    ).reset_index()

    crosswalk_summary = (
        crosswalk.groupby(["status", "match_method"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["status", "count"], ascending=[True, False])
    )
    unmatched = (
        crosswalk.loc[
            crosswalk["status"] == UNMATCHED, ["vote_id", "player_name", "team_key", "season"]
        ]
        .sort_values(["season", "player_name"])
    )
    ambiguous = (
        crosswalk.loc[
            crosswalk["status"] == AMBIGUOUS,
            ["vote_id", "cd_id", "n_games", "n_team_agree", "player_name", "team_key", "season"],
        ]
        .sort_values(["season", "player_name"])
    )
    return {
        "by_season": summary,
        "crosswalk_summary": crosswalk_summary,
        "unmatched": unmatched,
        "ambiguous": ambiguous,
    }
