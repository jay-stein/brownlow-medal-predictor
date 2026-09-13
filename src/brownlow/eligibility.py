"""Award-stage eligibility from suspension records.

The Brownlow Medal is awarded to the fairest and best player: any player
suspended during the home-and-away season remains in the count and keeps their
votes, but cannot win the medal (the next eligible player is awarded it, and
ineligible players are excluded from joint-first calculations).

Two committed sources are used:

- ``data/eligibility/ineligible_brownlow.csv`` - ineligible leading
  vote-getters for 2012-2025, compiled from the Wikipedia Brownlow Medal
  articles, which mark such players with an asterisk and cite the suspension.
- ``data/eligibility/suspensions_<season>.csv`` - full in-season suspension
  lists for the current season from a published MRO/tribunal tracker. These
  files are appended, never overwritten, so a count can be reproduced.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import paths

ELIGIBILITY_DIR = paths.DATA_DIR / "eligibility"
PLAYER_COLUMN = "PLAYER_PLAYER_PLAYER_PLAYERID"


def load_ineligible(season: int, directory: Path | None = None) -> pd.DataFrame:
    """Ineligible players for a season, from the curated committed files."""
    directory = directory or ELIGIBILITY_DIR
    frames: list[pd.DataFrame] = []
    historical = directory / "ineligible_brownlow.csv"
    if historical.exists():
        table = pd.read_csv(historical)
        frames.append(table[table["season"] == season])
    seasonal = directory / f"suspensions_{season}.csv"
    if seasonal.exists():
        frames.append(pd.read_csv(seasonal))
    if not frames:
        return pd.DataFrame(columns=["season", "player_id", "player_name"])
    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined["season"] == season]
    return combined.drop_duplicates(subset=["player_id"]).reset_index(drop=True)


def ineligible_ids(season: int, directory: Path | None = None) -> set[str]:
    """Player ids ineligible to win the medal in a season."""
    table = load_ineligible(season, directory=directory)
    if table.empty:
        return set()
    return {str(value) for value in table["player_id"]}
