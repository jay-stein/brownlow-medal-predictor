"""Official AFL match-centre scoring events (CFS API, 2012+).

The public ``afl.com.au`` match pages are client-rendered and carry no
embedded data, but the feed behind them is open: an unauthenticated POST to
``/cfs/afl/WMCTok`` returns a token, and ``/cfs/afl/matchItem/{providerId}``
with the ``x-media-mis-token`` header (the same mechanism fitzRoy uses)
returns the official scoring timeline. Each event records the scorer, period,
exact clock second, score type and the running match score, which supports
fourth-quarter and late-game momentum features.

Coverage starts in 2012, so the events can be used in training rather than
only as a recent-season add-on.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import pandas as pd

TOKEN_URL = "https://api.afl.com.au/cfs/afl/WMCTok"
MATCH_ITEM_URL = "https://api.afl.com.au/cfs/afl/matchItem/{provider_id}"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36 brownlow-research"
)

EVENT_COLUMNS = [
    "SEASON",
    "PROVIDERID",
    "PERIOD",
    "PERIOD_SECONDS",
    "SCORE_TYPE",
    "SCORE_VALUE",
    "HOME_AWAY",
    "TEAM_ID",
    "TEAM_NAME",
    "PLAYER_ID",
    "PLAYER_NAME",
    "AGG_HOME",
    "AGG_AWAY",
]


class CfsError(RuntimeError):
    """CFS request failed; ``status`` carries the HTTP code when there was one."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def _request(url: str, method: str = "GET", token: str | None = None, timeout: int = 30):
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json, text/plain, */*",
        "Origin": "https://www.afl.com.au",
        "Referer": "https://www.afl.com.au/",
    }
    if token:
        headers["x-media-mis-token"] = token
    request = urllib.request.Request(url, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read() if error.fp else b""
    except Exception as error:
        raise CfsError(f"request failed for {url}: {error!r}") from error


def fetch_token() -> str:
    """Obtain a media token for the CFS endpoints."""
    status, body = _request(TOKEN_URL, method="POST")
    if status != 200:
        raise CfsError(f"WMCTok returned {status}", status=status)
    try:
        token = json.loads(body)["token"]
    except (json.JSONDecodeError, KeyError) as error:
        raise CfsError("WMCTok response did not contain a token") from error
    return str(token)


def fetch_match_item(provider_id: str, token: str) -> dict:
    """Fetch the raw match-centre payload for one match."""
    status, body = _request(MATCH_ITEM_URL.format(provider_id=provider_id), token=token)
    if status != 200:
        raise CfsError(f"matchItem {provider_id} returned {status}", status=status)
    try:
        return json.loads(body)
    except json.JSONDecodeError as error:
        raise CfsError(f"matchItem {provider_id} returned invalid JSON") from error


def _player_name(player: dict) -> str:
    name = player.get("playerName") or {}
    parts = [name.get("givenName"), name.get("surname")]
    return " ".join(str(part) for part in parts if part)


def parse_scoring_events(payload: dict, provider_id: str, season: int) -> pd.DataFrame:
    """Flatten ``score.scoreWorm.scoringEvents`` into tidy rows.

    Rushed behinds and other unattributed scores legitimately have no player;
    they are kept with empty player fields because they still move the score.
    """
    score = payload.get("score") or {}
    events = (score.get("scoreWorm") or {}).get("scoringEvents") or []
    rows: list[dict] = []
    for event in events:
        team = event.get("teamName") or {}
        player = ((event.get("playerScore") or {}).get("player")) or {}
        rows.append(
            {
                "SEASON": int(season),
                "PROVIDERID": str(provider_id),
                "PERIOD": int(event.get("periodNumber") or 0),
                "PERIOD_SECONDS": int(event.get("periodSeconds") or 0),
                "SCORE_TYPE": str(event.get("scoreType") or ""),
                "SCORE_VALUE": int(event.get("scoreValue") or 0),
                "HOME_AWAY": str(event.get("homeOrAway") or ""),
                "TEAM_ID": str(event.get("teamId") or ""),
                "TEAM_NAME": str(team.get("teamName") or ""),
                "PLAYER_ID": str(player.get("playerId") or ""),
                "PLAYER_NAME": _player_name(player),
                "AGG_HOME": int(event.get("aggregateHomeScore") or 0),
                "AGG_AWAY": int(event.get("aggregateAwayScore") or 0),
            }
        )
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def final_scores(payload: dict) -> tuple[int, int]:
    """Home and away totals from the payload's score summary."""
    score = payload.get("score") or {}
    home = (score.get("homeTeamScore") or {}).get("totalScore")
    away = (score.get("awayTeamScore") or {}).get("totalScore")
    return int(home or 0), int(away or 0)
