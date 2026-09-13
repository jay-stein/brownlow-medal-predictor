"""AFL Coaches Association (AFLCA) Champion Player votes.

Each home-and-away match, both coaching panels award 5-4-3-2-1 votes; the
published page lists the combined total per player (0-10). These are human
judgements of the game's standout players, independent of the Brownlow
umpires, and are available from 2004 onwards.

This module scrapes the published leaderboards into a tidy table of
``(season, round, match, player, votes)`` rows. The site is data-light but
HTML-only; parsing uses regular expressions over stable class names.
"""

from __future__ import annotations

import re
import time
import urllib.request
from html import unescape as html_unescape

import pandas as pd

from . import naming

BASE = "https://aflcoaches.com.au/awards/the-aflca-champion-player-of-the-year-award/leaderboard"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36 brownlow-research"
)

CLUBS = {
    "CARL": "Carlton",
    "ESS": "Essendon",
    "COLL": "Collingwood",
    "MELB": "Melbourne",
    "PORT": "Port Adelaide",
    "GCFC": "Gold Coast SUNS",
    "NMFC": "North Melbourne",
    "ADEL": "Adelaide Crows",
    "RICH": "Richmond",
    "GEEL": "Geelong Cats",
    "WCE": "West Coast Eagles",
    "SYD": "Sydney Swans",
    "GWS": "GWS GIANTS",
    "STK": "St Kilda",
    "WB": "Western Bulldogs",
    "FRE": "Fremantle",
    "BL": "Brisbane Lions",
    "HAW": "Hawthorn",
}

# Published names that differ from the dataset's (mostly nicknames), keyed by
# ``(club_key, aflca_name_key)``; an empty club key applies to any club.
NAME_ALIASES: dict[tuple[str, str], str] = {
    ("", "NATHAN FYFE"): "NAT FYFE",
    ("", "JAMES BARTEL"): "JIMMY BARTEL",
    ("", "MATTHEW THOMAS"): "MATT THOMAS",
    ("SYDNEY SWANS", "JOSH KENNEDY"): "JOSH P KENNEDY",
    ("WEST COAST EAGLES", "JOSH KENNEDY"): "JOSH J KENNEDY",
}

PLAYER_COLUMN = "PLAYER_PLAYER_PLAYER_PLAYERID"

_SUFFIXES = {"JNR", "JR", "SR", "II", "III", "IV"}


def _canonical_name(value: str) -> str:
    """Spaced upper-case name without generational suffixes."""
    tokens = [token for token in naming.normalise_name(value).split() if token not in _SUFFIXES]
    return " ".join(tokens)

_DATE_HEADER = re.compile(
    r'<div class="col-12 text-center text-uppercase">\s*<strong>(.*?)</strong>', re.S
)
_BLOCK_START = re.compile(r'<div class="mb-3">')
_CLUB_ALT = re.compile(r'class="club_logo"[^>]*alt="([^"]+)"|alt="([^"]+)"[^>]*class="club_logo"')
_VOTE_ROW = re.compile(
    r'<div class="col-2 text-center">\s*<strong>(\d+)</strong>\s*</div>\s*'
    r'<div class="col-10">\s*(.*?)\s*<span[^>]*class="small font-weight-bold"[^>]*>'
    r"\(([A-Za-z]+)\)</span>",
    re.S,
)
_ROUND_HEADING = re.compile(r"<h[23][^>]*>\s*Round\s*([^<]+?)\s*</h[23]>", re.I)
_ROUND_URL = re.compile(r'href="([^"]*leaderboard/(\d{4})/(\d+))"')
_TAGS = re.compile(r"<[^>]+>")


def fetch(url: str, retries: int = 3, pause: float = 1.0) -> str:
    """Fetch a URL with a small retry/backoff loop."""
    if url.startswith("/"):
        url = "https://aflcoaches.com.au" + url
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read().decode("utf-8", errors="replace")
        except Exception as error:  # noqa: BLE001 - network scrape, retry any failure
            last_error = error
            time.sleep(pause * (attempt + 1))
    raise RuntimeError(f"failed to fetch {url}: {last_error}")


def round_links(season: int) -> list[tuple[int, str]]:
    """``(url identifier, url)`` pairs listed on a season's leaderboard page.

    Identifiers are chronological (the opening round sorts first), so callers
    can rely on their order rather than parsing the visible round labels.
    """
    html = fetch(f"{BASE}/{season}")
    links = sorted(
        {
            (int(identifier), url)
            for url, link_season, identifier in _ROUND_URL.findall(html)
            if int(link_season) == season
        }
    )
    return links


def parse_round_page(html: str, season: int) -> list[dict]:
    """Parse one round page into per-player vote rows."""
    date_marks = [(match.start(), match.group(1)) for match in _DATE_HEADER.finditer(html)]
    heading = _ROUND_HEADING.search(html)
    round_label = heading.group(1).strip() if heading else ""

    rows: list[dict] = []
    starts = [match.start() for match in _BLOCK_START.finditer(html)]
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(html)
        block = html[start:end]
        date_text = next((text for position, text in reversed(date_marks) if position < start), "")
        teams = [a or b for a, b in _CLUB_ALT.findall(block)][:2]
        if len(teams) < 2:
            continue
        for votes, player, club in _VOTE_ROW.findall(block):
            player = html_unescape(_TAGS.sub(" ", player))
            player = " ".join(player.split())
            if not player:
                continue
            rows.append(
                {
                    "season": season,
                    "round": round_label,
                    "date": " ".join(date_text.split()),
                    "home_team": teams[0],
                    "away_team": teams[1],
                    "player_name": player,
                    "club": club,
                    "votes": int(votes),
                }
            )
    return rows


def fetch_season(season: int, pause: float = 1.0) -> pd.DataFrame:
    """Scrape every listed round of a season."""
    rows: list[dict] = []
    links = round_links(season)
    for order, (identifier, url) in enumerate(links):
        html = fetch(url)
        parsed = parse_round_page(html, season)
        label = parsed[0]["round"] if parsed else ("OR" if order == 0 else f"R{order}")
        for row in parsed:
            row["round_link"] = label
            row["round_id"] = identifier
        rows.extend(parsed)
        print(f"  {season} {label}: {len(parsed)} rows")
        time.sleep(pause)
    return pd.DataFrame(rows)


def load_votes(path) -> pd.DataFrame:
    """Load scraped votes with normalised match and player keys."""
    frame = pd.read_csv(path)
    for column in ("player_name", "home_team", "away_team"):
        frame[column] = frame[column].map(lambda value: html_unescape(str(value)))
    frame["home_key"] = frame["home_team"].map(naming.compact_key)
    frame["away_key"] = frame["away_team"].map(naming.compact_key)
    frame["team_key"] = frame["club"].map(CLUBS).map(naming.compact_key)
    parsed = pd.to_datetime(
        frame["date"] + " " + frame["season"].astype(str),
        format="%a, %B %d %Y",
        errors="coerce",
    )
    frame["match_date"] = parsed
    frame["date_md"] = parsed.dt.strftime("%m-%d")
    frame["name_key"] = frame["player_name"].map(_canonical_name)
    return frame


def match_lookup(games: pd.DataFrame) -> pd.DataFrame:
    """Unique matches keyed by season, team and calendar date.

    A team plays at most one match per date, so ``(season, date, club)`` is a
    robust match key that does not depend on parsing the page's home/away
    logos (which are unreliable on some older pages).
    """
    matches = (
        games[["ROUND_YEAR", "TEAM_NAME", "GAME_DATE", "PROVIDERID"]]
        .drop_duplicates()
        .copy()
    )
    matches["team_key"] = matches["TEAM_NAME"].map(naming.compact_key)
    matches["date_md"] = pd.to_datetime(matches["GAME_DATE"]).dt.strftime("%m-%d")
    matches = matches.drop_duplicates(subset=["ROUND_YEAR", "date_md", "team_key"])
    return matches


# Common published-vs-dataset first-name variants.
NICKNAMES: dict[str, set[str]] = {
    "THOMAS": {"TOM", "TOMMY"},
    "TOM": {"THOMAS", "TOMMY"},
    "NICHOLAS": {"NICK", "NICKY"},
    "NICK": {"NICHOLAS", "NICKY"},
    "ALEXANDER": {"ALEX"},
    "ALEX": {"ALEXANDER"},
    "SEBASTIAN": {"SEB"},
    "SEB": {"SEBASTIAN"},
    "MATTHEW": {"MATT", "MATTY"},
    "MATT": {"MATTHEW", "MATTY"},
    "JAMES": {"JIMMY", "JIM"},
    "JIMMY": {"JAMES", "JIM"},
    "MICHAEL": {"MICK", "MIKE"},
    "MICK": {"MICHAEL", "MIKE"},
    "MIKE": {"MICHAEL", "MICK"},
    "DANIEL": {"DAN", "DANNY"},
    "DAN": {"DANIEL", "DANNY"},
    "BENJAMIN": {"BEN"},
    "BEN": {"BENJAMIN"},
    "JOSHUA": {"JOSH"},
    "JOSH": {"JOSHUA"},
    "SAMUEL": {"SAM"},
    "SAM": {"SAMUEL"},
    "JONATHAN": {"JON"},
    "JON": {"JONATHAN"},
    "CHRISTOPHER": {"CHRIS"},
    "CHRIS": {"CHRISTOPHER"},
    "LACHLAN": {"LACHIE"},
    "LACHIE": {"LACHLAN"},
    "NATHANIEL": {"NATHAN", "NATE"},
    "NATHAN": {"NATE"},
}


def _first_names_match(left: str, right: str) -> bool:
    if left == right:
        return True
    return right in NICKNAMES.get(left, set()) or left in NICKNAMES.get(right, set())


def _resolve_name(name_key: str, club_key: str, roster: list[tuple[str, str]]) -> str | None:
    exact = [player_id for player_id, key in roster if key == name_key]
    if len(exact) == 1:
        return exact[0]
    alias = NAME_ALIASES.get((club_key, name_key)) or NAME_ALIASES.get(("", name_key))
    if alias:
        aliased = [player_id for player_id, key in roster if key == alias]
        if len(aliased) == 1:
            return aliased[0]
    surname = name_key.split()[-1]
    candidates = [player_id for player_id, key in roster if key.endswith(" " + surname)]
    if len(candidates) == 1:
        return candidates[0]
    first = name_key.split()[0]
    candidates = [
        player_id
        for player_id, key in roster
        if key.endswith(" " + surname) and _first_names_match(first, key.split()[0])
    ]
    if len(candidates) == 1:
        return candidates[0]
    return None


def resolve_player_games(votes: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Attach ``PROVIDERID`` and dataset player ids to scraped vote rows.

    Matches are identified by ``(season, date, club)``, which is unique per
    round and independent of the AFLCA's round numbering and page markup.
    Unmatched dates fall back to the club's nearest match within ten days.
    """
    work = games.copy()
    work["team_key"] = work["TEAM_NAME"].map(naming.compact_key)
    work["date_md"] = pd.to_datetime(work["GAME_DATE"]).dt.strftime("%m-%d")
    roster_frame = work[["PROVIDERID", "team_key", PLAYER_COLUMN, "FULL_NAME"]].copy()
    roster_frame["roster_key"] = roster_frame["FULL_NAME"].map(_canonical_name)
    rosters: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for (provider, team), group in roster_frame.groupby(["PROVIDERID", "team_key"]):
        rosters[(provider, team)] = list(zip(group[PLAYER_COLUMN], group["roster_key"]))

    matches = match_lookup(work)
    joined = votes.merge(
        matches[["ROUND_YEAR", "date_md", "team_key", "PROVIDERID"]],
        how="left",
        left_on=["season", "date_md", "team_key"],
        right_on=["ROUND_YEAR", "date_md", "team_key"],
    )
    missing = joined["PROVIDERID"].isna()
    if missing.any():
        dated = work[["ROUND_YEAR", "team_key", "GAME_DATE", "PROVIDERID"]].drop_duplicates(
            subset=["ROUND_YEAR", "team_key", "GAME_DATE"]
        )
        dated["game_date"] = pd.to_datetime(dated["GAME_DATE"])
        for index in joined.index[missing]:
            row = joined.loc[index]
            candidates = dated[
                (dated["ROUND_YEAR"] == row["season"]) & (dated["team_key"] == row["team_key"])
            ]
            if candidates.empty or pd.isna(row["match_date"]):
                continue
            deltas = (candidates["game_date"] - row["match_date"]).abs()
            closest = deltas.idxmin()
            if deltas.min() <= pd.Timedelta(days=10):
                joined.loc[index, "PROVIDERID"] = candidates.loc[closest, "PROVIDERID"]

    resolved: list[str | None] = []
    for row in joined.itertuples(index=False):
        roster = rosters.get((row.PROVIDERID, row.team_key))
        if not roster:
            resolved.append(None)
            continue
        resolved.append(_resolve_name(row.name_key, row.team_key, roster))
    joined["player_id"] = resolved
    return joined


def attach_coaches_votes(
    table: pd.DataFrame,
    votes_path,
) -> tuple[pd.DataFrame, dict]:
    """Add ``COACH_VOTES`` and ``COACH_VOTES_SHARE`` to a player-game table.

    Players who did not receive votes from either panel get a genuine zero.
    The share is the player's votes divided by their team's votes in the same
    match. Returns the table plus a small coverage summary.
    """
    votes = load_votes(votes_path)
    resolved = resolve_player_games(votes, table)
    matched = resolved.dropna(subset=["PROVIDERID", "player_id"])
    grouped = (
        matched.groupby(["ROUND_YEAR", "PROVIDERID", "player_id"], as_index=False)
        .agg(votes=("votes", "sum"))
    )
    lookup = grouped.set_index(["ROUND_YEAR", "PROVIDERID", "player_id"])["votes"]

    out = table.copy()
    keys = list(zip(out["ROUND_YEAR"], out["PROVIDERID"], out[PLAYER_COLUMN]))
    out["COACH_VOTES"] = [float(lookup.get(key, 0.0)) for key in keys]

    team_totals = out.groupby(["PROVIDERID", "TEAM_NAME"])["COACH_VOTES"].transform("sum")
    share = out["COACH_VOTES"] / team_totals
    out["COACH_VOTES_SHARE"] = share.where(team_totals > 0)

    summary = {
        "rows": int(len(votes)),
        "matched_rows": int(len(matched)),
        "match_rate": float(len(matched) / len(votes)) if len(votes) else 0.0,
        "matched_matches": int(matched["PROVIDERID"].nunique()),
    }
    return out, summary
