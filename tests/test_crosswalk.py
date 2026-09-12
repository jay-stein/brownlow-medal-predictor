import pandas as pd

from brownlow import ingest


def _stats(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _votes(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _stats_row(cd_id: str, first: str, surname: str, team: str, opponent: str, date: str) -> dict:
    return {
        "player.player.player.playerId": cd_id,
        "utcStartTime": f"{date}T08:40:00.000+0000",
        "player.givenName": first,
        "player.surname": surname,
        "team.name": team,
        "home.team.name": team,
        "away.team.name": opponent,
    }


def _votes_row(vote_id: int, first: str, surname: str, team: str, opponent: str, date: str, votes: int) -> dict:
    return {
        "ID": vote_id,
        "Season": int(date[:4]),
        "Date": date,
        "First.name": first,
        "Surname": surname,
        "Playing.for": team,
        "Home.team": team,
        "Away.team": opponent,
        "Brownlow.Votes": votes,
    }


def test_crosswalk_matches_punctuation_variants_by_name():
    stats = _stats([_stats_row("CD_I1", "Cameron", "O'Shea", "Carlton", "Richmond", "2018-03-22")])
    votes = _votes([_votes_row(1, "Cameron", "OShea", "Carlton", "Richmond", "2018-03-22", 1)] + [])
    crosswalk = ingest.build_crosswalk(stats, votes)
    accepted = crosswalk[crosswalk["status"] == "accepted"]
    assert len(accepted) == 1
    assert accepted.iloc[0]["cd_id"] == "CD_I1"
    assert accepted.iloc[0]["match_method"] == "name"


def test_crosswalk_surname_fallback_for_nicknames():
    stats = _stats([_stats_row("CD_I1", "Joshua", "Draper", "Fremantle", "Brisbane Lions", "2024-03-17")])
    votes = _votes([_votes_row(1, "Josh", "Draper", "Fremantle", "Brisbane", "2024-03-17", 2)])
    crosswalk = ingest.build_crosswalk(stats, votes)
    accepted = crosswalk[crosswalk["status"] == "accepted"]
    assert len(accepted) == 1
    assert accepted.iloc[0]["cd_id"] == "CD_I1"
    assert accepted.iloc[0]["match_method"] == "surname_team"


def test_crosswalk_quarantines_unresolvable_names():
    stats = _stats(
        [
            _stats_row("CD_I1", "Sam", "Reid", "GWS GIANTS", "Sydney Swans", "2022-04-01"),
            _stats_row("CD_I2", "Xavier", "Reid", "GWS GIANTS", "Sydney Swans", "2022-04-01"),
        ]
    )
    votes = _votes([_votes_row(9, "Terry", "Reid", "GWS", "Sydney", "2022-04-01", 3)])
    crosswalk = ingest.build_crosswalk(stats, votes)
    assert crosswalk.iloc[0]["status"] == "ambiguous"


def test_crosswalk_reports_unmatched_players():
    stats = _stats([_stats_row("CD_I1", "Nick", "Daicos", "Collingwood", "Carlton", "2023-03-16")])
    votes = _votes([_votes_row(4, "Mystery", "Player", "Hawthorn", "Essendon", "2023-03-16", 3)])
    crosswalk = ingest.build_crosswalk(stats, votes)
    assert crosswalk.iloc[0]["status"] == "unmatched"


def test_crosswalk_does_not_accept_single_candidate_without_team_agreement():
    stats = _stats(
        [_stats_row("CD_I1", "Bailey", "Williams", "Western Bulldogs", "Richmond", "2024-03-17")]
    )
    votes = _votes([_votes_row(7, "Bailey", "Williams", "West Coast", "Richmond", "2024-03-17", 1)])
    crosswalk = ingest.build_crosswalk(stats, votes)
    assert crosswalk.iloc[0]["status"] == "ambiguous"


def test_crosswalk_strips_generational_suffixes():
    stats = _stats(
        [_stats_row("CD_I1", "Robert", "Hansen Jr", "North Melbourne", "Essendon", "2024-03-17")]
    )
    votes = _votes([_votes_row(3, "Robert", "Hansen", "North Melbourne", "Essendon", "2024-03-17", 2)])
    crosswalk = ingest.build_crosswalk(stats, votes)
    assert crosswalk.iloc[0]["status"] == "accepted"
    assert crosswalk.iloc[0]["match_method"] == "name"
