import pandas as pd

from brownlow import aflca

FIXTURE = """
<div class="row bg-primary text-white py-2">
  <div class="col-12 text-center text-uppercase"><strong>Sat, March 16</strong></div>
</div>
<div class="mb-3">
  <div class="row votes-by-match py-3">
    <div class="d-flex flex-column align-items-center px-2">
      <img src="x" class="club_logo" alt="Carlton" title="Carlton">
    </div>
    <div class="d-flex flex-column align-items-center font-weight-bold"><strong>vs</strong></div>
    <div class="d-flex flex-column align-items-center px-2">
      <img src="y" class="club_logo" alt="Richmond" title="Richmond">
    </div>
  </div>
  <div class="row match-watermark">
    <div class="col-12 py-2">
      <div class="row border-bottom pt-1 pb-1 div-hover">
        <div class="col-2 text-center"><strong>9</strong></div>
        <div class="col-10">Patrick Cripps <span class="small font-weight-bold">(CARL)</span></div>
      </div>
      <div class="row border-bottom pt-1 pb-1 div-hover">
        <div class="col-2 text-center"><strong>8</strong></div>
        <div class="col-10">Ryan O&#039;Keefe <span
          class="small font-weight-bold">(RICH)</span></div>
      </div>
    </div>
  </div>
</div>
"""


def test_parse_round_page_extracts_match_votes_and_unescapes():
    rows = aflca.parse_round_page(FIXTURE, 2024)
    assert len(rows) == 2
    assert rows[0]["player_name"] == "Patrick Cripps"
    assert rows[0]["club"] == "CARL"
    assert rows[0]["votes"] == 9
    assert rows[0]["home_team"] == "Carlton"
    assert rows[1]["player_name"] == "Ryan O'Keefe"


def test_name_resolution_handles_nicknames_and_suffixes():
    roster = [("p1", "NAT FYFE"), ("p2", "MALCOLM ROSAS"), ("p3", "SCOTT D THOMPSON")]
    assert aflca._resolve_name("NATHAN FYFE", "FREMANTLE", roster) == "p1"
    assert aflca._resolve_name("MALCOLM ROSAS JR", "SYDNEY SWANS", roster) == "p2"
    assert aflca._resolve_name("SCOTT THOMPSON", "NORTH MELBOURNE", roster) == "p3"
    assert aflca._resolve_name("TOM LOGAN", "PORT ADELAIDE", [("p4", "THOMAS LOGAN")]) == "p4"


def test_resolve_player_games_matches_by_team_and_date(tmp_path):
    games = pd.DataFrame(
        {
            "ROUND_YEAR": [2024, 2024, 2024, 2024],
            "PROVIDERID": ["m1", "m1", "m2", "m2"],
            "TEAM_NAME": ["Carlton", "Richmond", "Carlton", "Richmond"],
            "HOME_TEAM_NAME": ["Carlton", "Carlton", "Richmond", "Richmond"],
            "AWAY_TEAM_NAME": ["Richmond", "Richmond", "Carlton", "Carlton"],
            "GAME_DATE": ["2024-03-16", "2024-03-16", "2024-06-01", "2024-06-01"],
            "PLAYER_PLAYER_PLAYER_PLAYERID": ["p1", "p2", "p1", "p2"],
            "FULL_NAME": ["PATRICK CRIPPS", "DUSTIN MARTIN", "PATRICK CRIPPS", "DUSTIN MARTIN"],
        }
    )
    raw = pd.DataFrame(
        {
            "season": [2024, 2024],
            "round": [1, 1],
            "date": ["Sat, March 16", "Sat, March 16"],
            "home_team": ["Carlton", "Carlton"],
            "away_team": ["Richmond", "Richmond"],
            "player_name": ["Patrick Cripps", "Dustin Martin"],
            "club": ["CARL", "RICH"],
            "votes": [9, 8],
        }
    )
    path = tmp_path / "votes.csv"
    raw.to_csv(path, index=False)
    resolved = aflca.resolve_player_games(aflca.load_votes(path), games)
    assert list(resolved["PROVIDERID"]) == ["m1", "m1"]
    assert list(resolved["player_id"]) == ["p1", "p2"]


def test_attach_coaches_votes_fills_zeros_and_shares(tmp_path):
    games = pd.DataFrame(
        {
            "ROUND_YEAR": [2024, 2024],
            "PROVIDERID": ["m1", "m1"],
            "TEAM_NAME": ["Carlton", "Carlton"],
            "HOME_TEAM_NAME": ["Carlton", "Carlton"],
            "AWAY_TEAM_NAME": ["Richmond", "Richmond"],
            "GAME_DATE": ["2024-03-16", "2024-03-16"],
            "PLAYER_PLAYER_PLAYER_PLAYERID": ["p1", "p2"],
            "FULL_NAME": ["PATRICK CRIPPS", "SAM DOCERTY"],
        }
    )
    raw = pd.DataFrame(
        {
            "season": [2024],
            "round": [1],
            "date": ["Sat, March 16"],
            "home_team": ["Carlton"],
            "away_team": ["Richmond"],
            "player_name": ["Patrick Cripps"],
            "club": ["CARL"],
            "votes": [9],
        }
    )
    path = tmp_path / "votes.csv"
    raw.to_csv(path, index=False)
    table, summary = aflca.attach_coaches_votes(games, path)
    assert summary["matched_rows"] == 1
    assert table.loc[table["FULL_NAME"] == "PATRICK CRIPPS", "COACH_VOTES"].iloc[0] == 9
    assert table.loc[table["FULL_NAME"] == "SAM DOCERTY", "COACH_VOTES"].iloc[0] == 0
    assert table["COACH_VOTES_SHARE"].iloc[0] == 1.0
