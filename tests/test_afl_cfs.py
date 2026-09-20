from brownlow import afl_cfs


def _payload() -> dict:
    return {
        "score": {
            "homeTeamScore": {"totalScore": 12},
            "awayTeamScore": {"totalScore": 1},
            "scoreWorm": {
                "scoringEvents": [
                    {
                        "teamName": {
                            "teamAbbr": "SYD",
                            "teamName": "Sydney Swans",
                            "teamNickname": "Swans",
                        },
                        "teamId": "CD_T160",
                        "playerScore": {
                            "player": {
                                "playerId": "CD_I996765",
                                "playerName": {"givenName": "Tom", "surname": "Papley"},
                                "captain": False,
                            },
                            "scoreBreakdown": {"totalScore": 6, "goals": 1, "behinds": 0},
                        },
                        "periodNumber": 1,
                        "periodSeconds": 81,
                        "scoreType": "GOAL",
                        "homeOrAway": "AWAY",
                        "aggregateHomeScore": 0,
                        "aggregateAwayScore": 6,
                        "scoreValue": 6,
                    },
                    {
                        "teamName": {
                            "teamAbbr": "HAW",
                            "teamName": "Hawthorn",
                            "teamNickname": "Hawks",
                        },
                        "teamId": "CD_T80",
                        "playerScore": None,
                        "periodNumber": 4,
                        "periodSeconds": 1793,
                        "scoreType": "RUSHED",
                        "homeOrAway": "HOME",
                        "aggregateHomeScore": 12,
                        "aggregateAwayScore": 1,
                        "scoreValue": 1,
                    },
                ]
            },
        }
    }


def test_parse_scoring_events_flattens_scorer_and_clock():
    frame = afl_cfs.parse_scoring_events(_payload(), "CD_M20260140201", 2026)
    assert list(frame.columns) == afl_cfs.EVENT_COLUMNS
    assert len(frame) == 2
    goal = frame.iloc[0]
    assert goal["PLAYER_NAME"] == "Tom Papley"
    assert goal["SCORE_TYPE"] == "GOAL"
    assert goal["PERIOD"] == 1
    assert goal["PERIOD_SECONDS"] == 81
    assert goal["AGG_AWAY"] == 6
    rushed = frame.iloc[1]
    assert rushed["PLAYER_ID"] == ""
    assert rushed["PLAYER_NAME"] == ""
    assert rushed["SCORE_TYPE"] == "RUSHED"
    assert rushed["AGG_HOME"] == 12


def test_parse_scoring_events_handles_missing_worm():
    frame = afl_cfs.parse_scoring_events({"score": {}}, "CD_M20120140101", 2012)
    assert frame.empty
    assert list(frame.columns) == afl_cfs.EVENT_COLUMNS


def test_final_scores_reads_totals():
    assert afl_cfs.final_scores(_payload()) == (12, 1)


def test_cfs_error_carries_status():
    error = afl_cfs.CfsError("unauthorised", status=401)
    assert error.status == 401


def test_parse_scoring_events_types_are_stable():
    frame = afl_cfs.parse_scoring_events(_payload(), "CD_M20260140201", 2026)
    assert frame["SEASON"].dtype == "int64"
    assert frame["PERIOD"].dtype == "int64"
    assert frame["PERIOD_SECONDS"].dtype == "int64"
    assert frame["PLAYER_NAME"].map(lambda value: isinstance(value, str)).all()
