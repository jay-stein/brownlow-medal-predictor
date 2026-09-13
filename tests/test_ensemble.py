import numpy as np
import pandas as pd

from brownlow import ensemble

PLAYER_COLUMN = "PLAYER_PLAYER_PLAYER_PLAYERID"
KEY_COLUMNS = ensemble.KEY_COLUMNS


def _member(utilities: list[float]) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "PROVIDERID": ["m1"] * 4,
            "ROUND_YEAR": [2024] * 4,
            PLAYER_COLUMN: ["p1", "p2", "p3", "p4"],
            "FULL_NAME": ["A", "B", "C", "D"],
            "TEAM_NAME": ["T"] * 4,
            "ROUND_ROUNDNUMBER": [0] * 4,
            "GAME_DATE": ["2024-03-16"] * 4,
            "LABEL_STATUS": ["voted"] * 4,
            "BROWNLOW_VOTES_AUDITED": [3.0, 2.0, 1.0, 0.0],
        }
    )
    frame["UTILITY"] = utilities
    return frame


def test_weight_candidates_form_a_simplex():
    candidates = ensemble.weight_candidates(["a", "b", "c"])
    assert len(candidates) == 15
    for weights in candidates:
        assert abs(sum(weights.values()) - 1.0) < 1e-9


def test_combine_scores_z_normalises_within_matches():
    first = _member([3.0, 2.0, 1.0, 0.0])
    second = _member([0.0, 1.0, 2.0, 3.0])
    combined = ensemble.combine_scores({"a": first, "b": second}, {"a": 1.0, "b": 0.0})
    grouped = combined.groupby("PROVIDERID")["UTILITY"]
    assert np.allclose(grouped.mean(), 0.0, atol=1e-9)
    assert np.allclose(grouped.std(ddof=0), 1.0, atol=1e-9)

    blended = ensemble.combine_scores({"a": first, "b": second}, {"a": 0.5, "b": 0.5})
    assert np.allclose(blended["UTILITY"], 0.0, atol=1e-9)

    a_only = ensemble.combine_scores({"a": first, "b": second}, {"a": 1.0, "b": 0.0})
    order_a = np.argsort(-a_only["UTILITY"].to_numpy())
    assert list(order_a) == [0, 1, 2, 3]


def test_rolling_ensemble_uses_only_earlier_seasons():
    def row(weights, season, crps, nll):
        return {
            "weights": weights,
            "w_a": 1.0 if weights == "a" else 0.0,
            "w_b": 0.0 if weights == "a" else 1.0,
            "tau": 0.7,
            "scale": 0.4,
            "season": season,
            "match_nll": nll,
            "mean_crps_contenders": crps,
            "mean_crps": crps,
            "coverage_50_contenders": 0.5,
            "coverage_90_contenders": 0.9,
            "width_50_contenders": 5.0,
            "width_90_contenders": 12.0,
            "favorite_won": True,
            "winner_probability": 0.3,
            "winner_mean_rank": 1,
            "n_players": 10,
        }

    grid = pd.DataFrame(
        [
            row("a", 2020, 3.0, 5.5),
            row("b", 2020, 2.5, 5.4),
            row("a", 2021, 3.5, 5.8),
            row("b", 2021, 2.0, 5.2),
        ]
    )
    table = ensemble.rolling_ensemble(grid, report_seasons=[2021], min_evidence=1)
    assert list(table["season"]) == [2021]
    # Member b wins the 2020 evidence and is applied to 2021.
    assert table.iloc[0]["weights_selected"] == "b"
    assert table.iloc[0]["mean_crps_contenders"] == 2.0
