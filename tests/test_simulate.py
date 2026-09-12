import numpy as np
import pandas as pd

from brownlow import pl, simulate


def _season_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ROUND_YEAR": [2024] * 4,
            "PROVIDERID": ["m1"] * 4,
            "ROUND_ROUNDNUMBER": [0] * 4,
            "PLAYER_PLAYER_PLAYER_PLAYERID": ["p1", "p2", "p3", "p4"],
            "FULL_NAME": ["A", "B", "C", "D"],
            "TEAM_NAME": ["T"] * 4,
            "UTILITY": [3.0, 2.0, 1.0, 0.0],
            "BROWNLOW_VOTES_AUDITED": [3.0, 2.0, 1.0, 0.0],
        }
    )


def _multi_round_frame() -> pd.DataFrame:
    first = _season_frame()
    second = first.copy()
    second["PROVIDERID"] = "m2"
    second["ROUND_ROUNDNUMBER"] = 1
    return pd.concat([first, second], ignore_index=True)


def test_simulate_season_awards_six_votes_per_match_per_simulation():
    simulation = simulate.simulate_season(_season_frame(), tau=1.0, n_sims=200, seed=1)
    assert simulation.totals.shape == (4, 200)
    assert (simulation.totals.sum(axis=0) == 6).all()


def test_simulate_season_matches_pl_marginals_without_effect():
    simulation = simulate.simulate_season(_season_frame(), tau=1.0, n_sims=4000, seed=2)
    expected = pl.pl_marginals(np.array([3.0, 2.0, 1.0, 0.0]))[:, 3]
    observed = (simulation.totals == 3).mean(axis=1)
    np.testing.assert_allclose(observed, expected, atol=0.03)


def test_top5_probability_favours_strong_players():
    utilities = [3.0, 2.0, 1.0, 0.0, -1.0, -2.0]
    frame = pd.DataFrame(
        {
            "ROUND_YEAR": [2024] * 6,
            "PROVIDERID": ["m1"] * 6,
            "ROUND_ROUNDNUMBER": [0] * 6,
            "PLAYER_PLAYER_PLAYER_PLAYERID": [f"p{i}" for i in range(6)],
            "FULL_NAME": list("ABCDEF"),
            "TEAM_NAME": ["T"] * 6,
            "UTILITY": utilities,
            "BROWNLOW_VOTES_AUDITED": [3.0, 2.0, 1.0, 0.0, 0.0, 0.0],
        }
    )
    simulation = simulate.simulate_season(frame, tau=1.0, n_sims=500, seed=6)
    players = simulation.players.sort_values("sim_mean", ascending=False).reset_index(drop=True)
    assert players["p_top5"].between(0.0, 1.0).all()
    assert players.loc[0, "p_top5"] == 1.0
    assert players["p_top5"].sum() >= 5.0


def test_persistent_effect_widens_season_spread():
    without = simulate.simulate_season(
        _season_frame(), tau=1.0, effect_scale=0.0, n_sims=2000, seed=3
    )
    with_effect = simulate.simulate_season(
        _season_frame(), tau=1.0, effect_scale=1.0, n_sims=2000, seed=3
    )
    assert with_effect.totals.var(axis=1).mean() > without.totals.var(axis=1).mean()


def test_season_metrics_are_bounded():
    simulation = simulate.simulate_season(
        _season_frame(), tau=1.0, effect_scale=0.2, n_sims=500, seed=4
    )
    metrics = simulate.season_metrics(simulation, contender_count=2)
    assert 0.0 <= metrics["coverage_50"] <= 1.0
    assert 0.0 <= metrics["coverage_90"] <= 1.0
    assert 0.0 <= metrics["coverage_90_contenders"] <= 1.0
    assert metrics["mean_crps"] >= 0.0
    assert metrics["mean_crps_contenders"] >= 0.0


def test_calibrate_effect_scales_covers_the_grid():
    grid = simulate.calibrate_effect_scales(
        {2024: _season_frame()},
        {2024: 1.0},
        [0.0, 0.5],
        [2024],
        n_sims=100,
        seed=5,
        contender_count=2,
    )
    assert set(grid["effect_scale"]) == {0.0, 0.5}
    assert (grid["season"] == 2024).all()


def test_tracked_rounds_are_cumulative_and_match_parity():
    simulation = simulate.simulate_season(
        _multi_round_frame(),
        tau=1.0,
        n_sims=200,
        seed=8,
        track_rounds=True,
        path_count=20,
    )
    assert simulation.rounds == [0, 1]
    cumulative = simulation.cumulative_mean
    assert (np.diff(cumulative, axis=1) >= 0).all()
    assert np.allclose(cumulative[:, -1], simulation.totals.mean(axis=1))
    increment = simulation.increment_mean
    assert np.allclose(increment[:, 0], cumulative[:, 0])
    assert np.allclose(increment[:, 1], cumulative[:, 1] - cumulative[:, 0])
    assert simulation.path_totals.shape == (4, 2, 20)
    assert (np.diff(simulation.path_totals, axis=1) >= 0).all()


def test_forecast_export_payload_shapes():
    simulation = simulate.simulate_season(
        _multi_round_frame(),
        tau=1.0,
        n_sims=100,
        seed=9,
        track_rounds=True,
        path_count=10,
    )
    players = simulation.players.sort_values("sim_mean", ascending=False)
    payload = simulate.forecast_export(simulation, players, top=3, metadata={"season": 2024})
    assert payload["season"] == 2024
    assert [entry["label"] for entry in payload["rounds"]] == ["OR", "R1"]
    assert len(payload["players"]) == 3
    leader = payload["players"][0]
    assert {"cumMean", "cumMedian", "cumQ95", "incMean", "paths"} <= set(leader["rounds"])
    assert len(leader["rounds"]["paths"]) == 10
    assert len(leader["rounds"]["paths"][0]) == 2
    assert leader["pFirstOrJoint"] == max(player["pFirstOrJoint"] for player in payload["players"])
