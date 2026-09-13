import numpy as np
import pandas as pd
import pytest

from brownlow import evaluate, pl, simulate


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


def test_integrated_match_log_loss_matches_exact_allocation_loss_without_effects():
    frame = _season_frame()
    exact = evaluate.allocation_log_loss(evaluate.build_matches(frame), 1.0)
    integrated = simulate.integrated_match_log_loss(frame, 1.0, effect_scale=0.0, n_draws=8)
    assert integrated == pytest.approx(exact)


def test_integrated_match_log_loss_increases_under_large_effects():
    frame = _season_frame()
    sharp = simulate.integrated_match_log_loss(frame, 0.2, effect_scale=0.0, n_draws=512)
    wide = simulate.integrated_match_log_loss(frame, 0.2, effect_scale=2.0, n_draws=512)
    assert wide > sharp


def test_season_metrics_reports_award_outcomes():
    simulation = simulate.simulate_season(
        _season_frame(), tau=1.0, effect_scale=0.2, n_sims=500, seed=4
    )
    metrics = simulate.season_metrics(simulation, contender_count=2)
    assert metrics["favorite_won"] is True
    assert 0.0 <= metrics["favorite_probability"] <= 1.0
    assert 0.0 <= metrics["winner_probability"] <= 1.0
    assert metrics["winner_mean_rank"] == 1


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
    probabilities = simulation.round_p1 + simulation.round_p2 + simulation.round_p3
    assert (probabilities >= 0).all()
    assert (probabilities <= 1.0 + 1e-12).all()
    assert simulation.round_p3.shape == (4, 2)


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
    assert {"cumMean", "cumMedian", "cumQ95", "incMean", "p1", "p2", "p3", "paths"} <= set(
        leader["rounds"]
    )
    assert len(leader["rounds"]["paths"]) == 10
    assert len(leader["rounds"]["paths"][0]) == 2
    assert len(leader["rounds"]["p3"]) == 2
    assert leader["pFirstOrJoint"] == max(player["pFirstOrJoint"] for player in payload["players"])
    assert len(payload["teams"]) == 1
    assert payload["matches"] == []


def test_team_round_tracking_sums_player_totals():
    simulation = simulate.simulate_season(
        _multi_round_frame(), tau=1.0, n_sims=200, seed=8, track_rounds=True
    )
    assert simulation.team_names == ["T"]
    assert simulation.team_cumulative_quantiles.shape == (1, 2, 5)
    np.testing.assert_allclose(
        simulation.team_cumulative_mean[0, -1], simulation.totals.sum(axis=0).mean()
    )
    assert simulation.team_cumulative_mean[0, 0] == pytest.approx(6.0)
    np.testing.assert_allclose(simulation.team_increment_mean[0], [6.0, 6.0])
    assert simulation.team_increment_mean.shape == (1, 2)
    assert simulation.team_path_totals.shape[0] == 1


def test_team_increment_probabilities_form_a_distribution():
    simulation = simulate.simulate_season(
        _multi_round_frame(), tau=1.0, n_sims=200, seed=8, track_rounds=True
    )
    probs = simulation.team_increment_probs
    assert probs.shape == (1, 2, 7)
    np.testing.assert_allclose(probs.sum(axis=2), 1.0, atol=1e-9)
    np.testing.assert_allclose(
        probs[0] @ np.arange(7), simulation.team_increment_mean[0], atol=1e-6
    )


def test_forecast_export_includes_team_rounds():
    simulation = simulate.simulate_season(
        _multi_round_frame(),
        tau=1.0,
        n_sims=200,
        seed=8,
        track_rounds=True,
        track_matches=True,
        path_count=10,
    )
    players = simulation.players.sort_values("sim_mean", ascending=False)
    payload = simulate.forecast_export(simulation, players, top=4, metadata={"season": 2024})
    team = payload["teams"][0]
    assert team["rounds"] is not None
    assert len(team["rounds"]["cumMean"]) == len(payload["rounds"])
    assert len(team["rounds"]["incMean"]) == len(payload["rounds"])
    assert len(team["rounds"]["paths"]) == 10


def test_match_tracking_records_slots_and_triples():
    simulation = simulate.simulate_season(
        _multi_round_frame(), tau=1.0, n_sims=200, seed=8, track_matches=True
    )
    assert simulation.match_groups is not None
    assert len(simulation.match_groups) == 2
    for counts in simulation.match_slot_counts:
        assert counts.sum() == 200 * 3
    for triples in simulation.match_triple_counts:
        assert sum(triples.values()) <= 200
        assert max(triples.values()) > 0


def test_forecast_export_includes_matches_and_reports():
    simulation = simulate.simulate_season(
        _multi_round_frame(),
        tau=1.0,
        n_sims=200,
        seed=11,
        track_rounds=True,
        track_matches=True,
        path_count=10,
    )
    players = simulation.players.sort_values("sim_mean", ascending=False)
    payload = simulate.forecast_export(
        simulation,
        players,
        top=4,
        metadata={"season": 2024},
        reports={"m1": {"source": "test", "text": "A classic."}},
    )
    assert len(payload["matches"]) == 2
    match = payload["matches"][0]
    assert match["id"] == "m1"
    assert {"home", "away", "votes", "triples"} <= set(match)
    assert len(match["votes"]) == 4
    assert match["triples"][0]["p"] > 0
    assert match["report"]["text"] == "A classic."
    team = payload["teams"][0]
    assert team["expected"] > 0
    assert len(team["players"]) == 4
