import pandas as pd

from brownlow import validation


def _season(year: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ROUND_YEAR": [year] * 4,
            "PROVIDERID": [f"m{year}"] * 4,
            "ROUND_ROUNDNUMBER": [0] * 4,
            "PLAYER_PLAYER_PLAYER_PLAYERID": [f"p{index}" for index in range(4)],
            "FULL_NAME": list("ABCD"),
            "TEAM_NAME": ["T"] * 4,
            "UTILITY": [3.0, 2.0, 1.0, 0.0],
            "BROWNLOW_VOTES_AUDITED": [3.0, 2.0, 1.0, 0.0],
        }
    )


def _frames(years: list[int]) -> dict[int, pd.DataFrame]:
    return {year: _season(year) for year in years}


def test_joint_grid_covers_all_candidates():
    grid = validation.joint_calibration_grid(
        _frames([2020, 2021]),
        [2020, 2021],
        [0.5, 1.0],
        [0.0, 0.2],
        n_sims=50,
        n_draws=16,
    )
    assert len(grid) == 2 * 2 * 2
    assert set(grid["tau"]) == {0.5, 1.0}
    assert set(grid["effect_scale"]) == {0.0, 0.2}
    assert grid["match_nll"].notna().all()
    assert grid["mean_crps_contenders"].notna().all()


def test_select_joint_params_returns_grid_member_and_reports_tension():
    grid = validation.joint_calibration_grid(
        _frames([2020, 2021, 2022]),
        [2020, 2021, 2022],
        [0.5, 1.0],
        [0.0, 0.2],
        n_sims=50,
        n_draws=16,
    )
    selection = validation.select_joint_params(grid, [2020, 2021])
    assert selection.tau in {0.5, 1.0}
    assert selection.effect_scale in {0.0, 0.2}
    assert len(selection.summary) == 4
    best = selection.summary.sort_values(["combined", "match_nll"]).iloc[0]
    assert selection.tau == best["tau"]
    assert selection.effect_scale == best["effect_scale"]
    assert selection.best_match_nll_tau in {0.5, 1.0}
    assert selection.best_crps_tau in {0.5, 1.0}


def test_rolling_validation_selection_uses_only_earlier_seasons():
    years = list(range(2018, 2023))
    grid = validation.joint_calibration_grid(
        _frames(years), years, [0.5, 1.0], [0.0, 0.2], n_sims=50, n_draws=16
    )
    table = validation.rolling_validation(
        grid, report_seasons=[2021, 2022], min_evidence=3
    )
    assert list(table["season"]) == [2021, 2022]
    for _, row in table.iterrows():
        evidence = [year for year in years if year < row["season"]]
        expected = validation.select_joint_params(grid, evidence)
        assert row["tau_selected"] == expected.tau
        assert row["scale_selected"] == expected.effect_scale
        assert row["evidence_seasons"] == len(evidence)
        assert row["evidence_last"] < row["season"]


def test_rolling_validation_skips_insufficient_evidence():
    grid = validation.joint_calibration_grid(
        _frames([2020, 2021]), [2020, 2021], [0.5], [0.0], n_sims=30, n_draws=8
    )
    table = validation.rolling_validation(grid, report_seasons=[2021], min_evidence=3)
    assert table.empty
