import numpy as np
import pandas as pd
import pytest

from brownlow import evaluate, pl


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PROVIDERID": ["m1"] * 4 + ["m2"] * 4,
            "UTILITY": [2.0, 1.0, 0.5, 0.0, 0.0, 3.0, 2.0, 1.0],
            "BROWNLOW_VOTES_AUDITED": [3.0, 2.0, 1.0, 0.0, 0.0, 3.0, 2.0, 1.0],
        }
    )


def test_build_matches_groups_and_orders_recipients():
    matches = evaluate.build_matches(_frame())
    assert len(matches) == 2
    assert matches[0].triple == (0, 1, 2)
    assert list(matches[0].votes) == [3.0, 2.0, 1.0, 0.0]
    assert matches[1].triple == (1, 2, 3)


def test_build_matches_keeps_unresolved_rows_in_the_pool():
    frame = _frame()
    extra = pd.DataFrame(
        {"PROVIDERID": ["m1"], "UTILITY": [0.1], "BROWNLOW_VOTES_AUDITED": [np.nan]}
    )
    matches = evaluate.build_matches(pd.concat([frame, extra], ignore_index=True))
    assert len(matches[0].utilities) == 5
    assert matches[0].votes[-1] == 0.0


def test_allocation_log_loss_matches_match_likelihood():
    matches = evaluate.build_matches(_frame())
    expected = np.mean(
        [-pl.match_log_likelihood(match.utilities, match.triple, 0.7) for match in matches]
    )
    assert evaluate.allocation_log_loss(matches, 0.7) == pytest.approx(expected)


def test_uniform_log_loss_formula():
    matches = evaluate.build_matches(_frame())
    assert evaluate.uniform_log_loss(matches) == pytest.approx(np.log(4 * 3 * 2))


def test_topk_metrics_perfect_order():
    metrics = evaluate.topk_metrics(evaluate.build_matches(_frame()))
    assert metrics["top1_hit"] == 1.0
    assert metrics["top3_slot_overlap"] == 1.0
    assert metrics["top3_set_hit"] == 1.0


def test_brier_score_in_valid_range():
    matches = evaluate.build_matches(_frame())
    score = evaluate.brier_score(matches, 1.0)
    uniform = evaluate.brier_score_uniform(matches)
    assert 0.0 <= score <= 2.0
    assert 0.0 <= uniform <= 2.0


def test_fit_pooled_tau_uses_all_frames():
    tau, n_matches = evaluate.fit_pooled_tau([_frame(), _frame()])
    assert n_matches == 4
    assert tau > 0


def test_crps_degenerate_ensemble():
    assert evaluate.crps_ensemble(np.full(10, 2.0), 2.0) == pytest.approx(0.0)
    assert evaluate.crps_ensemble(np.full(10, 2.0), 5.0) == pytest.approx(3.0)


def test_crps_small_ensembles_match_manual_values():
    assert evaluate.crps_ensemble(np.array([0.0, 2.0]), 1.0) == pytest.approx(0.5)
    assert evaluate.crps_ensemble(np.array([1.0, 2.0, 3.0]), 2.0) == pytest.approx(2 / 9)


def test_interval_coverage_width_and_miss():
    samples = np.arange(101, dtype=float)
    covered, width = evaluate.interval_coverage(samples, 50.0, level=0.9)
    assert covered is True
    assert width == pytest.approx(90.0)
    covered_out, _ = evaluate.interval_coverage(samples, 200.0, level=0.9)
    assert covered_out is False


def test_leak_free_taus_flags_seasons_without_prior_scores():
    taus = evaluate.leak_free_taus({2020: _frame(), 2021: _frame()})
    assert taus[2020] == {"tau": 1.0, "source_matches": 0}
    assert taus[2021]["source_matches"] == 2

