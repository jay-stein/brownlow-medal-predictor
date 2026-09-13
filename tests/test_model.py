import numpy as np
import pandas as pd

from brownlow import model, pl


def _synthetic_frame(
    n_matches: int = 4, n_players: int = 6
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    rng = np.random.default_rng(0)
    rows = []
    labels = []
    match_ids = []
    season_ids = []
    round_ids = []
    votes = [3, 2, 1, 0, 0, 0]
    for match in range(n_matches):
        season = 2020 + match // 2
        round_number = match
        for player in range(n_players):
            rows.append(
                {
                    "F_GAME": rng.normal(),
                    "F_PLAYER": rng.normal(),
                    "PLAYER_PLAYER_POSITION": pd.Categorical(["MID"])[0],
                }
            )
            labels.append(votes[player])
            match_ids.append(f"m{match}")
            season_ids.append(season)
            round_ids.append(round_number)
    X = pd.DataFrame(rows)
    X["PLAYER_PLAYER_POSITION"] = pd.Categorical(X["PLAYER_PLAYER_POSITION"], categories=["MID", "FWD"])
    return (
        X,
        pd.Series(labels, dtype=float),
        pd.Series(match_ids),
        pd.Series(season_ids),
        pd.Series(round_ids),
    )


def test_objective_params_returns_registered_models():
    assert model.objective_params(model.REGRESSION)["objective"] == "reg:pseudohubererror"
    assert model.objective_params(model.RANKING)["objective"] == "rank:ndcg"
    assert model.is_ranking(model.objective_params(model.RANKING))
    assert not model.is_ranking(model.objective_params(model.REGRESSION))


def test_select_best_round_ranking_smoke():
    X, y, match_ids, season_ids, round_ids = _synthetic_frame()
    result = model.select_best_round(
        X,
        y,
        match_ids,
        season_ids,
        round_ids,
        params=model.RANKING_PARAMS,
        num_boost_round=10,
        early_stopping_rounds=5,
        n_splits=1,
        seed=0,
    )
    assert result["best_round"] >= 1
    assert result["objective"] == "rank:ndcg"
    assert result["cv_scheme"] == "time_ordered_seasons"
    assert result["fold_validation_seasons"] == [[2021]]
    assert len(result["fold_scores"]) == 1
    booster = model.fit_utilities(
        X, y, result["best_round"], params=model.RANKING_PARAMS, match_ids=match_ids, seed=0
    )
    predictions = model.predict_utilities(booster, X)
    assert predictions.shape == (len(X),)
    assert np.isfinite(predictions).all()


def test_ranking_requires_match_ids():
    X, y, _, _, _ = _synthetic_frame()
    try:
        model.build_dmatrix(X, y, match_ids=None, ranking=True)
    except ValueError as error:
        assert "match_ids" in str(error)
    else:
        raise AssertionError("expected ValueError when ranking without match ids")


def test_ranking_dmatrix_accepts_instance_weights():
    X, y, match_ids, _, _ = _synthetic_frame()
    weights = pd.Series([0.5] * len(X))
    dmatrix = model.build_dmatrix(X, y, match_ids, ranking=True, weight=weights)
    assert dmatrix.num_row() == len(X)


def test_model_options_registry():
    assert model.model_options(model.RANKING_RECENT)["train_window"] == 6
    assert model.model_options(model.RANKING_WEIGHTED)["recency_half_life"] == 4.0
    assert model.model_options(model.RANKING_NOCBA)["drop_features"] == model.CBA_FEATURES
    assert model.model_options(model.RANKING_COACHES)["extra_features"] == model.COACH_FEATURES
    form_features = model.model_options(model.RANKING_FORM)["extra_features"]
    assert "SEASON_COACH_VOTES_TOTAL" in form_features
    assert "OPPONENT_ELO" in form_features
    assert model.is_pl(model.RANKING_PL)
    assert not model.is_pl(model.RANKING)


def test_default_device_reads_environment(monkeypatch):
    monkeypatch.setenv("BROWNLOW_XGB_DEVICE", "cuda")
    monkeypatch.setattr(model, "_DEVICE", None)
    assert model.default_device() == "cuda"
    monkeypatch.setenv("BROWNLOW_XGB_DEVICE", "")
    monkeypatch.setattr(model, "_DEVICE", None)
    assert model.default_device() == "cpu"


def test_pl_gradient_matches_finite_differences():
    rng = np.random.default_rng(5)
    for _ in range(5):
        n = int(rng.integers(4, 9))
        scores = rng.normal(size=n)
        order = rng.choice(n, size=3, replace=False)
        gradient, _ = model._pl_grad_hess(scores, order, 1.0)
        numeric = np.zeros(n)
        step = 1e-6
        for index in range(n):
            plus = scores.copy()
            plus[index] += step
            minus = scores.copy()
            minus[index] -= step
            triple = tuple(int(value) for value in order)
            numerator = pl.match_log_likelihood(plus, triple, 1.0) - pl.match_log_likelihood(
                minus, triple, 1.0
            )
            numeric[index] = -numerator / (2 * step)
        np.testing.assert_allclose(gradient, numeric, atol=1e-5)


def test_pl_training_smoke():
    X, y, match_ids, season_ids, round_ids = _synthetic_frame()
    result = model.select_best_round(
        X,
        y,
        match_ids,
        season_ids,
        round_ids,
        params=model.PL_PARAMS,
        num_boost_round=5,
        early_stopping_rounds=3,
        n_splits=1,
        seed=0,
        use_pl=True,
    )
    booster = model.fit_utilities(
        X,
        y,
        result["best_round"],
        params=model.PL_PARAMS,
        match_ids=match_ids,
        seed=0,
        use_pl=True,
    )
    predictions = model.predict_utilities(booster, X)
    assert np.isfinite(predictions).all()
    assert result["eval_metric"] == "pl_nll"
