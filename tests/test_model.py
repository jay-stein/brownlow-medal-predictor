import numpy as np
import pandas as pd

from brownlow import model


def _synthetic_frame(n_matches: int = 4, n_players: int = 6) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    rng = np.random.default_rng(0)
    rows = []
    labels = []
    match_ids = []
    votes = [3, 2, 1, 0, 0, 0]
    for match in range(n_matches):
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
    X = pd.DataFrame(rows)
    X["PLAYER_PLAYER_POSITION"] = pd.Categorical(X["PLAYER_PLAYER_POSITION"], categories=["MID", "FWD"])
    return X, pd.Series(labels, dtype=float), pd.Series(match_ids)


def test_objective_params_returns_registered_models():
    assert model.objective_params(model.REGRESSION)["objective"] == "reg:pseudohubererror"
    assert model.objective_params(model.RANKING)["objective"] == "rank:ndcg"
    assert model.is_ranking(model.objective_params(model.RANKING))
    assert not model.is_ranking(model.objective_params(model.REGRESSION))


def test_select_best_round_ranking_smoke():
    X, y, match_ids = _synthetic_frame()
    result = model.select_best_round(
        X,
        y,
        match_ids,
        params=model.RANKING_PARAMS,
        num_boost_round=10,
        early_stopping_rounds=5,
        n_splits=2,
        seed=0,
    )
    assert result["best_round"] >= 1
    assert result["objective"] == "rank:ndcg"
    assert len(result["fold_scores"]) == 2
    booster = model.fit_utilities(
        X, y, result["best_round"], params=model.RANKING_PARAMS, match_ids=match_ids, seed=0
    )
    predictions = model.predict_utilities(booster, X)
    assert predictions.shape == (len(X),)
    assert np.isfinite(predictions).all()


def test_ranking_requires_match_ids():
    X, y, _ = _synthetic_frame()
    try:
        model.build_dmatrix(X, y, match_ids=None, ranking=True)
    except ValueError as error:
        assert "match_ids" in str(error)
    else:
        raise AssertionError("expected ValueError when ranking without match ids")
