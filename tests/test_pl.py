import itertools

import numpy as np
import pytest

from brownlow.pl import fit_tau, match_log_likelihood, pl_marginals, pl_weights


def brute_force_marginals(s: np.ndarray, tau: float = 1.0) -> np.ndarray:
    n = len(s)
    weights = pl_weights(s, tau)
    marginal = np.zeros((n, 4))
    for a, b, c in itertools.permutations(range(n), 3):
        probability = (
            weights[a]
            / weights.sum()
            * weights[b]
            / (weights.sum() - weights[a])
            * weights[c]
            / (weights.sum() - weights[a] - weights[b])
        )
        marginal[a, 3] += probability
        marginal[b, 2] += probability
        marginal[c, 1] += probability
    marginal[:, 0] = 1.0 - marginal[:, 1] - marginal[:, 2] - marginal[:, 3]
    return marginal


def test_marginals_match_brute_force():
    rng = np.random.default_rng(7)
    utilities = rng.normal(size=6) * 1.3
    np.testing.assert_allclose(
        pl_marginals(utilities, tau=1.7),
        brute_force_marginals(utilities, tau=1.7),
        atol=1e-12,
    )


def test_marginals_are_distributions_and_award_six_votes():
    rng = np.random.default_rng(11)
    utilities = rng.normal(size=20) * 2.0
    marginal = pl_marginals(utilities)
    np.testing.assert_allclose(marginal.sum(axis=1), 1.0, atol=1e-12)
    expected_votes = marginal @ np.array([0.0, 1.0, 2.0, 3.0])
    assert expected_votes.sum() == pytest.approx(6.0, abs=1e-9)


def test_match_log_likelihood_prefers_the_observed_order():
    utilities = np.array([10.0, 0.0, 0.0, 0.0])
    assert match_log_likelihood(utilities, (0, 1, 2)) > match_log_likelihood(utilities, (1, 2, 3))


def test_fit_tau_recovers_known_temperature():
    rng = np.random.default_rng(3)
    scores = [rng.normal(size=10) * 1.5 for _ in range(1500)]
    triples = []
    for utilities in scores:
        weights = pl_weights(utilities, 1.0)
        picks = rng.choice(len(utilities), size=3, replace=False, p=weights / weights.sum())
        triples.append((int(picks[0]), int(picks[1]), int(picks[2])))
    assert 0.7 < fit_tau(scores, triples) < 1.4
