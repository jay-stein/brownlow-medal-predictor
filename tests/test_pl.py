import itertools

import numpy as np
import pytest

from brownlow.pl import (
    fit_tau,
    match_log_likelihood,
    match_log_likelihood_samples,
    pl_marginals,
    pl_weights,
)


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


def explicit_sequential_marginals(weights: np.ndarray) -> np.ndarray:
    """P(1/2/3) written as explicit sums over the first two placements.

    Sequential Plackett-Luce denominators depend on the preceding picks, so
    the symmetric-polynomial formulas used by other weighted-subset models do
    not apply; this is the direct two-placement expansion.
    """
    n = len(weights)
    total = weights.sum()
    p1 = np.zeros(n)
    p2 = np.zeros(n)
    p3 = np.zeros(n)
    for i in range(n):
        p3[i] = weights[i] / total
        for a in range(n):
            if a == i:
                continue
            first = weights[a] / total
            p2[i] += first * weights[i] / (total - weights[a])
            for b in range(n):
                if b == a or b == i:
                    continue
                p1[i] += (
                    first
                    * (weights[b] / (total - weights[a]))
                    * (weights[i] / (total - weights[a] - weights[b]))
                )
    return np.column_stack([1.0 - p1 - p2 - p3, p1, p2, p3])


def monte_carlo_marginals(
    s: np.ndarray, tau: float, n_draws: int, rng: np.random.Generator
) -> np.ndarray:
    """Exact Plackett-Luce draws via the Gumbel-max trick.

    Adding i.i.d. Gumbel noise to the log-weights and sorting gives the same
    distribution as sequential weighted sampling without replacement.
    """
    weights = pl_weights(s, tau)
    perturbed = np.log(weights)[:, None] + rng.gumbel(size=(len(weights), n_draws))
    order = np.argsort(-perturbed, axis=0)[:3]
    counts = np.zeros((len(weights), 4))
    for slot, rank in zip((3, 2, 1), range(3)):
        np.add.at(counts, (order[rank], slot), 1)
    counts[:, 0] = n_draws - counts[:, 1:].sum(axis=1)
    return counts / n_draws


def test_marginals_match_brute_force():
    rng = np.random.default_rng(7)
    utilities = rng.normal(size=6) * 1.3
    np.testing.assert_allclose(
        pl_marginals(utilities, tau=1.7),
        brute_force_marginals(utilities, tau=1.7),
        atol=1e-12,
    )


def test_marginals_match_explicit_sequential_formulas():
    rng = np.random.default_rng(23)
    for trial in range(20):
        n = int(rng.integers(3, 9))
        utilities = rng.normal(size=n) * rng.uniform(0.5, 4.0)
        tau = float(rng.uniform(0.3, 3.0))
        weights = pl_weights(utilities, tau)
        np.testing.assert_allclose(
            pl_marginals(utilities, tau),
            explicit_sequential_marginals(weights),
            atol=1e-10,
            err_msg=f"trial {trial}: n={n}, tau={tau:.3f}",
        )


def test_marginals_match_monte_carlo_simulation():
    rng = np.random.default_rng(19)
    utilities = rng.normal(size=12) * 1.5
    tau = 0.7
    n_draws = 50_000
    exact = pl_marginals(utilities, tau)
    sampled = monte_carlo_marginals(utilities, tau, n_draws, rng)
    tolerance = 4.0 * np.sqrt(exact * (1.0 - exact) / n_draws) + 0.002
    assert np.abs(sampled - exact).max() < tolerance.max()
    np.testing.assert_allclose(sampled.sum(axis=1), 1.0, atol=1e-12)


def test_marginals_are_distributions_and_award_six_votes():
    rng = np.random.default_rng(11)
    utilities = rng.normal(size=20) * 2.0
    marginal = pl_marginals(utilities)
    np.testing.assert_allclose(marginal.sum(axis=1), 1.0, atol=1e-12)
    expected_votes = marginal @ np.array([0.0, 1.0, 2.0, 3.0])
    assert expected_votes.sum() == pytest.approx(6.0, abs=1e-9)


def test_each_vote_slot_probability_sums_to_one():
    rng = np.random.default_rng(29)
    for n, tau in ((3, 0.4), (17, 0.9), (44, 1.0), (44, 0.3)):
        utilities = rng.normal(size=n) * 2.0
        marginal = pl_marginals(utilities, tau)
        np.testing.assert_allclose(marginal[:, 1:].sum(axis=0), np.ones(3), atol=1e-10)


def test_marginals_are_finite_for_extreme_utilities():
    utilities = np.array([80.0, 40.0, 0.0, -40.0, -80.0])
    marginal = pl_marginals(utilities, tau=0.2)
    assert np.isfinite(marginal).all()
    assert marginal.min() >= 0.0
    np.testing.assert_allclose(marginal.sum(axis=1), 1.0, atol=1e-12)


def test_match_log_likelihood_prefers_the_observed_order():
    utilities = np.array([10.0, 0.0, 0.0, 0.0])
    assert match_log_likelihood(utilities, (0, 1, 2)) > match_log_likelihood(utilities, (1, 2, 3))


def test_match_log_likelihood_samples_matches_scalar_version():
    rng = np.random.default_rng(31)
    for _ in range(5):
        n = int(rng.integers(4, 10))
        utilities = rng.normal(size=n) * 1.5
        triple = tuple(int(index) for index in rng.choice(n, size=3, replace=False))
        scalar = match_log_likelihood(utilities, triple, tau=0.8)
        sampled = match_log_likelihood_samples(utilities[:, None], triple, tau=0.8)
        assert sampled.shape == (1,)
        np.testing.assert_allclose(sampled[0], scalar, atol=1e-12)


def test_fit_tau_recovers_known_temperature():
    rng = np.random.default_rng(3)
    scores = [rng.normal(size=10) * 1.5 for _ in range(1500)]
    triples = []
    for utilities in scores:
        weights = pl_weights(utilities, 1.0)
        picks = rng.choice(len(utilities), size=3, replace=False, p=weights / weights.sum())
        triples.append((int(picks[0]), int(picks[1]), int(picks[2])))
    assert 0.7 < fit_tau(scores, triples) < 1.4
