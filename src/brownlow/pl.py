"""Plackett-Luce allocation model for a match's 3-2-1 votes.

Utilities ``s`` score each player in a match; weights ``w = exp(s / tau)``.
The probability of an ordered triple ``(a, b, c)`` (a gets 3 votes) is

    w_a / sum_i w_i * w_b / sum_{i != a} w_i * w_c / sum_{i not in {a,b}} w_i

so every match awards exactly one 3, one 2 and one 1.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

TAU_BOUNDS = (1e-3, 1e3)


def pl_weights(s: np.ndarray, tau: float = 1.0) -> np.ndarray:
    """Stable Plackett-Luce weights (max utility shifted to zero)."""
    s = np.asarray(s, dtype=float)
    return np.exp((s - s.max()) / tau)


def match_log_likelihood(s: np.ndarray, triple: tuple[int, int, int], tau: float = 1.0) -> float:
    """Log probability of the observed ordered triple."""
    weights = pl_weights(s, tau)
    remaining = weights.copy()
    log_prob = 0.0
    for idx in triple:
        total = remaining.sum()
        log_prob += float(np.log(remaining[idx] / total))
        remaining[idx] = 0.0
    return log_prob


def negative_log_likelihood(
    scores: list[np.ndarray],
    triples: list[tuple[int, int, int]],
    tau: float,
) -> float:
    return -sum(
        match_log_likelihood(s, triple, tau) for s, triple in zip(scores, triples)
    )


def fit_tau(
    scores: list[np.ndarray],
    triples: list[tuple[int, int, int]],
    bounds: tuple[float, float] = TAU_BOUNDS,
) -> float:
    """Fit a single positive temperature on fixed utilities."""
    if len(scores) != len(triples):
        raise ValueError("scores and triples must have the same length")
    result = minimize_scalar(
        lambda log_tau: negative_log_likelihood(scores, triples, float(np.exp(log_tau))),
        bounds=(np.log(bounds[0]), np.log(bounds[1])),
        method="bounded",
    )
    if not result.success:
        raise RuntimeError(f"temperature fit failed: {result.message}")
    return float(np.exp(result.x))


def pl_marginals(s: np.ndarray, tau: float = 1.0) -> np.ndarray:
    """Exact marginal probabilities of receiving 0, 1, 2 or 3 votes.

    Returns an ``(n_players, 4)`` array whose columns are ``P(0), P(1), P(2),
    P(3)``.

    Sequential denominators can underflow to zero for very sharp temperatures.
    Terms with a zero denominator are dropped: they are either multiplied by a
    zero weight (so contribute nothing) or belong to the exactly-two-positive-
    weights case, where the finite margins are still zero. This keeps the
    computation stable without perturbing the exact probabilities.
    """
    weights = pl_weights(s, tau)
    total = weights.sum()

    # P(3 votes) = P(selected first)
    p3 = weights / total

    # P(2 votes) = P(selected second)
    remaining = total - weights
    inverse_remaining = np.zeros_like(weights)
    positive = remaining > 0.0
    inverse_remaining[positive] = 1.0 / remaining[positive]
    shared = float(np.sum(weights * inverse_remaining) / total)
    p2 = weights * (shared - weights * inverse_remaining / total)

    # P(1 vote) = P(selected third); see docstring for the derivation.
    denominator = total * (total - weights)[:, None] * (total - weights[:, None] - weights[None, :])
    valid = denominator > 0.0
    pair_terms = np.zeros_like(denominator)
    pair_terms[valid] = (weights[:, None] * weights[None, :])[valid] / denominator[valid]
    np.fill_diagonal(pair_terms, 0.0)
    pair_total = float(pair_terms.sum())
    p1 = weights * (pair_total - pair_terms.sum(axis=1) - pair_terms.sum(axis=0))

    marginal = np.column_stack([1.0 - p1 - p2 - p3, p1, p2, p3])
    marginal = np.clip(marginal, 0.0, 1.0)
    return marginal / marginal.sum(axis=1, keepdims=True)
