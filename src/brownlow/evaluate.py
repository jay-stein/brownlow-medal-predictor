"""Match-level allocation metrics and leak-free temperature evaluation.

Each match is treated as a Plackett-Luce draw over its players. Metrics cover
the observed ordered triple (allocation log-loss), the player-game vote
marginals (multiclass Brier score), and ranking quality (top-1 hit rate,
top-3 slot overlap). Temperatures for evaluation are fitted only on earlier
out-of-sample seasons, so no evaluation season informs its own tau.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import pl

VOTE_COLUMN = "BROWNLOW_VOTES_AUDITED"
MATCH_COLUMN = "PROVIDERID"
UTILITY_COLUMN = "UTILITY"


@dataclass
class MatchData:
    match_id: str
    utilities: np.ndarray
    votes: np.ndarray
    triple: tuple[int, int, int]


def build_matches(frame: pd.DataFrame, utility_column: str = UTILITY_COLUMN) -> list[MatchData]:
    """Group a scored season into evaluable matches.

    A match is evaluable when exactly three of its players polled 1, 2 and 3
    votes. Unresolved labels are kept in the player pool (they did not poll)
    but never treated as recipients.
    """
    matches: list[MatchData] = []
    for match_id, group in frame.groupby(MATCH_COLUMN, sort=False):
        votes = group[VOTE_COLUMN]
        recipient_mask = votes.notna() & (votes > 0)
        if not recipient_mask.any():
            continue
        recipients = group.loc[recipient_mask].sort_values(VOTE_COLUMN, ascending=False)
        if len(recipients) != 3 or sorted(recipients[VOTE_COLUMN].tolist()) != [1, 2, 3]:
            continue
        positions = {label: index for index, label in enumerate(group.index)}
        matches.append(
            MatchData(
                match_id=str(match_id),
                utilities=group[utility_column].to_numpy(dtype=float),
                votes=votes.fillna(0.0).to_numpy(dtype=float),
                triple=tuple(int(positions[label]) for label in recipients.index),
            )
        )
    return matches


def allocation_log_loss(matches: list[MatchData], tau: float) -> float:
    """Mean negative log probability of the observed ordered triple."""
    if not matches:
        return float("nan")
    losses = [-pl.match_log_likelihood(match.utilities, match.triple, tau) for match in matches]
    return float(np.mean(losses))


def uniform_log_loss(matches: list[MatchData]) -> float:
    """Reference log-loss when every ordered triple is equally likely."""
    if not matches:
        return float("nan")
    losses = []
    for match in matches:
        n = len(match.utilities)
        losses.append(float(np.log(n * (n - 1) * (n - 2))))
    return float(np.mean(losses))


def _brier_from_marginals(marginals: np.ndarray, votes: np.ndarray) -> float:
    observed = np.zeros_like(marginals)
    observed[np.arange(len(votes)), votes.astype(int)] = 1.0
    return float(np.mean(np.sum((marginals - observed) ** 2, axis=1)))


def brier_score(matches: list[MatchData], tau: float) -> float:
    """Mean multiclass Brier score over player-games for P(0/1/2/3)."""
    if not matches:
        return float("nan")
    scores = [
        _brier_from_marginals(pl.pl_marginals(match.utilities, tau), match.votes)
        for match in matches
    ]
    return float(np.mean(scores))


def brier_score_uniform(matches: list[MatchData]) -> float:
    """Brier score when all players are exchangeable (uniform weights)."""
    if not matches:
        return float("nan")
    scores = [
        _brier_from_marginals(pl.pl_marginals(np.zeros(len(match.utilities))), match.votes)
        for match in matches
    ]
    return float(np.mean(scores))


def topk_metrics(matches: list[MatchData]) -> dict:
    """Ranking quality of the utility order against the observed triple."""
    if not matches:
        return {
            "top1_hit": float("nan"),
            "top3_slot_overlap": float("nan"),
            "top3_set_hit": float("nan"),
        }
    top1 = 0
    set_hits = 0
    slots = 0.0
    for match in matches:
        order = np.argsort(-match.utilities, kind="stable")
        top3 = order[:3]
        top1 += int(top3[0] == match.triple[0])
        slots += len({int(i) for i in top3} & set(match.triple)) / 3.0
        set_hits += int({int(i) for i in top3} == set(match.triple))
    n = len(matches)
    return {
        "top1_hit": top1 / n,
        "top3_slot_overlap": slots / n,
        "top3_set_hit": set_hits / n,
    }


def reliability_table(matches: list[MatchData], tau: float, n_bins: int = 10) -> pd.DataFrame:
    """Reliability of P(polls at least one vote) over quantile bins."""
    rows = []
    for match in matches:
        marginals = pl.pl_marginals(match.utilities, tau)
        rows.append(
            pd.DataFrame(
                {
                    "p_poll": marginals[:, 1:].sum(axis=1),
                    "observed": (match.votes > 0).astype(int),
                }
            )
        )
    data = pd.concat(rows, ignore_index=True)
    data["bin"] = pd.qcut(data["p_poll"], q=n_bins, duplicates="drop")
    table = (
        data.groupby("bin", observed=True)
        .agg(mean_predicted=("p_poll", "mean"), observed_rate=("observed", "mean"), n=("observed", "size"))
        .reset_index()
    )
    table["bin"] = table["bin"].astype(str)
    return table


def fit_pooled_tau(frames: list[pd.DataFrame]) -> tuple[float, int]:
    """Fit one temperature on pooled matches from the supplied score frames."""
    score_arrays: list[np.ndarray] = []
    triples: list[tuple[int, int, int]] = []
    for frame in frames:
        for match in build_matches(frame):
            score_arrays.append(match.utilities)
            triples.append(match.triple)
    if not score_arrays:
        return 1.0, 0
    return pl.fit_tau(score_arrays, triples), len(score_arrays)


def leak_free_taus(scores_by_season: dict[int, pd.DataFrame]) -> dict[int, dict]:
    """Leak-free temperature per season (fitted on every earlier cached season)."""
    taus: dict[int, dict] = {}
    for season in sorted(scores_by_season):
        prior_frames = [scores_by_season[s] for s in sorted(scores_by_season) if s < season]
        tau, source_matches = fit_pooled_tau(prior_frames)
        taus[season] = {"tau": tau, "source_matches": source_matches}
    return taus


def crps_ensemble(samples: np.ndarray, observed: float) -> float:
    """Continuous ranked probability score for a finite ensemble."""
    ordered = np.sort(np.asarray(samples, dtype=float))
    n = len(ordered)
    weights = 2.0 * np.arange(1, n + 1) - n - 1
    pairwise = float(np.sum(weights * ordered))
    return float(np.mean(np.abs(ordered - observed)) - pairwise / n**2)


def interval_coverage(samples: np.ndarray, observed: float, level: float = 0.9) -> tuple[bool, float]:
    """Whether the central interval covers the observation, and its width."""
    alpha = (1.0 - level) / 2.0
    lower = float(np.quantile(samples, alpha))
    upper = float(np.quantile(samples, 1.0 - alpha))
    return bool(lower <= observed <= upper), upper - lower


def evaluate_backtests(
    scores_by_season: dict[int, pd.DataFrame],
    seasons: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leak-free evaluation for every season with at least one earlier season.

    ``tau`` for season Y is fitted on out-of-sample scores from every cached
    season before Y. The oracle tau (fitted on Y itself) is reported only as a
    diagnostic and is never used for the headline metrics. ``seasons`` limits
    which seasons are reported while all supplied frames remain tau sources.
    """
    metric_rows: list[dict] = []
    reliability_frames: list[pd.DataFrame] = []
    taus = leak_free_taus(scores_by_season)
    report_seasons = sorted(scores_by_season) if seasons is None else sorted(seasons)
    for season in report_seasons:
        if season not in scores_by_season:
            continue
        frame = scores_by_season[season]
        matches = build_matches(frame)
        if not matches:
            continue
        tau_leak_free = taus[season]["tau"]
        tau_source_matches = taus[season]["source_matches"]
        tau_oracle = pl.fit_tau(
            [match.utilities for match in matches],
            [match.triple for match in matches],
        )
        row = {
            "season": season,
            "n_fixtures": int(frame[MATCH_COLUMN].nunique()),
            "n_matches": len(matches),
            "tau_leak_free": tau_leak_free,
            "tau_source_matches": tau_source_matches,
            "tau_oracle": tau_oracle,
            "nll_leak_free": allocation_log_loss(matches, tau_leak_free),
            "nll_tau1": allocation_log_loss(matches, 1.0),
            "nll_oracle": allocation_log_loss(matches, tau_oracle),
            "nll_uniform": uniform_log_loss(matches),
            "brier_leak_free": brier_score(matches, tau_leak_free),
            "brier_tau1": brier_score(matches, 1.0),
            "brier_uniform": brier_score_uniform(matches),
        }
        row["nll_skill_vs_uniform"] = 1.0 - row["nll_leak_free"] / row["nll_uniform"]
        row.update(topk_metrics(matches))
        metric_rows.append(row)

        reliability = reliability_table(matches, tau_leak_free)
        reliability.insert(0, "season", season)
        reliability_frames.append(reliability)

    metrics = pd.DataFrame(metric_rows)
    if reliability_frames:
        reliability = pd.concat(reliability_frames, ignore_index=True)
    else:
        reliability = pd.DataFrame()
    return metrics, reliability
