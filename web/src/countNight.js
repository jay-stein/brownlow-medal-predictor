const Z90 = 1.6449;

export function remainingCount(roundIndex, roundCount) {
  return Math.max(0, roundCount - 1 - roundIndex);
}

function quantileAt(table, index) {
  if (!Array.isArray(table) || !table.length) return 0;
  const clamped = Math.min(Math.max(index, 0), table.length - 1);
  return table[clamped] ?? 0;
}

export function playerProjection(player, roundIndex, observed, calibration, roundCount) {
  const cumulative = player?.rounds?.cumMean;
  if (!Array.isArray(cumulative) || !cumulative.length) return null;
  const last = cumulative.length - 1;
  const index = Math.min(Math.max(roundIndex, 0), last);
  const priorThrough = cumulative[index] ?? 0;
  const priorFinal = cumulative[last];
  const remaining = remainingCount(index, roundCount ?? cumulative.length);
  return {
    priorThrough,
    priorFinal,
    projected: priorFinal + (observed ?? 0) - priorThrough,
    half80: quantileAt(calibration?.q80, remaining),
    half90: quantileAt(calibration?.q90, remaining),
    remaining,
  };
}

export function remainingEstimates(player, roundIndex) {
  const increments = player?.rounds?.incMean;
  if (!Array.isArray(increments)) return [];
  return increments.slice(roundIndex + 1);
}

/**
 * Cumulative path after the count: the observed total at the current round,
 * then the observed total plus the model's expected increments for the rounds
 * still to come (the lambda = 1 rule). Entries before the current round are
 * null; the chart draws the prior path there instead.
 */
export function projectedPath(player, roundIndex, observed) {
  const cumulative = player?.rounds?.cumMean;
  const increments = player?.rounds?.incMean;
  if (!Array.isArray(cumulative) || !cumulative.length) return [];
  const last = cumulative.length - 1;
  const index = Math.min(Math.max(roundIndex, 0), last);
  const path = cumulative.map(() => null);
  let running = observed ?? 0;
  path[index] = running;
  const inc = Array.isArray(increments) ? increments : [];
  for (let round = index + 1; round <= last; round += 1) {
    running += inc[round] ?? 0;
    path[round] = running;
  }
  return path;
}

/**
 * Fan of uncertainty around the projected path: zero at the round you are up
 * to (the total is known), widening with the rounds in between to the
 * calibrated range for the final total. It answers "how unsure are we about
 * the total at this point in the future?", so it grows as you look further
 * ahead - unlike the final-total range, which shrinks as the count runs.
 */
export function projectionBand(player, roundIndex, observed, calibration) {
  const path = projectedPath(player, roundIndex, observed);
  if (!path.length) return { lower: [], upper: [] };
  const last = path.length - 1;
  const index = Math.min(Math.max(roundIndex, 0), last);
  const finalHalf = quantileAt(calibration?.q90, last - index);
  const span = Math.max(last - index, 1);
  const lower = path.map(() => null);
  const upper = path.map(() => null);
  for (let round = index; round <= last; round += 1) {
    if (path[round] === null) continue;
    const half = finalHalf * Math.sqrt((round - index) / span);
    lower[round] = path[round] - half;
    upper[round] = path[round] + half;
  }
  return { lower, upper };
}

export function sigmaFromHalfWidth(halfWidth, z = Z90) {
  return Math.max((halfWidth ?? 0) / z, 1e-6);
}

/** The player's club match in a given round, or null on a bye. */
export function findRoundMatch(matches, team, roundNumber) {
  if (!Array.isArray(matches) || !team) return null;
  return (
    matches.find(
      (match) => match.round === roundNumber && (match.home === team || match.away === team)
    ) ?? null
  );
}

function normalPdf(value) {
  return Math.exp(-0.5 * value * value) / Math.sqrt(2 * Math.PI);
}

function normalCdf(value) {
  return 0.5 * (1 + erf(value / Math.SQRT2));
}

// Abramowitz & Stegun 7.1.26
function erf(value) {
  const sign = value < 0 ? -1 : 1;
  const x = Math.abs(value);
  const t = 1 / (1 + 0.3275911 * x);
  const poly =
    ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t +
      0.254829592) *
    t;
  return sign * (1 - poly * Math.exp(-x * x));
}

/**
 * Probability each entry finishes with the highest total among the group,
 * respecting eligibility (ineligible players cannot win). Exact up to grid
 * integration: P(i wins) = integral of phi_i(x) * prod_{j != i} Phi_j(x) dx.
 */
export function winOdds(entries, points = 600) {
  const active = entries.map((entry) => ({
    ...entry,
    sigma: Math.max(entry.sigma ?? 0, 1e-6),
  }));
  const result = Object.fromEntries(active.map((entry) => [entry.id, 0]));
  if (!active.length) return result;
  const eligible = active.filter((entry) => entry.eligible !== false);
  if (!eligible.length) return result;

  const low = Math.min(...active.map((entry) => entry.projected - 8 * entry.sigma));
  const high = Math.max(...active.map((entry) => entry.projected + 8 * entry.sigma));
  const width = Math.max(high - low, 1e-6);
  const step = width / points;
  const others = eligible; // ineligible players are excluded from the field

  for (let index = 0; index < points; index += 1) {
    const x = low + (index + 0.5) * step;
    for (const entry of eligible) {
      const own = normalPdf((x - entry.projected) / entry.sigma) / entry.sigma;
      if (own === 0) continue;
      let product = own;
      for (const other of others) {
        if (other.id === entry.id) continue;
        product *= normalCdf((x - other.projected) / other.sigma);
        if (product === 0) break;
      }
      result[entry.id] += product * step;
    }
  }
  const total = Object.values(result).reduce((sum, value) => sum + value, 0);
  if (total > 0) {
    for (const id of Object.keys(result)) result[id] /= total;
  }
  return result;
}
