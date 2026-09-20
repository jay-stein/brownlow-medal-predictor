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

export function sigmaFromHalfWidth(halfWidth, z = Z90) {
  return Math.max((halfWidth ?? 0) / z, 1e-6);
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
