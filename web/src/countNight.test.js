import assert from "node:assert/strict";
import test from "node:test";

import {
  playerProjection,
  remainingCount,
  remainingEstimates,
  sigmaFromHalfWidth,
  winOdds,
} from "./countNight.js";

const player = {
  id: "p1",
  name: "Test Player",
  rounds: {
    cumMean: [2, 5, 9, 14],
    incMean: [2, 3, 4, 5],
  },
};

test("projection adds observed votes to the remaining expectation", () => {
  const projection = playerProjection(player, 1, 7, { q80: [0, 1, 2], q90: [0, 2, 3] }, 4);
  // prior through round 1 = 5, prior final = 14, observed = 7
  assert.equal(projection.priorThrough, 5);
  assert.equal(projection.priorFinal, 14);
  assert.equal(projection.projected, 16);
  assert.equal(projection.remaining, 2);
  assert.equal(projection.half90, 3);
  assert.equal(projection.half80, 2);
});

test("round index clamps to the season length", () => {
  assert.equal(remainingCount(0, 4), 3);
  assert.equal(remainingCount(3, 4), 0);
  assert.equal(remainingCount(9, 4), 0);
  const projection = playerProjection(player, 99, 14, null, 4);
  assert.equal(projection.priorThrough, 14);
  assert.equal(projection.projected, 14);
  assert.equal(projection.half90, 0);
});

test("remaining estimates slice the round increments", () => {
  assert.deepEqual(remainingEstimates(player, 1), [4, 5]);
  assert.deepEqual(remainingEstimates(player, 3), []);
});

test("sigma conversion keeps a floor", () => {
  assert.equal(sigmaFromHalfWidth(1.6449), 1);
  assert.ok(sigmaFromHalfWidth(0) > 0);
});

test("win odds with one player is certain", () => {
  const odds = winOdds([{ id: "a", projected: 30, sigma: 3, eligible: true }]);
  assert.equal(odds.a, 1);
});

test("win odds split evenly between identical players", () => {
  const odds = winOdds([
    { id: "a", projected: 30, sigma: 3, eligible: true },
    { id: "b", projected: 30, sigma: 3, eligible: true },
  ]);
  assert.ok(Math.abs(odds.a - 0.5) < 0.001);
  assert.ok(Math.abs(odds.b - 0.5) < 0.001);
});

test("a clear leader takes almost all probability", () => {
  const odds = winOdds([
    { id: "a", projected: 40, sigma: 2, eligible: true },
    { id: "b", projected: 30, sigma: 2, eligible: true },
  ]);
  assert.ok(odds.a > 0.999);
});

test("ineligible players cannot win", () => {
  const odds = winOdds([
    { id: "a", projected: 40, sigma: 2, eligible: false },
    { id: "b", projected: 30, sigma: 2, eligible: true },
  ]);
  assert.equal(odds.a, 0);
  assert.equal(odds.b, 1);
});

test("probabilities sum to one across a realistic field", () => {
  const odds = winOdds([
    { id: "a", projected: 41, sigma: 6, eligible: true },
    { id: "b", projected: 38, sigma: 6, eligible: true },
    { id: "c", projected: 36, sigma: 6, eligible: true },
    { id: "d", projected: 30, sigma: 6, eligible: true },
    { id: "e", projected: 28, sigma: 6, eligible: true },
  ]);
  const total = Object.values(odds).reduce((sum, value) => sum + value, 0);
  assert.ok(Math.abs(total - 1) < 1e-6);
  assert.ok(odds.a > odds.b && odds.b > odds.c && odds.c > odds.d && odds.d > odds.e);
});
