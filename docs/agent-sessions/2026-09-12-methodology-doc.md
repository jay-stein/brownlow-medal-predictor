# Session: Methodology Document

**Date:** 2026-09-12
**Branch:** `agent/methodology-doc`

## Goal

Add a detailed `docs/methodology.md` covering the mathematical approach, the Brownlow fixed-budget mechanics (teammate vote stealing), and — at the user's request — a dedicated treatment of uncertainty emphasising how the original model was overconfident and how the redesign addresses it.

## Files Changed

| File | Change |
|---|---|
| `docs/methodology.md` | New — 10 sections: fixed-budget mechanics with worked Plackett–Luce example and same-team recipient statistics; problem statement; data and labels; feature map; score generator; allocation and temperature; **uncertainty treatment** (four causes of the original overconfidence, six mechanisms in the redesign, what remains unmodelled); validation results; eleven modelling weak points; production parameters |
| `docs/agent-sessions/2026-09-12-methodology-doc.md` | This summary |

## Commands Executed

- `git checkout -b agent/methodology-doc`
- `uv run pytest` and `uv run ruff check .` (documentation-only change)
- `gh pr create` + `gh pr merge --merge`

## Important Decisions

1. **Uncertainty gets its own section** (requested): the original design's overconfidence is explained through ensemble spread measuring the wrong variance, i.i.d. noise cancelling over a season while persistent bias does not, the wrong distributional shape, and MC precision masquerading as model precision. The redesign's answers are listed alongside the validation evidence (88.5% contender coverage at the nominal 90% level, 36% average winner probability, specification ranges).
2. **Numbers are grounded in the implemented pipeline**: τ = 0.845, σ = 0.4, 10,000 simulations, and the 2012–2025 same-team recipient counts (48.9% of matches have all three recipients from one team).
3. **The worked example uses the production temperature** so the "stealing" magnitudes (≈0.4 expected votes per game, ~28pp of P(3)) are directly interpretable.

## Blockers / Follow-ups

- None. Optional follow-up: link the methodology doc from the README.
