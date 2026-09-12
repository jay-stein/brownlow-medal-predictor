# Session: MIT License and Data Attribution

**Date:** 2026-09-12
**Branch:** `agent/license-mit`
**Commit:** `docs: add MIT license and data-source attribution`

## Goal

Add an MIT license for the code and document the third-party data sources, including the fitzRoy package.

## Files Changed

| File | Change |
|---|---|
| `LICENSE` | New — MIT license (Copyright 2026 Jay Stein) plus a data sources and attribution note explaining that provider data remains subject to its own terms |
| `README.md` | New "Licence and attribution" section expanding the credits: fitzRoy, AFL API (Champion Data), AFL Tables, AFL.com.au Brownlow endpoint, Betfair, ESPN, WheeloRatings |
| `docs/agent-sessions/2026-09-12-mit-license.md` | This summary |

## Commands Executed

- `git checkout -b agent/license-mit`
- `uv run pytest` and `uv run ruff check .` (documentation-only change; ran anyway)
- `gh pr create` + `gh pr merge --merge`

## Important Decisions

1. **Pure MIT text with a clearly separated attribution section**: the license grant is unmodified MIT; the data note sits below a horizontal rule and explicitly says it is not part of the license.
2. **Attribution lists actual, current sources** rather than the older generic wording, and credits fitzRoy by name as the tool behind both the AFL API and AFL Tables extracts.
3. **Snapshot vs raw data**: the note clarifies that small committed prediction snapshots exist for reproducibility while the large raw extracts are regenerated locally.

## Blockers / Follow-ups

- None. The licence is recognised automatically by GitHub once merged to `main`.
