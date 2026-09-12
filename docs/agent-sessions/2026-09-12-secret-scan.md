# Session: Full-History Secret Scan

**Date:** 2026-09-12
**Branch:** `agent/secret-scan-summary`

## Goal

Verify that the now-public repository contains no API keys, tokens, passwords or other credentials in the working tree, in git history, or in deleted files.

## What Was Checked

| Check | Tool | Result |
|---|---|---|
| Full git history (21 commits, 25.6 MB of blobs) | `gitleaks 8.30.1` (`gitleaks git`) | **no leaks found** |
| Worktree including untracked directories | `gitleaks dir` | 11 hits, all inside `.venv/Lib/site-packages/...` vendored dependency tests (`key_sha256`, `account_key`, `footer_key`, `data_key_lo/hi`) — not tracked by git |
| All 134 text blobs in history | custom scan with 16 named rules: GitHub/AWS/Google/Slack/Discord/Stripe/OpenAI/Anthropic/HuggingFace/SendGrid/Twilio/Heroku tokens, JWTs, private-key blocks, basic-auth URLs, connection strings, `KEY/SECRET/TOKEN/PASSWORD=` assignments, generic `api_key =` patterns | **0 named-pattern hits** |
| Entropy heuristic (Shannon > 3.8 on keyword lines, images excluded) | custom scan | 11 hits, all code identifiers (`paths.PROCESSED_...`, `pd.MultiIndex.fr...`, test names) and `web/package-lock.json` npm integrity hashes |
| File names ever added to history | `git log --diff-filter=A` | no `.env`, `.pem`, `id_rsa`, credentials, token or secrets files |
| Tracked files with sensitive-looking names | `git ls-files` | none |
| `.venv`, `web/node_modules`, `data/processed`, `web/dist` | `git ls-files` | not tracked |
| GitHub secret scanning (provider patterns) | GitHub API | **enabled** |
| GitHub push protection | GitHub API | **enabled** |

## Files Changed

| File | Change |
|---|---|
| `docs/agent-sessions/2026-09-12-secret-scan.md` | This summary |

No code or data changes were required.

## Important Decisions

1. **No history rewrite needed**: nothing sensitive was found, so no `filter-repo` or force-push was warranted.
2. **GitHub secret scanning and push protection enabled** on the public repository after the scan. Non-provider patterns and validity checks require GitHub Secret Protection (paid) and remain disabled.
3. **Expected false positives documented**: npm integrity hashes and vendored dependency test fixtures are not secrets.

## Residual Limitations

- Regex and entropy scanners cannot inspect binary images; the only committed images are generated matplotlib/seaborn plots.
- Secrets that were only ever present in local files or in other repositories were outside this repository's scope.
