"""Repository paths and file locations."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
RESULT_DIR = ROOT / "result"
BACKCAST_DIR = ROOT / "backcast"

PLAYER_STATS_CSV = DATA_DIR / "player_stats_2012_2026_fitzroy.csv"
TEAM_STATS_CSV = DATA_DIR / "team_stats_2012_2026_fitzroy.csv"
BROWNLOW_VOTES_CSV = DATA_DIR / "brownlow_stats_2012_2025_fitzroy.csv"
PLAYER_DETAILS_CSV = DATA_DIR / "player_details_2012_2026_afl.csv"


def ensure_dirs() -> None:
    """Create output directories that are not tracked in git."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
