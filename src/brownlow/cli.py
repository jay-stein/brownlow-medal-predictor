"""Command line entry points, e.g. ``uv run python -m brownlow.cli audit``."""

from __future__ import annotations

import argparse

from . import features, ingest, paths


def run_audit() -> None:
    """Build the crosswalk, audited labels and the processed dataset."""
    paths.ensure_dirs()

    player_stats = ingest.load_player_stats()
    team_stats = ingest.load_team_stats()
    votes = ingest.load_brownlow_votes()

    table = features.build_feature_table(player_stats, team_stats, train_through=2025)
    crosswalk = ingest.build_crosswalk(player_stats, votes)
    labelled, audit = ingest.attach_labels(table, votes, crosswalk=crosswalk)

    crosswalk.to_csv(paths.PROCESSED_DIR / "player_crosswalk.csv", index=False)
    audit["by_season"].to_csv(paths.PROCESSED_DIR / "label_audit_by_season.csv", index=False)
    audit["crosswalk_summary"].to_csv(paths.PROCESSED_DIR / "crosswalk_summary.csv", index=False)
    audit["unmatched"].to_csv(paths.PROCESSED_DIR / "unmatched_players.csv", index=False)
    audit["ambiguous"].to_csv(paths.PROCESSED_DIR / "ambiguous_players.csv", index=False)
    labelled.to_parquet(paths.PROCESSED_DIR / "labelled_player_games.parquet", index=False)

    print("=== Crosswalk summary ===")
    print(audit["crosswalk_summary"].to_string(index=False))
    print("\n=== Label audit by season ===")
    print(audit["by_season"].to_string(index=False))
    print(f"\nUnmatched AFL Tables players: {len(audit['unmatched'])}")
    print(f"Ambiguous AFL Tables players: {len(audit['ambiguous'])}")
    print(f"\nWrote processed files to {paths.PROCESSED_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="brownlow")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("audit", help="build crosswalk, audited labels and processed parquet")
    args = parser.parse_args()

    if args.command == "audit":
        run_audit()


if __name__ == "__main__":
    main()
