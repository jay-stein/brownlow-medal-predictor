"""Command line entry points, e.g. ``uv run python -m brownlow.cli audit``."""

from __future__ import annotations

import argparse
import json

import pandas as pd

from . import evaluate, features, ingest, model, paths, scores, simulate


def run_audit() -> None:
    """Build the crosswalk, audited labels and the processed dataset."""
    paths.ensure_dirs()

    player_stats = ingest.load_player_stats()
    team_stats = ingest.load_team_stats()
    votes = ingest.load_brownlow_votes()
    player_details = ingest.load_player_details()

    table = features.build_feature_table(
        player_stats, team_stats, player_details=player_details, train_through=2025
    )
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


def parse_seasons(value: str) -> list[int]:
    """Parse ``2020-2024`` or ``2021,2023`` into a season list."""
    if "-" in value:
        start, end = value.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(part) for part in value.split(",")]


def _load_labelled() -> pd.DataFrame:
    path = paths.PROCESSED_DIR / "labelled_player_games.parquet"
    if not path.exists():
        raise SystemExit("labelled dataset missing: run `python -m brownlow.cli audit` first")
    return pd.read_parquet(path)


def run_baseline(args: argparse.Namespace) -> None:
    """Generate (and cache) chronological out-of-sample scores per season."""
    labelled = _load_labelled()
    for season in parse_seasons(args.seasons):
        cached = scores.load_scores(season, args.model)
        if cached is not None and not args.force:
            metadata = cached.metadata
            print(
                f"{args.model} {season}: cached | best_round={metadata['best_round']} | "
                f"eval rows={metadata['n_eval_rows']}"
            )
            continue
        result = scores.train_season_scores(
            labelled,
            season,
            model_key=args.model,
            n_splits=args.n_splits,
            num_boost_round=args.max_rounds,
            early_stopping_rounds=args.early_stopping,
            seed=args.seed,
        )
        scores.save_scores(result)
        metadata = result.metadata
        fold_scores = [round(value, 4) for value in metadata["fold_scores"]]
        print(
            f"{args.model} {season}: trained | objective={metadata['objective']} | "
            f"best_round={metadata['best_round']} | "
            f"fold_rounds={metadata['fold_best_iterations']} | fold_scores={fold_scores} | "
            f"train rows={metadata['n_train_rows']}"
        )


def _load_cached_scores(model_key: str) -> dict[int, pd.DataFrame]:
    scores_by_season: dict[int, pd.DataFrame] = {}
    for season in scores.cached_seasons(model_key):
        cached = scores.load_scores(season, model_key)
        if cached is not None:
            scores_by_season[season] = cached.frame
    return scores_by_season


def run_evaluate(args: argparse.Namespace) -> None:
    """Fit leak-free temperatures and report allocation metrics."""
    requested = parse_seasons(args.seasons)
    scores_by_season = _load_cached_scores(args.model)
    missing = [season for season in requested if season not in scores_by_season]
    if missing:
        raise SystemExit(
            f"no cached {args.model} scores for {missing}: "
            f"run `baseline --model {args.model} --seasons {missing[0]}` first"
        )

    metrics, reliability = evaluate.evaluate_backtests(scores_by_season, seasons=requested)
    output_dir = paths.PROCESSED_DIR / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / f"allocation_metrics_{args.model}.csv", index=False)
    reliability.to_csv(output_dir / f"reliability_{args.model}.csv", index=False)

    with pd.option_context("display.width", 220, "display.max_columns", 60):
        print(metrics.round(4).to_string(index=False))
    print(f"\nWrote evaluation files to {output_dir}")


def run_calibrate_effects(args: argparse.Namespace) -> None:
    """Grid-search the player-season effect scale on historical season totals."""
    scores_by_season = _load_cached_scores(args.model)
    requested = parse_seasons(args.seasons)
    missing = [season for season in requested if season not in scores_by_season]
    if missing:
        raise SystemExit(
            f"no cached {args.model} scores for {missing}: "
            f"run `baseline --model {args.model} --seasons {missing[0]}` first"
        )

    taus = {season: info["tau"] for season, info in evaluate.leak_free_taus(scores_by_season).items()}
    scales = [float(part) for part in args.scales.split(",")]
    tune_seasons = parse_seasons(args.tune_seasons)

    grid = simulate.calibrate_effect_scales(
        scores_by_season,
        taus,
        scales,
        requested,
        n_sims=args.n_sims,
        seed=args.seed,
        contender_count=args.contenders,
    )
    output_dir = paths.PROCESSED_DIR / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    grid.to_csv(output_dir / f"effect_scale_grid_{args.model}.csv", index=False)

    summary = (
        grid[grid["season"].isin(tune_seasons)]
        .groupby("effect_scale")
        .agg(
            mean_crps=("mean_crps", "mean"),
            mean_crps_contenders=("mean_crps_contenders", "mean"),
            coverage_50=("coverage_50", "mean"),
            coverage_90=("coverage_90", "mean"),
        )
        .reset_index()
    )
    best_scale = float(summary.sort_values("mean_crps_contenders").iloc[0]["effect_scale"])
    summary.to_csv(output_dir / f"effect_scale_summary_{args.model}.csv", index=False)
    (output_dir / f"calibrated_effect_scale_{args.model}.json").write_text(
        json.dumps(
            {
                "effect_scale": best_scale,
                "model_key": args.model,
                "selection_metric": "mean_crps_contenders",
                "tune_seasons": tune_seasons,
                "scales": scales,
                "n_sims": args.n_sims,
                "seed": args.seed,
            },
            indent=2,
        )
    )

    with pd.option_context("display.width", 220, "display.max_columns", 60):
        print("=== tuning grid (mean over tune seasons) ===")
        print(summary.round(4).to_string(index=False))
        print(f"\nselected effect scale: {best_scale} (minimum contender CRPS)")
        print("\n=== per-season metrics at the selected scale ===")
        print(grid[grid["effect_scale"] == best_scale].round(4).to_string(index=False))

    report_seasons = [season for season in requested if season not in tune_seasons]
    if report_seasons:
        report = report_seasons[-1]
        simulation = simulate.simulate_season(
            scores_by_season[report],
            taus[report],
            effect_scale=best_scale,
            n_sims=args.n_sims,
            seed=args.seed,
        )
        simulation.players.to_csv(
            output_dir / f"season_simulation_{args.model}_{report}.csv", index=False
        )
        top = simulation.players.sort_values("sim_mean", ascending=False).head(10)
        columns = [
            "FULL_NAME",
            "TEAM_NAME",
            "observed_votes",
            "sim_mean",
            "sim_median",
            "sim_q05",
            "sim_q95",
            "p_outright_first",
            "p_first_or_joint",
        ]
        with pd.option_context("display.width", 220):
            print(f"\n=== {report} simulation at scale {best_scale} (top 10 by mean) ===")
            print(top[columns].round(3).to_string(index=False))
        print(f"\nWrote {report} player simulation to {output_dir}")
    else:
        print(f"\nWrote effect-scale outputs to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="brownlow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("audit", help="build crosswalk, audited labels and processed parquet")

    baseline = subparsers.add_parser(
        "baseline", help="train per-season baseline scores (chronological, cached)"
    )
    baseline.add_argument("--seasons", default="2020-2024", help="season or range, e.g. 2020-2024")
    baseline.add_argument("--model", choices=sorted(model.MODEL_PARAMS), default=model.REGRESSION)
    baseline.add_argument("--force", action="store_true", help="retrain even when cached")
    baseline.add_argument("--n-splits", type=int, default=5)
    baseline.add_argument("--max-rounds", type=int, default=3000)
    baseline.add_argument("--early-stopping", type=int, default=100)
    baseline.add_argument("--seed", type=int, default=42)

    evaluation = subparsers.add_parser("evaluate", help="leak-free tau fitting and allocation metrics")
    evaluation.add_argument(
        "--seasons",
        default="2020-2024",
        help="evaluation seasons; earlier cached seasons are used to fit tau",
    )
    evaluation.add_argument("--model", choices=sorted(model.MODEL_PARAMS), default=model.REGRESSION)

    effects = subparsers.add_parser(
        "calibrate-effects", help="grid-search the player-season effect scale"
    )
    effects.add_argument("--seasons", default="2021-2024", help="seasons to simulate")
    effects.add_argument("--tune-seasons", default="2021-2023", help="seasons used for selection")
    effects.add_argument("--scales", default="0,0.02,0.05,0.1,0.2,0.4")
    effects.add_argument("--n-sims", type=int, default=1000)
    effects.add_argument("--contenders", type=int, default=15)
    effects.add_argument("--seed", type=int, default=42)
    effects.add_argument("--model", choices=sorted(model.MODEL_PARAMS), default=model.REGRESSION)

    args = parser.parse_args()
    if args.command == "audit":
        run_audit()
    elif args.command == "baseline":
        run_baseline(args)
    elif args.command == "evaluate":
        run_evaluate(args)
    elif args.command == "calibrate-effects":
        run_calibrate_effects(args)


if __name__ == "__main__":
    main()
