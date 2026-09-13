"""Command line entry points, e.g. ``uv run python -m brownlow.cli audit``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import (
    aflca,
    eligibility,
    evaluate,
    features,
    folds,
    history,
    ingest,
    model,
    paths,
    scores,
    simulate,
    validation,
)


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
    coaches_path = paths.DATA_DIR / "aflca_votes.csv"
    if coaches_path.exists():
        table, coaches_summary = aflca.attach_coaches_votes(table, coaches_path)
        print(
            "=== AFL Coaches Association votes ==="
            f"\nmatched {coaches_summary['matched_rows']} / {coaches_summary['rows']} rows "
            f"({coaches_summary['match_rate']:.1%}) to "
            f"{coaches_summary['matched_matches']} matches\n"
        )
    else:
        print("AFLCA votes not found: run `fetch-coaches` to populate COACH_VOTES\n")
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
    """Parse ``2020-2024``, ``2021,2023`` or mixed ``2015-2017,2026``."""
    seasons: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            seasons.extend(range(int(start), int(end) + 1))
        else:
            seasons.append(int(part))
    return seasons


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


def run_calibrate_joint(args: argparse.Namespace) -> None:
    """Score the joint (tau, effect scale) grid and select production parameters."""
    scores_by_season = _load_cached_scores(args.model)
    requested = parse_seasons(args.seasons)
    missing = [season for season in requested if season not in scores_by_season]
    if missing:
        raise SystemExit(
            f"no cached {args.model} scores for {missing}: "
            f"run `baseline --model {args.model} --seasons {missing[0]}` first"
        )

    tau_values = [float(part) for part in args.taus.split(",")]
    scales = [float(part) for part in args.scales.split(",")]
    grid = validation.joint_calibration_grid(
        scores_by_season,
        requested,
        tau_values,
        scales,
        n_sims=args.n_sims,
        n_draws=args.n_draws,
        contender_count=args.contenders,
        seed=args.seed,
    )

    output_dir = paths.PROCESSED_DIR / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    grid_path = output_dir / f"joint_grid_{args.model}.csv"
    grid.to_csv(grid_path, index=False)

    tune_seasons = parse_seasons(args.tune_seasons)
    selection = validation.select_joint_params(grid, tune_seasons)
    selection_path = output_dir / f"joint_selection_{args.model}.json"
    selection_path.write_text(
        json.dumps(
            {
                "tau": selection.tau,
                "effect_scale": selection.effect_scale,
                "model_key": args.model,
                "selection_rule": (
                    "equal-weight z(effect-integrated match NLL) + "
                    "z(contender CRPS), tie-break lower NLL"
                ),
                "tune_seasons": tune_seasons,
                "taus": tau_values,
                "scales": scales,
                "match_nll_optimum": {
                    "tau": selection.best_match_nll_tau,
                    "effect_scale": selection.best_match_nll_scale,
                },
                "crps_optimum": {
                    "tau": selection.best_crps_tau,
                    "effect_scale": selection.best_crps_scale,
                },
                "n_sims": args.n_sims,
                "n_draws": args.n_draws,
                "seed": args.seed,
            },
            indent=2,
        )
    )

    with pd.option_context("display.width", 200):
        print("=== selection summary over the evidence seasons ===")
        print(selection.summary.round(4).to_string(index=False))
    print(
        f"\nselected tau={selection.tau} scale={selection.effect_scale}"
        f" | match-NLL optimum tau={selection.best_match_nll_tau} "
        f"scale={selection.best_match_nll_scale}"
        f" | CRPS optimum tau={selection.best_crps_tau} scale={selection.best_crps_scale}"
    )
    print(f"\nWrote {grid_path} and {selection_path}")


def run_rolling_backtest(args: argparse.Namespace) -> None:
    """Fully rolling per-season validation from a saved joint grid."""
    grid_path = paths.PROCESSED_DIR / "evaluation" / f"joint_grid_{args.model}.csv"
    if not grid_path.exists():
        raise SystemExit(
            f"missing {grid_path}: run `calibrate-joint --model {args.model}` first"
        )
    grid = pd.read_csv(grid_path)
    table = validation.rolling_validation(
        grid,
        report_seasons=parse_seasons(args.report_seasons),
        min_evidence=args.min_evidence,
    )
    scores_by_season = _load_cached_scores(args.model)
    if not table.empty:
        for index, row in table.iterrows():
            season = int(row["season"])
            frame = scores_by_season.get(season)
            if frame is None:
                continue
            ineligible = eligibility.ineligible_ids(season)
            simulation = simulate.simulate_season(
                frame,
                float(row["tau_selected"]),
                effect_scale=float(row["scale_selected"]),
                n_sims=args.n_sims,
                seed=args.seed,
                ineligible=ineligible,
            )
            metrics = simulate.season_metrics(
                simulation, contender_count=args.contenders, ineligible=ineligible
            )
            table.loc[index, "favorite_won"] = metrics["favorite_won"]
            table.loc[index, "winner_probability"] = metrics["winner_probability"]
            table.loc[index, "winner_mean_rank"] = metrics["winner_mean_rank"]
            table.loc[index, "n_ineligible"] = len(ineligible)
    output_dir = paths.PROCESSED_DIR / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"rolling_backtest_{args.model}.csv"
    table.to_csv(output_path, index=False)

    columns = [column for column in validation.ROLLING_COLUMNS if column in table.columns]
    if not table.empty:
        with pd.option_context("display.width", 240, "display.max_columns", 40):
            print(table[columns].round(4).to_string(index=False))
        summary = {
            "seasons": len(table),
            "coverage_90_contenders": table["coverage_90_contenders"].mean(),
            "coverage_50_contenders": table["coverage_50_contenders"].mean(),
            "mean_crps_contenders": table["mean_crps_contenders"].mean(),
            "favorites_won": int(table["favorite_won"].sum()),
        }
        print(f"\nrolling summary: {summary}")
    print(f"\nWrote {output_path}")


def run_player_effects(args: argparse.Namespace) -> None:
    """Grid-search partially pooled historical player effects."""
    scores_by_season = _load_cached_scores(args.model)
    requested = parse_seasons(args.seasons)
    missing = [season for season in requested if season not in scores_by_season]
    if missing:
        raise SystemExit(
            f"no cached {args.model} scores for {missing}: "
            f"run `baseline --model {args.model} --seasons {missing[0]}` first"
        )

    tau = args.tau
    effect_scale = args.scale
    joint_path = paths.PROCESSED_DIR / "evaluation" / f"joint_selection_{args.model}.json"
    if (tau is None or effect_scale is None) and joint_path.exists():
        joint = json.loads(joint_path.read_text())
        tau = float(joint["tau"]) if tau is None else tau
        effect_scale = float(joint["effect_scale"]) if effect_scale is None else effect_scale
    if tau is None or effect_scale is None:
        raise SystemExit("pass --tau and --scale, or run `calibrate-joint` first")

    residual_history = history.player_residual_history(scores_by_season, float(tau))
    shrinkages = [float(part) for part in args.shrinkages.split(",")]
    mappings = [float(part) for part in args.mappings.split(",")]
    half_lives = [float(part) for part in args.half_lives.split(",")]
    candidates = [(k, m, h) for k in shrinkages for m in mappings for h in half_lives]

    grid = validation.player_effect_grid(
        scores_by_season,
        residual_history,
        requested,
        tau=float(tau),
        effect_scale=float(effect_scale),
        candidates=candidates,
        n_sims=args.n_sims,
        n_draws=args.n_draws,
        contender_count=args.contenders,
        seed=args.seed,
    )
    output_dir = paths.PROCESSED_DIR / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    grid_path = output_dir / f"player_effect_grid_{args.model}.csv"
    grid.to_csv(grid_path, index=False)

    table = validation.rolling_player_effect_validation(
        grid,
        report_seasons=parse_seasons(args.report_seasons),
        min_evidence=args.min_evidence,
    )
    table_path = output_dir / f"rolling_player_effects_{args.model}.csv"
    table.to_csv(table_path, index=False)

    if not table.empty:
        columns = [column for column in validation.PLAYER_EFFECT_COLUMNS if column in table.columns]
        with pd.option_context("display.width", 240, "display.max_columns", 40):
            print("=== rolling selection with paired baseline ===")
            print(table[columns].round(4).to_string(index=False))
        improved = int((table["delta_crps_contenders"] < 0).sum())
        print(f"\nseasons where the adjustment improves contender CRPS: {improved}/{len(table)}")
        print(f"mean delta contender CRPS: {table['delta_crps_contenders'].mean():.4f}")

    selection = validation.select_player_effects(grid, requested)
    selection_path = output_dir / f"player_effect_selection_{args.model}.json"
    selection_path.write_text(
        json.dumps(
            {
                "shrinkage": selection.shrinkage,
                "mapping": selection.mapping,
                "half_life": selection.half_life,
                "tau": float(tau),
                "effect_scale": float(effect_scale),
                "model_key": args.model,
                "selection_rule": "equal-weight z(match NLL) + z(contender CRPS)",
                "tune_seasons": requested,
            },
            indent=2,
        )
    )
    print(
        f"\nproduction selection over {requested[0]}-{requested[-1]}: "
        f"shrinkage={selection.shrinkage} mapping={selection.mapping} "
        f"half_life={selection.half_life}"
    )
    print(f"\nWrote {grid_path}, {table_path} and {selection_path}")


def run_fetch_coaches(args: argparse.Namespace) -> None:
    """Scrape AFL Coaches Association per-match votes for the given seasons."""
    seasons = parse_seasons(args.seasons)
    frames = []
    for season in seasons:
        print(f"=== {season} ===")
        frame = aflca.fetch_season(season, pause=args.pause)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=False)
    print(f"\nwrote {len(combined)} rows to {output}")


def run_forecast(args: argparse.Namespace) -> None:
    """Simulate a season's count and report vote spreads and win probabilities."""
    season = args.season
    model_key = args.model
    labelled = _load_labelled()

    cached = scores.load_scores(season, model_key)
    if cached is None or args.force:
        print(f"training {model_key} scores for {season} (this trains on all earlier labels)...")
        result = scores.train_season_scores(
            labelled,
            season,
            model_key=model_key,
            n_splits=args.n_splits,
            num_boost_round=args.max_rounds,
            early_stopping_rounds=args.early_stopping,
            seed=args.seed,
        )
        scores.save_scores(result)
        metadata = result.metadata
        print(
            f"  objective={metadata['objective']} best_round={metadata['best_round']} "
            f"train rows={metadata['n_train_rows']}"
        )
        cached = result
    frame = cached.frame

    prior_frames = [
        scores.load_scores(candidate, model_key).frame
        for candidate in scores.cached_seasons(model_key)
        if candidate < season
    ]
    tau, tau_matches = evaluate.fit_pooled_tau(prior_frames)
    tau_source = f"leak-free MLE on {tau_matches} earlier matches"

    joint_path = paths.PROCESSED_DIR / "evaluation" / f"joint_selection_{model_key}.json"
    calibration_path = paths.PROCESSED_DIR / "evaluation" / f"calibrated_effect_scale_{model_key}.json"
    if args.effect_scale is not None:
        effect_scale = args.effect_scale
    elif joint_path.exists():
        joint = json.loads(joint_path.read_text())
        tau = float(joint["tau"])
        effect_scale = float(joint["effect_scale"])
        tau_source = (
            f"joint calibration on seasons {joint['tune_seasons'][0]}-{joint['tune_seasons'][-1]}"
        )
    elif calibration_path.exists():
        effect_scale = float(json.loads(calibration_path.read_text())["effect_scale"])
    else:
        effect_scale = 0.0

    player_effect_path = (
        paths.PROCESSED_DIR / "evaluation" / f"player_effect_selection_{model_key}.json"
    )
    player_effect = None
    if player_effect_path.exists():
        player_effect = json.loads(player_effect_path.read_text())
        prior = {
            candidate: scores.load_scores(candidate, model_key).frame
            for candidate in scores.cached_seasons(model_key)
            if candidate < season
        }
        residual_history = history.player_residual_history(prior, tau)
        frame = history.attach_player_effects(
            frame,
            residual_history,
            season,
            shrinkage=float(player_effect["shrinkage"]),
            mapping=float(player_effect["mapping"]),
            half_life=float(player_effect["half_life"]),
        )
        print(
            f"  player effects: shrinkage={player_effect['shrinkage']} "
            f"mapping={player_effect['mapping']} half_life={player_effect['half_life']} "
            f"({len(residual_history)} prior player-seasons)"
        )

    ineligible = eligibility.ineligible_ids(season)
    if ineligible:
        print(f"  eligibility: {len(ineligible)} suspended players excluded from the medal")

    simulation = simulate.simulate_season(
        frame,
        tau,
        effect_scale=effect_scale,
        n_sims=args.n_sims,
        seed=args.seed,
        track_rounds=args.web_json is not None,
        path_count=args.web_paths,
        ineligible=ineligible,
    )
    players = simulation.players.sort_values("sim_mean", ascending=False).reset_index(drop=True)

    scales = [float(part) for part in args.sensitivity_scales.split(",")]
    sensitivity = None
    if len(scales) > 1:
        parts = []
        for scale in scales:
            scenario = simulate.simulate_season(
                frame,
                tau,
                effect_scale=scale,
                n_sims=args.n_sims,
                seed=args.seed,
                ineligible=ineligible,
            )
            subset = scenario.players[
                [
                    "PLAYER_PLAYER_PLAYER_PLAYERID",
                    "FULL_NAME",
                    "sim_mean",
                    "p_outright_first",
                    "p_first_or_joint",
                    "p_top5",
                ]
            ].copy()
            subset["effect_scale"] = scale
            parts.append(subset)
        combined = pd.concat(parts, ignore_index=True)
        sensitivity = (
            combined.groupby(["PLAYER_PLAYER_PLAYER_PLAYERID", "FULL_NAME"], as_index=False)
            .agg(
                mean_votes_min=("sim_mean", "min"),
                mean_votes_max=("sim_mean", "max"),
                p_first_or_joint_min=("p_first_or_joint", "min"),
                p_first_or_joint_max=("p_first_or_joint", "max"),
            )
        )

    output_dir = paths.PROCESSED_DIR / "forecast"
    output_dir.mkdir(parents=True, exist_ok=True)
    players.to_csv(output_dir / f"forecast_{season}_{model_key}.csv", index=False)
    if sensitivity is not None:
        sensitivity.to_csv(output_dir / f"forecast_{season}_{model_key}_sensitivity.csv", index=False)

    columns = [
        "FULL_NAME",
        "TEAM_NAME",
        "sim_mean",
        "sim_median",
        "sim_q05",
        "sim_q95",
        "p_outright_first",
        "p_first_or_joint",
        "p_top5",
    ]
    if sensitivity is not None:
        players = players.merge(
            sensitivity[
                [
                    "PLAYER_PLAYER_PLAYER_PLAYERID",
                    "mean_votes_min",
                    "mean_votes_max",
                    "p_first_or_joint_min",
                    "p_first_or_joint_max",
                ]
            ],
            on="PLAYER_PLAYER_PLAYER_PLAYERID",
            how="left",
        )
        columns += ["mean_votes_min", "mean_votes_max", "p_first_or_joint_min", "p_first_or_joint_max"]

    print(
        f"\n=== {season} Brownlow forecast ({model_key}, tau={tau:.3f} from {tau_source}, "
        f"effect scale={effect_scale}, {args.n_sims} simulations) ===\n"
    )
    with pd.option_context("display.width", 240, "display.max_columns", 40):
        print(players.head(args.top)[columns].round(3).to_string(index=False))
    print(f"\nWrote {output_dir / f'forecast_{season}_{model_key}.csv'}")

    if args.web_json is not None:
        payload = simulate.forecast_export(
            simulation,
            players,
            top=args.web_players,
            metadata={
                "season": season,
                "model": model_key,
                "tau": round(tau, 4),
                "tauMatches": tau_matches,
                "tauSource": tau_source,
                "effectScale": effect_scale,
                "playerEffect": player_effect,
                "nIneligible": len(ineligible),
                "nSims": args.n_sims,
                "generated": pd.Timestamp.now().strftime("%Y-%m-%d"),
                "sensitivityScales": scales,
            },
        )
        web_path = Path(args.web_json)
        web_path.parent.mkdir(parents=True, exist_ok=True)
        web_path.write_text(json.dumps(payload, separators=(",", ":")))
        print(f"Wrote web visualisation data to {web_path}")


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

    joint = subparsers.add_parser(
        "calibrate-joint", help="joint (tau, effect scale) grid with integrated match NLL"
    )
    joint.add_argument("--seasons", default="2015-2025", help="seasons scored in the grid")
    joint.add_argument(
        "--tune-seasons", default="2015-2025", help="evidence seasons used for selection"
    )
    joint.add_argument("--taus", default="0.6,0.7,0.8,0.9,1.0,1.1")
    joint.add_argument("--scales", default="0,0.1,0.2,0.3,0.4,0.5,0.6")
    joint.add_argument("--n-sims", type=int, default=2000)
    joint.add_argument("--n-draws", type=int, default=256)
    joint.add_argument("--contenders", type=int, default=15)
    joint.add_argument("--seed", type=int, default=42)
    joint.add_argument("--model", choices=sorted(model.MODEL_PARAMS), default=model.RANKING)

    rolling = subparsers.add_parser(
        "rolling-backtest", help="fully rolling per-season validation from a saved joint grid"
    )
    rolling.add_argument("--report-seasons", default="2018-2025")
    rolling.add_argument("--min-evidence", type=int, default=3)
    rolling.add_argument("--n-sims", type=int, default=2000)
    rolling.add_argument("--contenders", type=int, default=15)
    rolling.add_argument("--seed", type=int, default=42)
    rolling.add_argument("--model", choices=sorted(model.MODEL_PARAMS), default=model.RANKING)

    player_effects = subparsers.add_parser(
        "player-effects", help="grid-search partially pooled historical player effects"
    )
    player_effects.add_argument("--seasons", default="2015-2025", help="seasons scored in the grid")
    player_effects.add_argument("--report-seasons", default="2018-2025")
    player_effects.add_argument("--min-evidence", type=int, default=3)
    player_effects.add_argument("--tau", type=float, default=None, help="default: joint selection")
    player_effects.add_argument("--scale", type=float, default=None, help="default: joint selection")
    player_effects.add_argument("--shrinkages", default="20,50,100,200")
    player_effects.add_argument("--mappings", default="0,0.5,1,2,4")
    player_effects.add_argument("--half-lives", default="0,5")
    player_effects.add_argument("--n-sims", type=int, default=2000)
    player_effects.add_argument("--n-draws", type=int, default=256)
    player_effects.add_argument("--contenders", type=int, default=15)
    player_effects.add_argument("--seed", type=int, default=42)
    player_effects.add_argument("--model", choices=sorted(model.MODEL_PARAMS), default=model.RANKING)

    coaches = subparsers.add_parser(
        "fetch-coaches", help="scrape AFL Coaches Association per-match votes"
    )
    coaches.add_argument("--seasons", default="2012-2026")
    coaches.add_argument("--out", default="data/aflca_votes.csv")
    coaches.add_argument("--pause", type=float, default=1.0, help="seconds between requests")

    forecast = subparsers.add_parser(
        "forecast", help="simulate a season's count and report probabilities"
    )
    forecast.add_argument("--season", type=int, default=folds.FORECAST_SEASON)
    forecast.add_argument("--model", choices=sorted(model.MODEL_PARAMS), default=model.RANKING)
    forecast.add_argument("--n-sims", type=int, default=10000)
    forecast.add_argument("--effect-scale", type=float, default=None)
    forecast.add_argument("--sensitivity-scales", default="0,0.2,0.4")
    forecast.add_argument("--top", type=int, default=20)
    forecast.add_argument("--web-json", default=None, help="write JSON for the interactive web app")
    forecast.add_argument("--web-players", type=int, default=60)
    forecast.add_argument("--web-paths", type=int, default=60)
    forecast.add_argument("--force", action="store_true", help="retrain scores even when cached")
    forecast.add_argument("--n-splits", type=int, default=5)
    forecast.add_argument("--max-rounds", type=int, default=3000)
    forecast.add_argument("--early-stopping", type=int, default=100)
    forecast.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    if args.command == "audit":
        run_audit()
    elif args.command == "baseline":
        run_baseline(args)
    elif args.command == "evaluate":
        run_evaluate(args)
    elif args.command == "calibrate-effects":
        run_calibrate_effects(args)
    elif args.command == "calibrate-joint":
        run_calibrate_joint(args)
    elif args.command == "rolling-backtest":
        run_rolling_backtest(args)
    elif args.command == "player-effects":
        run_player_effects(args)
    elif args.command == "fetch-coaches":
        run_fetch_coaches(args)
    elif args.command == "forecast":
        run_forecast(args)


if __name__ == "__main__":
    main()
