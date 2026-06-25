"""Validate parametric synthetic old-Nomon profiles on held-out real sessions.

Run from the repository root:

    python -m User_Simulation.evaluation.evaluation_synthetic_validation --user-id A
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from User_Simulation.evaluation.evaluation_baseline import (
    REPO_ROOT,
    build_phrase_comparison,
    lm_files,
    load_text_click_data,
    load_text_phrase_data,
    normalize_phrase_order,
    write_json,
)
from User_Simulation.evaluation.metrics import summarize_real_run, summarize_simulated_run
from User_Simulation.evaluation.synthetic_profiles import (
    estimate_profile,
    generate_synthetic_click_df,
    select_sessions,
    split_sessions,
)
from User_Simulation.simulated_user_text import SimulatedUser

import numpy as np


def is_missing(value: Any) -> bool:
    return value is None or pd.isna(value)


def build_output_dir(base_dir: Path, user_id: str) -> Path:
    timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    output_dir = base_dir / f"synthetic_validation_user_{user_id}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def run_old_nomon_simulator(
    click_df: pd.DataFrame,
    phrase_df: pd.DataFrame,
    lm_size: str,
    verbose: bool,
) -> SimulatedUser:
    """Run old Nomon with the supplied click dataframe exactly as provided."""

    sim = SimulatedUser()
    params = {
        "click_df": click_df,
        "phrase_df": phrase_df,
        "lm_files": lm_files(lm_size),
    }

    original_cwd = Path.cwd()
    try:
        os.chdir(REPO_ROOT / "User_Simulation")
        sim.parameter_metrics(params, trials=1, verbose=verbose)
    finally:
        os.chdir(original_cwd)

    return sim


def build_summary_distribution(trial_summaries_df: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        column
        for column in trial_summaries_df.columns
        if column not in {"trial", "synthetic_click_rows", "simulated_clicks_used"}
    ]

    rows = []
    for metric in metric_columns:
        values = pd.to_numeric(trial_summaries_df[metric], errors="coerce").dropna()
        if values.empty:
            rows.append(
                {
                    "metric": metric,
                    "synthetic_mean": None,
                    "synthetic_std": None,
                    "synthetic_min": None,
                    "synthetic_max": None,
                    "synthetic_p025": None,
                    "synthetic_p975": None,
                }
            )
            continue

        rows.append(
            {
                "metric": metric,
                "synthetic_mean": float(values.mean()),
                "synthetic_std": float(values.std(ddof=0)),
                "synthetic_min": float(values.min()),
                "synthetic_max": float(values.max()),
                "synthetic_p025": float(values.quantile(0.025)),
                "synthetic_p975": float(values.quantile(0.975)),
            }
        )

    return pd.DataFrame(rows)


def build_validation_metric_comparison(
    validation_real_summary: dict[str, float | None],
    distribution_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in distribution_df.to_dict("records"):
        metric = row["metric"]
        real_value = validation_real_summary.get(metric)
        synthetic_mean = row["synthetic_mean"]

        if is_missing(real_value) or is_missing(synthetic_mean):
            absolute_error = None
            relative_error = None
        else:
            absolute_error = synthetic_mean - real_value
            relative_error = None if real_value == 0 else absolute_error / real_value

        p025 = row["synthetic_p025"]
        p975 = row["synthetic_p975"]
        if is_missing(real_value) or is_missing(p025) or is_missing(p975):
            real_within_95pct = None
        else:
            real_within_95pct = bool(p025 <= real_value <= p975)

        rows.append(
            {
                "metric": metric,
                "real_value": real_value,
                "synthetic_mean": synthetic_mean,
                "synthetic_std": row["synthetic_std"],
                "synthetic_p025": p025,
                "synthetic_p975": p975,
                "absolute_error": absolute_error,
                "relative_error": relative_error,
                "real_within_synthetic_95pct": real_within_95pct,
            }
        )

    return pd.DataFrame(rows)


def build_trial_phrase_comparison(
    real_phrase_df: pd.DataFrame,
    simulated_phrase_df: pd.DataFrame,
    trials: int,
) -> pd.DataFrame:
    """Build phrase comparison rows for every trial and held-out phrase."""

    real_columns = {
        "Phrase Num": "Original Phrase Num",
        "Phrase Text": "Real Target Phrase",
        "Typed Text": "Real Typed Text",
        "Entry Rate (wpm)": "Real Entry Rate (wpm)",
        "Click Load (clicks/character)": "Real Click Load (clicks/character)",
        "Correction Rate (% of selections)": "Real Correction Rate (%)",
        "Final Error Rate (%)": "Real Error Rate (%)",
    }
    sim_columns = {
        "Phrase Num": "Sim Phrase Num",
        "Target Phrase": "Sim Target Phrase",
        "Typed Text": "Sim Typed Text",
        "Entry Rate (wpm)": "Sim Entry Rate (wpm)",
        "Click Load (clicks/character)": "Sim Click Load (clicks/character)",
        "Correction Rate (%)": "Sim Correction Rate (%)",
        "Error Rate (%)": "Sim Error Rate (%)",
    }

    real_compare = real_phrase_df[["Session Num"] + list(real_columns)].rename(columns=real_columns)
    trial_df = pd.DataFrame({"Trial Num": list(range(trials))})
    real_compare = real_compare.merge(trial_df, how="cross")

    if simulated_phrase_df.empty:
        sim_compare = pd.DataFrame(
            columns=["Trial Num", "Session Num", "Original Phrase Num"] + list(sim_columns.values())
        )
    else:
        sim_compare = simulated_phrase_df.copy()
        if "Original Phrase Num" not in sim_compare:
            sim_compare["Original Phrase Num"] = pd.NA
        sim_compare = sim_compare[
            ["Trial Num", "Session Num", "Original Phrase Num"] + list(sim_columns)
        ].rename(columns=sim_columns)

    comparison_df = real_compare.merge(
        sim_compare,
        on=["Trial Num", "Session Num", "Original Phrase Num"],
        how="outer",
        indicator=True,
    )
    comparison_df["Alignment Status"] = comparison_df["_merge"].map(
        {
            "both": "matched",
            "left_only": "missing_simulated",
            "right_only": "extra_simulated",
        }
    )
    return (
        comparison_df.drop(columns=["_merge"])
        .sort_values(["Trial Num", "Session Num", "Original Phrase Num"], na_position="last")
        .reset_index(drop=True)
    )


def build_trial_alignment_summary(
    real_phrase_df: pd.DataFrame,
    simulated_phrase_df: pd.DataFrame,
    trials: int,
) -> dict[str, int]:
    real_keys = set(
        zip(
            real_phrase_df["Session Num"].astype(int),
            real_phrase_df["Phrase Num"].astype(int),
        )
    )
    expected_rows = len(real_phrase_df) * trials

    if simulated_phrase_df.empty or "Original Phrase Num" not in simulated_phrase_df:
        matched_rows = 0
        extra_rows = len(simulated_phrase_df)
        rows_with_original_phrase_num = 0
    else:
        rows_with_original_phrase_num = int(simulated_phrase_df["Original Phrase Num"].notna().sum())
        sim_keys = list(
            zip(
                simulated_phrase_df["Session Num"].astype(int),
                pd.to_numeric(simulated_phrase_df["Original Phrase Num"], errors="coerce"),
            )
        )
        matched_rows = int(
            sum((session, int(original)) in real_keys for session, original in sim_keys if not pd.isna(original))
        )
        extra_rows = int(
            sum((session, int(original)) not in real_keys for session, original in sim_keys if not pd.isna(original))
        )

    return {
        "trials": int(trials),
        "real_phrase_rows": int(len(real_phrase_df)),
        "expected_trial_phrase_rows": int(expected_rows),
        "simulated_phrase_rows": int(len(simulated_phrase_df)),
        "matched_rows": int(matched_rows),
        "missing_simulated_rows": int(max(expected_rows - matched_rows, 0)),
        "extra_simulated_rows": int(extra_rows),
        "simulated_rows_with_original_phrase_num": int(rows_with_original_phrase_num),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-id", default="A", help="OSF user id to validate, for example A or B.")
    parser.add_argument("--lm-size", default="tiny", choices=["tiny", "medium"], help="Bundled KenLM model size.")
    parser.add_argument("--trials", type=int, default=30, help="Number of synthetic validation trials.")
    parser.add_argument("--validation-fraction", type=float, default=0.2, help="Final-session fraction to hold out.")
    parser.add_argument("--clicks-per-phrase", type=int, default=500, help="Synthetic active clicks per phrase.")
    parser.add_argument("--calibration-clicks", type=int, default=200, help="Synthetic calibration clicks per trial.")
    parser.add_argument("--seed", type=int, default=12345, help="Base random seed.")
    parser.add_argument("--verbose", action="store_true", help="Print target/typed details from the simulator.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Directory where validation reports are written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.trials < 1:
        raise ValueError("--trials must be at least 1")
    if args.clicks_per_phrase < 1:
        raise ValueError("--clicks-per-phrase must be at least 1")

    real_click_df = load_text_click_data(args.user_id)
    real_phrase_df = normalize_phrase_order(load_text_phrase_data(args.user_id))

    train_sessions, validation_sessions = split_sessions(real_click_df, args.validation_fraction)
    train_click_df, train_phrase_df = select_sessions(real_click_df, real_phrase_df, train_sessions)
    validation_click_df, validation_phrase_df = select_sessions(real_click_df, real_phrase_df, validation_sessions)
    validation_phrase_df = normalize_phrase_order(validation_phrase_df)

    profile = estimate_profile(
        args.user_id,
        train_click_df,
        train_phrase_df,
        validation_click_df,
        validation_phrase_df,
        train_sessions,
        validation_sessions,
    )
    train_real_summary = summarize_real_run(train_click_df, train_phrase_df)
    validation_real_summary = summarize_real_run(validation_click_df, validation_phrase_df)

    simulated_results = []
    trial_summaries = []

    for trial in range(args.trials):
        rng = np.random.default_rng(args.seed + trial)
        synthetic_click_df = generate_synthetic_click_df(
            profile,
            validation_phrase_df,
            trial=trial,
            rng=rng,
            clicks_per_phrase=args.clicks_per_phrase,
            calibration_clicks=args.calibration_clicks,
        )

        sim = run_old_nomon_simulator(
            click_df=synthetic_click_df,
            phrase_df=validation_phrase_df,
            lm_size=args.lm_size,
            verbose=args.verbose,
        )

        if not sim.result_df.empty:
            trial_result_df = sim.result_df.copy()
            trial_result_df["Trial Num"] = trial
            trial_result_df["Synthetic Profile User ID"] = args.user_id
            simulated_results.append(trial_result_df)

        trial_summary = summarize_simulated_run(
            sim.result_df,
            clicks_used=getattr(sim, "num_clicks_total", None),
        )
        trial_summary["trial"] = trial
        trial_summary["synthetic_click_rows"] = int(len(synthetic_click_df))
        trial_summary["simulated_clicks_used"] = int(getattr(sim, "num_clicks_total", 0))
        trial_summaries.append(trial_summary)

    simulated_phrase_df = (
        pd.concat(simulated_results, ignore_index=True)
        if simulated_results
        else pd.DataFrame()
    )
    trial_summaries_df = pd.DataFrame(trial_summaries)
    distribution_df = build_summary_distribution(trial_summaries_df)
    validation_comparison_df = build_validation_metric_comparison(
        validation_real_summary,
        distribution_df,
    )
    phrase_comparison_df = build_trial_phrase_comparison(
        validation_phrase_df,
        simulated_phrase_df,
        args.trials,
    )
    alignment_summary = build_trial_alignment_summary(
        validation_phrase_df,
        simulated_phrase_df,
        args.trials,
    )

    output_dir = build_output_dir(args.output_dir, args.user_id)
    write_json(output_dir / "profile.json", profile)
    write_json(output_dir / "train_real_summary.json", train_real_summary)
    write_json(output_dir / "validation_real_summary.json", validation_real_summary)
    trial_summaries_df.to_csv(output_dir / "synthetic_trial_summaries.csv", index=False)
    distribution_df.to_csv(output_dir / "synthetic_summary_distribution.csv", index=False)
    validation_comparison_df.to_csv(output_dir / "validation_metric_comparison.csv", index=False)
    validation_phrase_df.to_csv(output_dir / "real_phrase_results.csv", index=False)
    simulated_phrase_df.to_csv(output_dir / "simulated_phrase_results.csv", index=False)
    phrase_comparison_df.to_csv(output_dir / "phrase_comparison.csv", index=False)
    write_json(output_dir / "alignment_summary.json", alignment_summary)
    write_json(
        output_dir / "run_config.json",
        {
            "user_id": args.user_id,
            "lm_size": args.lm_size,
            "trials": args.trials,
            "validation_fraction": args.validation_fraction,
            "clicks_per_phrase": args.clicks_per_phrase,
            "calibration_clicks": args.calibration_clicks,
            "seed": args.seed,
            "train_sessions": train_sessions,
            "validation_sessions": validation_sessions,
            "alignment_summary": alignment_summary,
        },
    )

    print(f"Saved synthetic validation report to: {output_dir}")
    print("Train sessions:", train_sessions)
    print("Validation sessions:", validation_sessions)
    print("Alignment summary:")
    print(json.dumps(alignment_summary, indent=2, sort_keys=True))
    print(validation_comparison_df.to_string(index=False))


if __name__ == "__main__":
    main()
