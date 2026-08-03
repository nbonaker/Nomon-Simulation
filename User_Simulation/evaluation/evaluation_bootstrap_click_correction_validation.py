"""Validate bootstrapped OG Nomon clicks/corrections with zero synthetic dead time.

Run from the repository root:

    python -m User_Simulation.evaluation.evaluation_bootstrap_click_correction_validation --user-id C

This evaluator intentionally ignores WPM and other elapsed-time metrics. It
checks whether the regime-aware selection bootstrap reproduces the real user's
click and correction burden on held-out OG Nomon phrases.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from User_Simulation.evaluation.evaluation_baseline import (
    DEFAULT_LM_CACHE_DIR,
    REPO_ROOT,
    lm_parameters,
    load_text_click_data,
    load_text_phrase_data,
    normalize_phrase_order,
    write_json,
)
from User_Simulation.evaluation.synthetic_profiles import (
    CLICK_OFFSET_ALIAS_COL,
    CLICK_OFFSET_COL,
    CLOCK_PERIOD_COL,
    DEAD_TIME_COL,
    SELECTION_COL,
    SELECTION_GROUP_COLS,
    classify_selection,
    estimate_profile,
    generate_synthetic_click_df,
    select_sessions,
    split_sessions,
)
from User_Simulation.simulated_user_text import SimulatedUser


CLICK_CORRECTION_METRICS = [
    "avg_clicks_per_phrase",
    "avg_selections_per_phrase",
    "avg_corrections_per_phrase",
    "correction_rate_percent",
    "phrase_completion_rate",
]
MIN_VALID_PHRASE_COMPLETION_RATE = 0.95


def build_output_dir(base_dir: Path, user_id: str) -> Path:
    timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    output_dir = base_dir / f"bootstrap_click_correction_validation_user_{user_id}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def _safe_rate(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def zero_active_dead_time(click_df: pd.DataFrame) -> pd.DataFrame:
    """Set synthetic active click dead time to zero while preserving calibration NaNs."""

    result = click_df.copy()
    active_mask = result["Session Num"].notna()
    result.loc[active_mask, DEAD_TIME_COL] = 0.0
    return result


def _real_phrase_completed(phrase_row: pd.Series) -> bool:
    if "Final Error Rate (%)" not in phrase_row or pd.isna(phrase_row["Final Error Rate (%)"]):
        return True
    return float(phrase_row["Final Error Rate (%)"]) == 0.0


def build_real_phrase_click_correction(
    click_df: pd.DataFrame,
    phrase_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build phrase-level real click/correction rows from raw validation logs."""

    rows: list[dict[str, Any]] = []
    for phrase_row in normalize_phrase_order(phrase_df).to_dict("records"):
        session_num = int(phrase_row["Session Num"])
        phrase_num = int(phrase_row["Phrase Num"])
        phrase_clicks = click_df[
            (click_df["Session Num"] == session_num)
            & (click_df["Phrase Num"] == phrase_num)
        ].copy()

        num_clicks = int(len(phrase_clicks))
        num_selections = 0
        num_corrections = 0
        if not phrase_clicks.empty and set(SELECTION_GROUP_COLS).issubset(phrase_clicks.columns):
            for _key, group_df in phrase_clicks.groupby(SELECTION_GROUP_COLS, dropna=False):
                num_selections += 1
                final_selection = None
                if SELECTION_COL in group_df:
                    selections = group_df[SELECTION_COL].dropna()
                    if not selections.empty:
                        final_selection = selections.iloc[-1]
                if classify_selection(final_selection) == "correction":
                    num_corrections += 1

        rows.append(
            {
                "Session Num": session_num,
                "Original Phrase Num": phrase_num,
                "Phrase Text": phrase_row.get("Phrase Text"),
                "Typed Text": phrase_row.get("Typed Text"),
                "Num Clicks": num_clicks,
                "Num Selections": num_selections,
                "Num Corrections": num_corrections,
                "Correction Rate (%)": _safe_rate(num_corrections, num_selections) * 100,
                "Phrase Completed": _real_phrase_completed(pd.Series(phrase_row)),
            }
        )

    return pd.DataFrame(rows)


def build_synthetic_phrase_click_correction(result_df: pd.DataFrame) -> pd.DataFrame:
    if result_df.empty:
        return pd.DataFrame()

    output = result_df.copy()
    if "Original Phrase Num" not in output and "Phrase Num" in output:
        output["Original Phrase Num"] = output["Phrase Num"]

    keep_columns = [
        "Trial Num",
        "Session Num",
        "Phrase Num",
        "Original Phrase Num",
        "Target Phrase",
        "Typed Text",
        "Num Clicks",
        "Num Selections",
        "Num Corrections",
        "Correction Rate (%)",
        "Phrase Completed",
        "Completion Fraction",
    ]
    available_columns = [column for column in keep_columns if column in output.columns]
    return output[available_columns].copy()


def summarize_phrase_click_correction(
    phrase_df: pd.DataFrame,
    expected_phrase_count: int,
    trial: int | None = None,
) -> dict[str, float | int | bool | None]:
    if expected_phrase_count <= 0:
        raise ValueError("expected_phrase_count must be positive")

    if phrase_df.empty:
        total_clicks = total_selections = total_corrections = completed = 0
    else:
        total_clicks = int(pd.to_numeric(phrase_df["Num Clicks"], errors="coerce").fillna(0).sum())
        total_selections = int(
            pd.to_numeric(phrase_df["Num Selections"], errors="coerce").fillna(0).sum()
        )
        total_corrections = int(
            pd.to_numeric(phrase_df["Num Corrections"], errors="coerce").fillna(0).sum()
        )
        completed = int(phrase_df["Phrase Completed"].fillna(False).astype(bool).sum())

    summary: dict[str, float | int | bool | None] = {
        "trial": trial,
        "expected_phrase_count": int(expected_phrase_count),
        "recorded_phrase_count": int(len(phrase_df)),
        "total_clicks": total_clicks,
        "total_selections": total_selections,
        "total_corrections": total_corrections,
        "avg_clicks_per_phrase": total_clicks / expected_phrase_count,
        "avg_selections_per_phrase": total_selections / expected_phrase_count,
        "avg_corrections_per_phrase": total_corrections / expected_phrase_count,
        "correction_rate_percent": _safe_rate(total_corrections, total_selections) * 100,
        "phrase_completion_rate": completed / expected_phrase_count,
        "all_phrases_recorded": int(len(phrase_df)) == expected_phrase_count,
    }
    return summary


def build_synthetic_trial_summaries(
    synthetic_phrase_df: pd.DataFrame,
    expected_phrase_count: int,
) -> pd.DataFrame:
    rows = []
    if synthetic_phrase_df.empty:
        return pd.DataFrame(rows)

    for trial, trial_df in synthetic_phrase_df.groupby("Trial Num", sort=True):
        rows.append(
            summarize_phrase_click_correction(
                trial_df,
                expected_phrase_count=expected_phrase_count,
                trial=int(trial),
            )
        )
    return pd.DataFrame(rows)


def mark_valid_trials(
    trial_summary_df: pd.DataFrame,
    min_phrase_completion_rate: float,
) -> pd.DataFrame:
    result = trial_summary_df.copy()
    if result.empty:
        result["trial_valid"] = pd.Series(dtype=bool)
        return result
    result["trial_valid"] = (
        pd.to_numeric(result["phrase_completion_rate"], errors="coerce").fillna(0)
        >= min_phrase_completion_rate
    )
    return result


def build_metric_distribution(trial_summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in CLICK_CORRECTION_METRICS:
        if metric not in trial_summary_df:
            values = pd.Series(dtype=float)
        else:
            values = pd.to_numeric(trial_summary_df[metric], errors="coerce").dropna()
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


def build_metric_comparison(
    real_summary: dict[str, Any],
    distribution_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for distribution_row in distribution_df.to_dict("records"):
        metric = distribution_row["metric"]
        real_value = real_summary.get(metric)
        synthetic_mean = distribution_row["synthetic_mean"]
        if real_value is None or synthetic_mean is None or pd.isna(real_value) or pd.isna(synthetic_mean):
            absolute_error = None
            relative_error = None
            real_within_95pct = None
        else:
            absolute_error = synthetic_mean - real_value
            relative_error = None if real_value == 0 else absolute_error / real_value
            p025 = distribution_row["synthetic_p025"]
            p975 = distribution_row["synthetic_p975"]
            real_within_95pct = bool(p025 <= real_value <= p975)

        rows.append(
            {
                "metric": metric,
                "real_value": real_value,
                **distribution_row,
                "absolute_error": absolute_error,
                "relative_error": relative_error,
                "real_within_synthetic_95pct": real_within_95pct,
            }
        )
    return pd.DataFrame(rows)


def run_old_nomon_simulator(
    click_df: pd.DataFrame,
    phrase_df: pd.DataFrame,
    lm_backend: str,
    lm_size: str,
    lm_cache_dir: Path,
    selection_bootstrap_seed: int,
    append_terminal_periods: bool,
    verbose: bool,
) -> SimulatedUser:
    sim = SimulatedUser()
    params = {
        "click_df": click_df,
        "phrase_df": phrase_df,
        "selection_bootstrap_seed": selection_bootstrap_seed,
        "record_attempted_phrases": True,
        "append_terminal_periods": append_terminal_periods,
    }
    params.update(lm_parameters(lm_backend, lm_size, lm_cache_dir))

    original_cwd = Path.cwd()
    try:
        os.chdir(REPO_ROOT / "User_Simulation")
        sim.parameter_metrics(params, trials=1, verbose=verbose)
    finally:
        os.chdir(original_cwd)

    return sim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-id", default="C")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--clicks-per-phrase", type=int, default=500)
    parser.add_argument("--calibration-clicks", type=int, default=200)
    parser.add_argument(
        "--min-phrase-completion-rate",
        type=float,
        default=MIN_VALID_PHRASE_COMPLETION_RATE,
        help="Only trials at or above this completion rate contribute to synthetic metric distributions.",
    )
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--lm-backend", choices=["kenlm", "imagineville"], default="imagineville")
    parser.add_argument("--lm-size", choices=["tiny", "medium"], default="tiny")
    parser.add_argument("--lm-cache-dir", type=Path, default=DEFAULT_LM_CACHE_DIR)
    parser.add_argument(
        "--append-terminal-periods",
        action="store_true",
        help="Use the legacy simulator target phrase suffix. Disabled by default for phrase-level click comparison.",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
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
    validation_click_df, validation_phrase_df = select_sessions(
        real_click_df,
        real_phrase_df,
        validation_sessions,
    )
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
    effective_bootstrap_source_sessions = (
        profile["bootstrap_source_sessions"] or train_sessions
    )

    real_phrase_summary_df = build_real_phrase_click_correction(
        validation_click_df,
        validation_phrase_df,
    )
    expected_phrase_count = len(validation_phrase_df)
    real_summary = summarize_phrase_click_correction(
        real_phrase_summary_df,
        expected_phrase_count=expected_phrase_count,
        trial=None,
    )

    synthetic_phrase_results = []
    for trial in range(args.trials):
        rng = np.random.default_rng(args.seed + trial)
        synthetic_click_df = generate_synthetic_click_df(
            profile,
            validation_phrase_df,
            trial=trial,
            rng=rng,
            clicks_per_phrase=args.clicks_per_phrase,
            calibration_clicks=args.calibration_clicks,
            train_click_df=train_click_df,
        )
        synthetic_click_df = zero_active_dead_time(synthetic_click_df)

        sim = run_old_nomon_simulator(
            click_df=synthetic_click_df,
            phrase_df=validation_phrase_df,
            lm_backend=args.lm_backend,
            lm_size=args.lm_size,
            lm_cache_dir=args.lm_cache_dir,
            selection_bootstrap_seed=args.seed + trial,
            append_terminal_periods=args.append_terminal_periods,
            verbose=args.verbose,
        )
        trial_phrase_df = build_synthetic_phrase_click_correction(sim.result_df)
        if not trial_phrase_df.empty:
            # Each simulator invocation runs one internal trial, so the raw
            # result frame labels every run as trial 0. Preserve the evaluator's
            # outer trial id before aggregating across runs.
            trial_phrase_df["Trial Num"] = trial
            trial_phrase_df["Synthetic Profile User ID"] = args.user_id
            synthetic_phrase_results.append(trial_phrase_df)

    synthetic_phrase_df = (
        pd.concat(synthetic_phrase_results, ignore_index=True)
        if synthetic_phrase_results
        else pd.DataFrame()
    )
    trial_summary_df = build_synthetic_trial_summaries(
        synthetic_phrase_df,
        expected_phrase_count=expected_phrase_count,
    )
    trial_summary_df = mark_valid_trials(
        trial_summary_df,
        min_phrase_completion_rate=args.min_phrase_completion_rate,
    )
    valid_trial_summary_df = trial_summary_df[trial_summary_df["trial_valid"]].copy()
    distribution_df = build_metric_distribution(valid_trial_summary_df)
    metric_comparison_df = build_metric_comparison(real_summary, distribution_df)
    validation_status = {
        "minimum_phrase_completion_rate": args.min_phrase_completion_rate,
        "total_trials": int(args.trials),
        "valid_trials": int(trial_summary_df["trial_valid"].sum()) if not trial_summary_df.empty else 0,
        "invalid_trials": (
            int((~trial_summary_df["trial_valid"]).sum()) if not trial_summary_df.empty else int(args.trials)
        ),
        "status": (
            "valid"
            if not trial_summary_df.empty and bool(trial_summary_df["trial_valid"].any())
            else "failed_no_valid_trials"
        ),
    }

    output_dir = build_output_dir(args.output_dir, args.user_id)
    write_json(output_dir / "profile.json", profile)
    real_phrase_summary_df.to_csv(output_dir / "real_phrase_click_correction.csv", index=False)
    synthetic_phrase_df.to_csv(output_dir / "synthetic_phrase_click_correction.csv", index=False)
    trial_summary_df.to_csv(output_dir / "trial_summary.csv", index=False)
    distribution_df.to_csv(output_dir / "synthetic_metric_distribution.csv", index=False)
    metric_comparison_df.to_csv(output_dir / "metric_comparison.csv", index=False)
    write_json(output_dir / "validation_status.json", validation_status)
    write_json(
        output_dir / "run_config.json",
        {
            "user_id": args.user_id,
            "trials": args.trials,
            "validation_fraction": args.validation_fraction,
            "clicks_per_phrase": args.clicks_per_phrase,
            "calibration_clicks": args.calibration_clicks,
            "min_phrase_completion_rate": args.min_phrase_completion_rate,
            "seed": args.seed,
            "lm_backend": args.lm_backend,
            "lm_size": args.lm_size,
            "lm_cache_dir": str(args.lm_cache_dir),
            "dead_time_mode": "synthetic_active_dead_time_zero",
            "ignored_metric_classes": ["wpm", "entry_rate", "elapsed_time", "dead_time"],
            "append_terminal_periods": args.append_terminal_periods,
            "train_sessions": train_sessions,
            "bootstrap_source_sessions": profile["bootstrap_source_sessions"],
            "effective_bootstrap_source_sessions": effective_bootstrap_source_sessions,
            "validation_sessions": validation_sessions,
            "expected_phrase_count": expected_phrase_count,
            "validation_status": validation_status,
        },
    )

    print(f"Saved bootstrap click/correction validation to: {output_dir}")
    print("Train sessions:", train_sessions)
    print("Bootstrap source sessions:", effective_bootstrap_source_sessions)
    print("Validation sessions:", validation_sessions)
    print("Validation status:")
    print(json.dumps(validation_status, indent=2, sort_keys=True))
    print(metric_comparison_df.to_string(index=False))


if __name__ == "__main__":
    main()
