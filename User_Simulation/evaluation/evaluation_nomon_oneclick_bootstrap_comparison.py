"""Compare OG Nomon and OneClick with shared bootstrapped click schedules.

Run from the repository root:

    python -m User_Simulation.evaluation.evaluation_nomon_oneclick_bootstrap_comparison

This is the bootstrap counterpart to the Gaussian comparison runner. It uses
the regime-aware selection bootstrap profile learned from real OG Nomon
training sessions, zeros active dead time for both systems, and reports click
and correction burden on the same held-out reachable phrase set.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from User_Simulation.evaluation.evaluation_baseline import (
    DEFAULT_LM_CACHE_DIR,
    REPO_ROOT,
    load_text_click_data,
    load_text_phrase_data,
    lm_parameters,
    normalize_phrase_order,
    write_json,
)
from User_Simulation.evaluation.evaluation_bootstrap_click_correction_validation import (
    zero_active_dead_time,
)
from User_Simulation.evaluation.evaluation_click_offset_std_sweep import (
    build_clock_regimes,
)
from User_Simulation.evaluation.evaluation_nomon_oneclick_comparison import (
    DEFAULT_USERS,
    build_comparison_phrase_df,
    build_comparison_summary,
    build_phrase_trial_comparison,
    build_trial_summary,
    normalize_system_results,
    parse_csv_values,
    plot_summary,
    run_oneclick,
)
from User_Simulation.evaluation.evaluation_oneclick_failure_plots import (
    create_oneclick_failure_comparison,
)
from User_Simulation.evaluation.synthetic_profiles import (
    CLICK_OFFSET_COL,
    CLOCK_PERIOD_COL,
    estimate_profile,
    generate_synthetic_click_df,
    select_sessions,
    split_sessions,
    summarize_selection_groups,
)
from User_Simulation.simulated_user_text import SimulatedUser as OldNomonSimulatedUser


def build_output_dir(base_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    output_dir = base_dir / f"nomon_oneclick_bootstrap_comparison_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def clean_click_rows(click_df: pd.DataFrame) -> pd.DataFrame:
    result = click_df.copy()
    result[CLOCK_PERIOD_COL] = pd.to_numeric(result[CLOCK_PERIOD_COL], errors="coerce")
    result[CLICK_OFFSET_COL] = pd.to_numeric(result[CLICK_OFFSET_COL], errors="coerce")
    invalid = result[[CLOCK_PERIOD_COL, CLICK_OFFSET_COL]].isna().any(axis=1)
    if invalid.any():
        raise ValueError(f"Found {int(invalid.sum())} click rows without numeric timing values")
    return result


def effective_bootstrap_source_sessions(profile: dict[str, Any], train_sessions: list[int]) -> list[int]:
    source_sessions = profile.get("bootstrap_source_sessions") or train_sessions
    return [int(session) for session in source_sessions]


def bootstrap_profile_row(
    user_id: str,
    profile: dict[str, Any],
    train_sessions: list[int],
    train_click_df: pd.DataFrame,
) -> dict[str, Any]:
    source_sessions = effective_bootstrap_source_sessions(profile, train_sessions)
    effective_click_df = train_click_df[train_click_df["Session Num"].isin(source_sessions)].copy()
    effective_selection_summary = summarize_selection_groups(effective_click_df)
    clock_period = float(profile["stable_clock_period_s"])
    if clock_period < 2.0:
        clock_pace = "fast"
    elif clock_period <= 4.0:
        clock_pace = "moderate"
    else:
        clock_pace = "slow"

    normalized_sd = (
        float(profile["click_offset_sd_s"]) / clock_period if clock_period else float("nan")
    )
    if normalized_sd < 0.05:
        timing_variability = "low"
    elif normalized_sd <= 0.10:
        timing_variability = "moderate"
    else:
        timing_variability = "high"

    normalized_mean = (
        float(profile["click_offset_mean_s"]) / clock_period if clock_period else float("nan")
    )
    if abs(normalized_mean) <= 0.02:
        click_offset_bias = "centered"
    elif normalized_mean > 0:
        click_offset_bias = "late/positive"
    else:
        click_offset_bias = "early/negative"

    source_policy = (
        "latest_stable_sessions"
        if profile.get("bootstrap_source_sessions")
        else "all_training_sessions_fallback"
    )
    description = (
        f"{clock_pace.capitalize()}-clock bootstrap profile with {timing_variability} "
        f"timing variability, {click_offset_bias} click-offset bias, "
        f"{int(effective_selection_summary['train_selection_groups'])} source selection groups, "
        f"and zero active dead time for comparison."
    )
    if source_policy == "all_training_sessions_fallback":
        description += " Uses all training sessions because no contiguous stable source pool was detected."

    return {
        "user_id": user_id,
        "clock_pace": clock_pace,
        "timing_variability": timing_variability,
        "click_offset_bias": click_offset_bias,
        "stable_clock_period_s": clock_period,
        "source_policy": source_policy,
        "train_sessions": ",".join(str(session) for session in train_sessions),
        "bootstrap_source_sessions": ",".join(str(session) for session in source_sessions),
        "bootstrap_source_click_rows": int(len(effective_click_df)),
        "train_selection_groups": int(effective_selection_summary["train_selection_groups"]),
        "train_character_selection_groups": int(effective_selection_summary["train_character_selection_groups"]),
        "train_word_prediction_selection_groups": int(
            effective_selection_summary["train_word_prediction_selection_groups"]
        ),
        "train_correction_selection_groups": int(
            effective_selection_summary["train_correction_selection_groups"]
        ),
        "train_mean_clicks_per_selection_group": float(
            effective_selection_summary["train_mean_clicks_per_selection_group"]
        ),
        "normalized_click_offset_mean": normalized_mean,
        "normalized_click_offset_sd": normalized_sd,
        "qualitative_description": description,
    }


def build_shared_bootstrap_click_df(
    profile: dict[str, Any],
    phrase_df: pd.DataFrame,
    train_click_df: pd.DataFrame,
    trial: int,
    seed: int,
    clicks_per_phrase: int,
    calibration_clicks: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    click_df = generate_synthetic_click_df(
        profile,
        phrase_df,
        trial=trial,
        rng=rng,
        clicks_per_phrase=clicks_per_phrase,
        calibration_clicks=calibration_clicks,
        train_click_df=train_click_df,
    )
    return zero_active_dead_time(click_df)


def run_old_nomon_bootstrap(
    click_df: pd.DataFrame,
    phrase_df: pd.DataFrame,
    lm_backend: str,
    lm_size: str,
    lm_cache_dir: Path,
    selection_bootstrap_seed: int,
    verbose: bool,
) -> pd.DataFrame:
    sim = OldNomonSimulatedUser()
    params = {
        "click_df": click_df,
        "phrase_df": phrase_df,
        "record_attempted_phrases": True,
        "append_terminal_periods": False,
        "selection_bootstrap_seed": selection_bootstrap_seed,
    }
    params.update(lm_parameters(lm_backend, lm_size, lm_cache_dir, None))

    original_cwd = Path.cwd()
    try:
        os.chdir(REPO_ROOT / "User_Simulation")
        sim.parameter_metrics(params, trials=1, verbose=verbose)
    finally:
        os.chdir(original_cwd)
    return sim.result_df.copy()


def write_bootstrap_markdown_report(
    output_dir: Path,
    summary_df: pd.DataFrame,
    profile_df: pd.DataFrame,
) -> None:
    def markdown_table(df: pd.DataFrame) -> str:
        columns = list(df.columns)
        rows = [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
        ]
        for record in df.to_dict("records"):
            values = []
            for column in columns:
                value = record[column]
                if pd.isna(value):
                    values.append("—")
                elif isinstance(value, float):
                    values.append(f"{value:.2f}")
                else:
                    values.append(str(value))
            rows.append("| " + " | ".join(values) + " |")
        return "\n".join(rows)

    lines = ["# OG Nomon vs OneClick Bootstrap Comparison", ""]
    lines.append("## Headline Summary")
    lines.append("")
    has_undo_only = summary_df["oneclick_undo_only_phrase_completion_rate"].notna().any()
    headline_columns = [
        "user_id",
        "mutually_completed_phrase_trials",
        "mutually_completed_og_mean_clicks",
        "mutually_completed_oneclick_mean_clicks",
        "mutually_completed_click_reduction_percent",
        "oneclick_phrase_completion_rate",
    ]
    headline_renames = {
        "mutually_completed_phrase_trials": "oneclick_paired_phrases",
        "mutually_completed_og_mean_clicks": "og_clicks",
        "mutually_completed_oneclick_mean_clicks": "oneclick_clicks",
        "mutually_completed_click_reduction_percent": "oneclick_click_reduction_percent",
        "oneclick_phrase_completion_rate": "oneclick_phrase_completion_rate",
    }
    if has_undo_only:
        headline_columns.extend(
            [
                "undo_only_mutually_completed_phrase_trials",
                "undo_only_mutually_completed_og_mean_clicks",
                "undo_only_mutually_completed_oneclick_mean_clicks",
                "undo_only_mutually_completed_click_reduction_percent",
                "oneclick_undo_only_phrase_completion_rate",
            ]
        )
        headline_renames.update(
            {
                "undo_only_mutually_completed_phrase_trials": "undo_only_paired_phrases",
                "undo_only_mutually_completed_og_mean_clicks": "undo_only_og_clicks",
                "undo_only_mutually_completed_oneclick_mean_clicks": "undo_only_oneclick_clicks",
                "undo_only_mutually_completed_click_reduction_percent": "undo_only_click_reduction_percent",
                "oneclick_undo_only_phrase_completion_rate": "undo_only_phrase_completion_rate",
            }
        )
    headline_df = summary_df[headline_columns].rename(columns=headline_renames)
    lines.append(
        markdown_table(headline_df)
    )
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    protected_reduction = summary_df["mutually_completed_click_reduction_percent"].mean()
    protected_label = (
        "n/a" if pd.isna(protected_reduction) else f"{protected_reduction:.1f}%"
    )
    lines.append(f"Average mutually-completed OneClick click reduction: {protected_label}.")
    lines.append(
        "Average OneClick correction-rate change: "
        f"{summary_df['mean_correction_rate_difference'].mean():+.1f} points."
    )
    if has_undo_only:
        undo_only_reduction = summary_df[
            "undo_only_mutually_completed_click_reduction_percent"
        ].mean()
        undo_only_label = (
            "n/a" if pd.isna(undo_only_reduction) else f"{undo_only_reduction:.1f}%"
        )
        lines.append(
            f"Average mutually-completed Undo-only click reduction: {undo_only_label}."
        )
    lines.append("")
    lines.append(
        "Headline click reductions use only phrase trials completed by both OG Nomon "
        "and OneClick."
    )
    lines.append(
        "OneClick uses protected correction: prediction clocks remain active, but "
        "non-Undo winners cannot create additional commits while correction is latched."
    )
    lines.append("")
    lines.append("## Bootstrap Profiles")
    lines.append("")
    for row in profile_df.sort_values("user_id").to_dict("records"):
        lines.append(f"- User {row['user_id']}: {row['qualitative_description']}")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- Active dead time is zeroed for both systems, so WPM/elapsed-time outcomes are not interpreted."
    )
    lines.append(
        "- OG Nomon uses the existing regime-aware selection-bootstrap path; OneClick consumes the same generated bootstrapped click rows in order."
    )
    (output_dir / "comparison_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--users", default=",".join(DEFAULT_USERS))
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--max-phrases", type=int, default=None)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--clicks-per-phrase", type=int, default=500)
    parser.add_argument("--calibration-clicks", type=int, default=200)
    parser.add_argument("--oneclick-max-word-attempts", type=int, default=5)
    parser.add_argument("--oneclick-max-enter-attempts", type=int, default=5)
    parser.add_argument("--oneclick-max-clicks-per-word", type=int, default=30)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--phrase-policy", choices=["reachable", "all"], default="reachable")
    parser.add_argument("--lm-backend", choices=["kenlm", "imagineville"], default="imagineville")
    parser.add_argument("--lm-size", choices=["tiny", "medium"], default="tiny")
    parser.add_argument("--lm-cache-dir", type=Path, default=DEFAULT_LM_CACHE_DIR)
    parser.add_argument(
        "--oneclick-cache-dir",
        type=Path,
        default=REPO_ROOT / ".cache" / "oneclick_phrase_audit",
    )
    parser.add_argument(
        "--oneclick-perfect-letter-observations",
        action="store_true",
        help="Diagnostic mode: replace OneClick letter observations with near-certain intended letters.",
    )
    parser.add_argument(
        "--include-undo-only",
        action="store_true",
        help="Also run the diagnostic Undo-only OneClick variant.",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
    )
    return parser.parse_args()


def run_comparison(args: argparse.Namespace) -> Path:
    user_ids = parse_csv_values(args.users)
    if args.trials < 1:
        raise ValueError("--trials must be at least 1")
    if args.clicks_per_phrase < 1:
        raise ValueError("--clicks-per-phrase must be at least 1")

    output_dir = build_output_dir(args.output_dir)
    all_phrase_rows = []
    all_audit_rows = []
    all_profiles = []
    all_system_results = []
    user_configs: dict[str, Any] = {}

    for user_id in user_ids:
        real_click_df = clean_click_rows(load_text_click_data(user_id))
        real_phrase_df = normalize_phrase_order(load_text_phrase_data(user_id))
        train_sessions, validation_sessions = split_sessions(
            real_click_df,
            args.validation_fraction,
        )
        train_click_df, train_phrase_df = select_sessions(
            real_click_df,
            real_phrase_df,
            train_sessions,
        )
        validation_click_df, validation_phrase_df = select_sessions(
            real_click_df,
            real_phrase_df,
            validation_sessions,
        )
        validation_phrase_df = normalize_phrase_order(validation_phrase_df)

        profile = estimate_profile(
            user_id,
            train_click_df,
            train_phrase_df,
            validation_click_df,
            validation_phrase_df,
            train_sessions,
            validation_sessions,
        )
        profile_row = bootstrap_profile_row(user_id, profile, train_sessions, train_click_df)
        all_profiles.append(profile_row)

        regimes_df = build_clock_regimes(train_click_df)
        phrase_df, audit_df = build_comparison_phrase_df(
            user_id=user_id,
            validation_phrase_df=validation_phrase_df,
            validation_click_df=validation_click_df,
            regimes_df=regimes_df,
            phrase_policy=args.phrase_policy,
            oneclick_cache_dir=args.oneclick_cache_dir,
            max_phrases=args.max_phrases,
        )
        phrase_df.insert(0, "user_id", user_id)
        all_phrase_rows.append(phrase_df)
        if not audit_df.empty:
            audit_df.insert(0, "user_id", user_id)
            all_audit_rows.append(audit_df)

        user_configs[user_id] = {
            "train_sessions": train_sessions,
            "validation_sessions": validation_sessions,
            "bootstrap_source_sessions": effective_bootstrap_source_sessions(
                profile,
                train_sessions,
            ),
            "available_phrase_count": int(len(phrase_df)),
        }

        for trial in range(args.trials):
            print(f"User {user_id} trial {trial + 1}/{args.trials}: running OG Nomon")
            trial_seed = args.seed + trial + sum(user_id.encode("utf-8")) * 1000
            click_df = build_shared_bootstrap_click_df(
                profile=profile,
                phrase_df=phrase_df,
                train_click_df=train_click_df,
                trial=trial,
                seed=trial_seed,
                clicks_per_phrase=args.clicks_per_phrase,
                calibration_clicks=args.calibration_clicks,
            )
            old_results = run_old_nomon_bootstrap(
                click_df=click_df.copy(),
                phrase_df=phrase_df,
                lm_backend=args.lm_backend,
                lm_size=args.lm_size,
                lm_cache_dir=args.lm_cache_dir,
                selection_bootstrap_seed=trial_seed,
                verbose=args.verbose,
            )

            all_system_results.append(
                normalize_system_results(old_results, "og_nomon", user_id, trial)
            )
            oneclick_modes = [("protected", "oneclick")]
            if args.include_undo_only:
                oneclick_modes.append(("undo_only", "oneclick_undo_only"))
            for undo_mode, system_name in oneclick_modes:
                print(
                    f"User {user_id} trial {trial + 1}/{args.trials}: "
                    f"running OneClick ({undo_mode})"
                )
                oneclick_results = run_oneclick(
                    click_df=click_df.copy(),
                    phrase_df=phrase_df,
                    max_word_attempts=args.oneclick_max_word_attempts,
                    max_enter_attempts=args.oneclick_max_enter_attempts,
                    max_clicks_per_word=args.oneclick_max_clicks_per_word,
                    undo_mode=undo_mode,
                    oneclick_cache_dir=args.oneclick_cache_dir,
                    perfect_letter_observations=args.oneclick_perfect_letter_observations,
                    verbose=args.verbose,
                )
                all_system_results.append(
                    normalize_system_results(oneclick_results, system_name, user_id, trial)
                )

    phrase_set_df = pd.concat(all_phrase_rows, ignore_index=True)
    profile_df = pd.DataFrame(all_profiles)
    system_results_df = pd.concat(all_system_results, ignore_index=True)
    phrase_trial_df = build_phrase_trial_comparison(system_results_df)
    trial_summary_df = build_trial_summary(system_results_df)
    comparison_summary_df = build_comparison_summary(phrase_trial_df)
    comparison_summary_df = comparison_summary_df.merge(
        profile_df[
            [
                "user_id",
                "clock_pace",
                "timing_variability",
                "click_offset_bias",
                "source_policy",
                "qualitative_description",
            ]
        ],
        on="user_id",
        how="left",
    )

    phrase_set_df.to_csv(output_dir / "phrase_set.csv", index=False)
    profile_df.to_csv(output_dir / "user_bootstrap_profiles.csv", index=False)
    system_results_df.to_csv(output_dir / "system_phrase_results.csv", index=False)
    phrase_trial_df.to_csv(output_dir / "phrase_trial_results.csv", index=False)
    trial_summary_df.to_csv(output_dir / "trial_summary.csv", index=False)
    comparison_summary_df.to_csv(output_dir / "comparison_summary.csv", index=False)
    if all_audit_rows:
        pd.concat(all_audit_rows, ignore_index=True).to_csv(
            output_dir / "oneclick_phrase_reachability.csv",
            index=False,
        )
    plot_summary(comparison_summary_df, output_dir)
    write_bootstrap_markdown_report(output_dir, comparison_summary_df, profile_df)

    write_json(
        output_dir / "run_config.json",
        {
            "comparison_model": "regime_aware_selection_bootstrap",
            "users": user_ids,
            "user_configs": user_configs,
            "trials": args.trials,
            "max_phrases": args.max_phrases,
            "validation_fraction": args.validation_fraction,
            "clicks_per_phrase": args.clicks_per_phrase,
            "calibration_clicks": args.calibration_clicks,
            "oneclick_max_word_attempts": args.oneclick_max_word_attempts,
            "oneclick_max_enter_attempts": args.oneclick_max_enter_attempts,
            "oneclick_max_clicks_per_word": args.oneclick_max_clicks_per_word,
            "oneclick_stop_phrase_on_failed_word": True,
            "oneclick_failure_telemetry_version": 1,
            "oneclick_undo_modes": (
                ["protected", "undo_only"]
                if args.include_undo_only
                else ["protected"]
            ),
            "oneclick_perfect_letter_observations": bool(
                args.oneclick_perfect_letter_observations
            ),
            "seed": args.seed,
            "phrase_policy": args.phrase_policy,
            "lm_backend": args.lm_backend,
            "lm_size": args.lm_size,
            "lm_cache_dir": str(args.lm_cache_dir.resolve()),
            "oneclick_cache_dir": str(args.oneclick_cache_dir.resolve()),
            "dead_time_mode": "zero_active_dead_time_for_both_systems",
            "old_nomon_append_terminal_periods": False,
            "old_nomon_click_mode": "selection_group_bootstrap_runtime",
            "oneclick_click_mode": "playthrough_same_generated_bootstrapped_rows",
            "outputs": [
                "phrase_set.csv",
                "user_bootstrap_profiles.csv",
                "system_phrase_results.csv",
                "phrase_trial_results.csv",
                "trial_summary.csv",
                "comparison_summary.csv",
                "comparison_report.md",
                "plots/",
            ],
        },
    )

    presentation_outputs = create_oneclick_failure_comparison(output_dir)
    print("Created exact-failure OneClick presentation:")
    for presentation_output in presentation_outputs:
        print(presentation_output)

    print(f"Saved bootstrap OG Nomon vs OneClick comparison to: {output_dir}")
    printed_columns = [
        "user_id",
        "mutually_completed_phrase_trials",
        "mutually_completed_click_reduction_percent",
        "oneclick_phrase_completion_rate",
    ]
    if args.include_undo_only:
        printed_columns.extend(
            [
                "undo_only_mutually_completed_phrase_trials",
                "undo_only_mutually_completed_click_reduction_percent",
                "oneclick_undo_only_phrase_completion_rate",
            ]
        )
    print(comparison_summary_df[printed_columns].to_string(index=False))
    return output_dir


def main() -> None:
    run_comparison(parse_args())


if __name__ == "__main__":
    main()
