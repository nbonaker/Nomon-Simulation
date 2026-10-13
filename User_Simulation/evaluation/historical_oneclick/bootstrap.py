"""Historical QuickClick data/analysis extracted from evaluation_nomon_oneclick_bootstrap_comparison.

No simulator execution. Saved result semantics are preserved.
"""
from __future__ import annotations

from pathlib import Path

from typing import Any

import numpy as np

import pandas as pd

from User_Simulation.evaluation.historical_oneclick.data_utils import zero_active_dead_time

from User_Simulation.evaluation.synthetic_profiles import CLICK_OFFSET_COL, CLOCK_PERIOD_COL, generate_synthetic_click_df, summarize_selection_groups

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
