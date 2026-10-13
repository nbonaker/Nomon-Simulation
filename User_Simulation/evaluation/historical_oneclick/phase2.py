"""Historical QuickClick data/analysis extracted from evaluation_oneclick_space_enter_phase2.

No simulator execution. Saved result semantics are preserved.
"""
from __future__ import annotations

import math

from pathlib import Path

from typing import Any

import numpy as np

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.ticker import PercentFormatter

from User_Simulation.evaluation.historical_oneclick.speed import configure_plot_style, validate_phrase_timing

from User_Simulation.evaluation.historical_oneclick.phase1 import AUC_HORIZON_S, TIME_HORIZONS_S, _plot_curve_subset, build_combo_summary

def wilson_interval(successes: int, attempts: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if attempts <= 0:
        return np.nan, np.nan
    proportion = successes / attempts
    denominator = 1.0 + z * z / attempts
    center = (proportion + z * z / (2.0 * attempts)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / attempts
            + z * z / (4.0 * attempts * attempts)
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)

def build_trial_summary(phrase_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_columns = [
        "user_id",
        "combo_id",
        "space_period_index",
        "space_period_s",
        "enter_period_index",
        "enter_period_s",
        "combo_label",
        "trial",
    ]
    for key, group in phrase_results.groupby(group_columns, sort=False):
        values = dict(zip(group_columns, key))
        completed = group["phrase_completed"].fillna(False).astype(bool)
        times = (
            pd.to_numeric(
                group.loc[completed, "simulated_completion_time_s"],
                errors="coerce",
            )
            .dropna()
            .to_numpy(float)
        )
        attempts = len(group)
        row = {
            **values,
            "phrase_attempts": attempts,
            "completed_phrases": int(completed.sum()),
            "phrase_completion_rate": float(completed.mean()),
            "normalized_auc_180": float(
                np.maximum(AUC_HORIZON_S - times, 0.0).sum()
                / (attempts * AUC_HORIZON_S)
            ),
            "median_completion_time_s": (
                float(np.median(times)) if len(times) else np.nan
            ),
        }
        for horizon in TIME_HORIZONS_S:
            row[f"completion_by_{int(horizon)}s"] = float(
                (times <= horizon).sum() / attempts
            )
        rows.append(row)
    return pd.DataFrame(rows)

def enrich_candidate_summary(
    phrase_results: pd.DataFrame,
    reliability_floor: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = build_combo_summary(phrase_results)
    trial_summary = build_trial_summary(phrase_results)
    trial_stats = (
        trial_summary.groupby(["user_id", "combo_id"], sort=False)
        .agg(
            trial_completion_mean=("phrase_completion_rate", "mean"),
            trial_completion_std=("phrase_completion_rate", "std"),
            trial_completion_min=("phrase_completion_rate", "min"),
            trial_completion_max=("phrase_completion_rate", "max"),
            trial_auc_180_mean=("normalized_auc_180", "mean"),
            trial_auc_180_std=("normalized_auc_180", "std"),
        )
        .reset_index()
    )
    summary = summary.merge(
        trial_stats,
        on=["user_id", "combo_id"],
        how="left",
        validate="one_to_one",
    )
    intervals = [
        wilson_interval(int(row.completed_phrases), int(row.phrase_attempts))
        for row in summary.itertuples()
    ]
    summary["completion_ci95_low"] = [value[0] for value in intervals]
    summary["completion_ci95_high"] = [value[1] for value in intervals]
    summary["reliability_floor"] = float(reliability_floor)
    summary["meets_reliability_floor"] = (
        summary["phrase_completion_rate"] >= reliability_floor
    )
    return summary, trial_summary

def build_selection_summary(
    summary: pd.DataFrame,
    reliability_floor: float,
) -> pd.DataFrame:
    rows = []
    for user_id, user_summary in summary.groupby("user_id", sort=False):
        eligible = user_summary[
            user_summary["phrase_completion_rate"] >= reliability_floor
        ]
        if eligible.empty:
            pool = user_summary
            sort_columns = [
                "phrase_completion_rate",
                "normalized_auc_180",
                "completion_by_120s",
                "combo_id",
            ]
            selection_reason = (
                "no candidate met reliability floor; maximum final completion"
            )
        else:
            pool = eligible
            sort_columns = [
                "normalized_auc_180",
                "phrase_completion_rate",
                "completion_by_120s",
                "combo_id",
            ]
            selection_reason = (
                "met reliability floor; maximum normalized AUC through 180 seconds"
            )
        selected = pool.sort_values(
            sort_columns,
            ascending=[False, False, False, True],
            kind="stable",
        ).iloc[0]
        rows.append(
            {
                "user_id": user_id,
                "reliability_floor": reliability_floor,
                "selection_reason": selection_reason,
                "selected_combo_id": selected["combo_id"],
                "selected_space_period_index": selected["space_period_index"],
                "selected_space_period_s": selected["space_period_s"],
                "selected_enter_period_index": selected["enter_period_index"],
                "selected_enter_period_s": selected["enter_period_s"],
                "selected_phrase_completion_rate": selected[
                    "phrase_completion_rate"
                ],
                "selected_completion_ci95_low": selected["completion_ci95_low"],
                "selected_completion_ci95_high": selected["completion_ci95_high"],
                "selected_completion_by_60s": selected["completion_by_60s"],
                "selected_completion_by_120s": selected["completion_by_120s"],
                "selected_completion_by_180s": selected["completion_by_180s"],
                "selected_normalized_auc_180": selected["normalized_auc_180"],
                "selected_trial_completion_min": selected["trial_completion_min"],
                "selected_trial_completion_max": selected["trial_completion_max"],
            }
        )
    return pd.DataFrame(rows)

def build_paired_comparisons(
    phrase_results: pd.DataFrame,
    summary: pd.DataFrame,
    selections: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    key_columns = ["trial", "Comparison Phrase ID"]
    for selection in selections.to_dict("records"):
        user_id = selection["user_id"]
        selected_id = selection["selected_combo_id"]
        user_results = phrase_results[phrase_results["user_id"] == user_id]
        selected = user_results[user_results["combo_id"] == selected_id][
            key_columns + ["phrase_completed"]
        ].rename(columns={"phrase_completed": "selected_completed"})
        selected["selected_completed"] = selected["selected_completed"].astype(bool)
        selected_summary = summary[
            (summary["user_id"] == user_id)
            & (summary["combo_id"] == selected_id)
        ].iloc[0]
        for combo_id, candidate in user_results.groupby("combo_id", sort=False):
            candidate_pairs = candidate[
                key_columns + ["phrase_completed"]
            ].rename(columns={"phrase_completed": "candidate_completed"})
            paired = candidate_pairs.merge(
                selected,
                on=key_columns,
                how="inner",
                validate="one_to_one",
            )
            candidate_summary = summary[
                (summary["user_id"] == user_id)
                & (summary["combo_id"] == combo_id)
            ].iloc[0]
            candidate_completed = paired["candidate_completed"].astype(bool)
            selected_completed = paired["selected_completed"].astype(bool)
            rows.append(
                {
                    "user_id": user_id,
                    "selected_combo_id": selected_id,
                    "candidate_combo_id": combo_id,
                    "paired_phrase_trials": len(paired),
                    "both_completed": int(
                        (candidate_completed & selected_completed).sum()
                    ),
                    "candidate_only_completed": int(
                        (candidate_completed & ~selected_completed).sum()
                    ),
                    "selected_only_completed": int(
                        (~candidate_completed & selected_completed).sum()
                    ),
                    "neither_completed": int(
                        (~candidate_completed & ~selected_completed).sum()
                    ),
                    "candidate_minus_selected_completion_rate": float(
                        candidate_summary["phrase_completion_rate"]
                        - selected_summary["phrase_completion_rate"]
                    ),
                    "candidate_minus_selected_auc_180": float(
                        candidate_summary["normalized_auc_180"]
                        - selected_summary["normalized_auc_180"]
                    ),
                }
            )
    return pd.DataFrame(rows)

def validate_final_outputs(
    phrase_results: pd.DataFrame,
    summary: pd.DataFrame,
    trial_summary: pd.DataFrame,
    curves: pd.DataFrame,
    manifest: pd.DataFrame,
    candidate_map: dict[str, list[dict[str, Any]]],
    trials: int,
    phrase_count: int,
) -> None:
    if not (manifest["status"] == "completed").all():
        raise ValueError("Phase 2 manifest contains incomplete conditions")
    expected_conditions = sum(len(value) for value in candidate_map.values()) * trials
    if len(manifest) != expected_conditions:
        raise ValueError("Phase 2 manifest has an unexpected condition count")
    expected_attempts = expected_conditions * phrase_count
    if len(phrase_results) != expected_attempts:
        raise ValueError("Phase 2 phrase results have an unexpected row count")
    counts = phrase_results.groupby(["user_id", "combo_id", "trial"]).size()
    if (counts != phrase_count).any():
        raise ValueError("not every candidate/trial contains the common phrase set")
    if len(trial_summary) != expected_conditions:
        raise ValueError("trial summary does not contain one row per condition")
    for user_id, combos in candidate_map.items():
        expected_ids = {combo["combo_id"] for combo in combos}
        observed_ids = set(
            phrase_results.loc[
                phrase_results["user_id"] == user_id,
                "combo_id",
            ]
        )
        if observed_ids != expected_ids:
            raise ValueError(f"candidate mismatch for user {user_id}")
    completed = phrase_results["phrase_completed"].fillna(False).astype(bool)
    typed = phrase_results["Typed Text"].fillna("").astype(str).str.rstrip()
    target = phrase_results["Target Phrase"].fillna("").astype(str).str.rstrip()
    if not typed[completed].eq(target[completed]).all():
        raise ValueError("completed phrase text does not exactly match target text")
    validate_phrase_timing(phrase_results)
    endpoints = curves[curves["event_type"] == "endpoint"][
        ["user_id", "combo_id", "cumulative_completion_rate"]
    ]
    merged = summary.merge(
        endpoints,
        on=["user_id", "combo_id"],
        validate="one_to_one",
    )
    if not np.allclose(
        merged["phrase_completion_rate"],
        merged["cumulative_completion_rate"],
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("curve plateaus do not equal candidate completion rates")
    schedule_counts = phrase_results.groupby(
        ["user_id", "trial"]
    ).paired_click_schedule_id.nunique()
    if (schedule_counts != 1).any():
        raise ValueError("candidate conditions do not share one schedule per trial")

def _short_label(row: pd.Series) -> str:
    return f"S {row['space_period_s']:.1f} / E {row['enter_period_s']:.1f} s"

def create_plots(
    run_dir: Path,
    phrase_results: pd.DataFrame,
    summary: pd.DataFrame,
    trial_summary: pd.DataFrame,
    curves: pd.DataFrame,
    selections: pd.DataFrame,
) -> list[Path]:
    configure_plot_style()
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for user_id, user_summary in summary.groupby("user_id", sort=False):
        ordered = user_summary.sort_values(
            ["meets_reliability_floor", "normalized_auc_180"],
            ascending=False,
        )
        combo_ids = ordered["combo_id"].tolist()
        outputs.extend(
            _plot_curve_subset(
                curves,
                summary,
                user_id,
                combo_ids,
                f"User {user_id} — Phase 2 candidate completion by time",
                plot_dir / f"user_{user_id}_phase2_completion_curves",
            )
        )

        labels = [_short_label(row) for _, row in ordered.iterrows()]
        positions = np.arange(len(ordered))
        figure, axes = plt.subplots(1, 2, figsize=(12, 5.2))
        rates = ordered["phrase_completion_rate"].to_numpy(float)
        lower = rates - ordered["completion_ci95_low"].to_numpy(float)
        upper = ordered["completion_ci95_high"].to_numpy(float) - rates
        axes[0].bar(positions, rates, color="#4C9F70", alpha=0.9)
        axes[0].errorbar(
            positions,
            rates,
            yerr=np.vstack([lower, upper]),
            fmt="none",
            ecolor="#1F2937",
            capsize=4,
        )
        axes[0].axhline(
            float(ordered["reliability_floor"].iloc[0]),
            color="#D97706",
            linestyle="--",
            linewidth=1.5,
            label="Reliability floor",
        )
        axes[0].set_title("Final phrase completion")
        axes[0].set_ylim(0.0, 1.05)
        axes[0].yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        axes[0].legend(frameon=False, fontsize=8)
        auc_values = ordered["normalized_auc_180"].to_numpy(float)
        axes[1].bar(positions, auc_values, color="#3979A8", alpha=0.9)
        axes[1].set_title("Completion-by-time AUC through 180 s")
        axes[1].set_ylim(0.0, 1.0)
        axes[1].yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        for axis in axes:
            axis.set_xticks(positions, labels, rotation=22, ha="right")
            axis.grid(axis="y", alpha=0.18)
        figure.suptitle(f"User {user_id} — Phase 2 candidate comparison")
        figure.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.25, wspace=0.25)
        for suffix in ["png", "pdf"]:
            path = plot_dir / f"user_{user_id}_phase2_candidate_comparison.{suffix}"
            figure.savefig(path, dpi=200 if suffix == "png" else None, bbox_inches="tight", facecolor="white")
            outputs.append(path)
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(10.5, 5.8))
        colors = plt.get_cmap("tab10")(np.linspace(0.0, 0.8, len(ordered)))
        for position, (color, combo_id) in enumerate(zip(colors, combo_ids)):
            values = trial_summary[
                (trial_summary["user_id"] == user_id)
                & (trial_summary["combo_id"] == combo_id)
            ].sort_values("trial")["phrase_completion_rate"].to_numpy(float)
            offsets = np.linspace(-0.10, 0.10, len(values))
            axis.scatter(
                np.full(len(values), position) + offsets,
                values,
                s=42,
                color=color,
                alpha=0.85,
                zorder=3,
            )
            axis.hlines(
                values.mean(),
                position - 0.22,
                position + 0.22,
                color="#111827",
                linewidth=2.0,
            )
        axis.axhline(
            float(ordered["reliability_floor"].iloc[0]),
            color="#D97706",
            linestyle="--",
            linewidth=1.5,
        )
        axis.set_xticks(positions, labels, rotation=22, ha="right")
        axis.set_ylim(0.0, 1.05)
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        axis.set_ylabel("Phrase completion per 20-phrase trial")
        axis.set_title(f"User {user_id} — completion stability across five trials")
        axis.grid(axis="y", alpha=0.18)
        figure.subplots_adjust(left=0.1, right=0.98, top=0.9, bottom=0.24)
        for suffix in ["png", "pdf"]:
            path = plot_dir / f"user_{user_id}_phase2_trial_stability.{suffix}"
            figure.savefig(path, dpi=200 if suffix == "png" else None, bbox_inches="tight", facecolor="white")
            outputs.append(path)
        plt.close(figure)

        failed = phrase_results[
            (phrase_results["user_id"] == user_id)
            & ~phrase_results["phrase_completed"].fillna(False).astype(bool)
        ].copy()
        failure_table = (
            failed.groupby(["combo_id", "phrase_failure_reason"], dropna=False)
            .size()
            .unstack(fill_value=0)
            .reindex(combo_ids, fill_value=0)
            / int(ordered["phrase_attempts"].iloc[0])
        )
        figure, axis = plt.subplots(figsize=(10.5, 5.8))
        if failure_table.shape[1]:
            bottom = np.zeros(len(failure_table))
            failure_colors = plt.get_cmap("Set2")(
                np.linspace(0.0, 0.9, len(failure_table.columns))
            )
            for color, reason in zip(failure_colors, failure_table.columns):
                values = failure_table[reason].to_numpy(float)
                axis.bar(
                    positions,
                    values,
                    bottom=bottom,
                    label=str(reason),
                    color=color,
                )
                bottom += values
            axis.legend(
                frameon=False,
                fontsize=8,
                loc="upper left",
                bbox_to_anchor=(1.01, 1.0),
            )
        else:
            axis.text(
                0.5,
                0.5,
                "No failed phrase trials",
                transform=axis.transAxes,
                ha="center",
                va="center",
                color="#4B5563",
            )
        axis.set_xticks(positions, labels, rotation=22, ha="right")
        axis.set_ylim(0.0, 1.0)
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        axis.set_ylabel("Share of all phrase trials")
        axis.set_title(f"User {user_id} — Phase 2 failure reasons")
        axis.grid(axis="y", alpha=0.18)
        figure.subplots_adjust(left=0.1, right=0.72, top=0.9, bottom=0.24)
        for suffix in ["png", "pdf"]:
            path = plot_dir / f"user_{user_id}_phase2_failure_reasons.{suffix}"
            figure.savefig(path, dpi=200 if suffix == "png" else None, bbox_inches="tight", facecolor="white")
            outputs.append(path)
        plt.close(figure)
    return outputs
