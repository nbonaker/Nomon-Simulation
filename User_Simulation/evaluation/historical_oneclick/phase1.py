"""Historical QuickClick data/analysis extracted from evaluation_oneclick_space_enter_phase1.

No simulator execution. Saved result semantics are preserved.
"""
from __future__ import annotations

import hashlib

import math

from pathlib import Path

from typing import Any

import numpy as np

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.ticker import MultipleLocator, PercentFormatter

from User_Simulation.evaluation.historical_oneclick.comparison import normalize_system_results

from User_Simulation.evaluation.historical_oneclick.speed import configure_plot_style, validate_phrase_timing

TIME_HORIZONS_S = (60.0, 120.0, 180.0)

AUC_HORIZON_S = 180.0

def condition_path(
    run_dir: Path,
    user_id: str,
    space_index: int,
    enter_index: int,
    trial: int,
) -> Path:
    return (
        run_dir
        / "conditions"
        / f"user_{user_id}"
        / f"space_{space_index:02d}"
        / f"enter_{enter_index:02d}"
        / f"trial_{trial:02d}.csv"
    )

def _frame_checksum(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(index=False, float_format="%.17g")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def normalize_condition_results(
    raw_results: pd.DataFrame,
    user_id: str,
    trial: int,
    combo: dict[str, Any],
    schedule_id: str,
) -> pd.DataFrame:
    result = normalize_system_results(raw_results, "oneclick", user_id, trial)
    for column in [
        "combo_id",
        "space_period_index",
        "space_period_s",
        "space_period_label",
        "enter_period_index",
        "enter_period_s",
        "enter_period_label",
        "combo_label",
        "is_diagonal",
    ]:
        result[column] = combo[column]
    result["paired_click_schedule_id"] = schedule_id
    return result

def _completion_times(group: pd.DataFrame) -> np.ndarray:
    completed = group["phrase_completed"].fillna(False).astype(bool)
    return (
        pd.to_numeric(
            group.loc[completed, "simulated_completion_time_s"],
            errors="coerce",
        )
        .dropna()
        .sort_values()
        .to_numpy(float)
    )

def build_combo_summary(phrase_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_columns = [
        "user_id",
        "combo_id",
        "space_period_index",
        "space_period_s",
        "space_period_label",
        "enter_period_index",
        "enter_period_s",
        "enter_period_label",
        "combo_label",
        "is_diagonal",
    ]
    for key, group in phrase_results.groupby(group_columns, sort=False):
        values = dict(zip(group_columns, key))
        completed = group["phrase_completed"].fillna(False).astype(bool)
        times = _completion_times(group)
        attempts = int(len(group))
        row = {
            **values,
            "phrase_attempts": attempts,
            "completed_phrases": int(completed.sum()),
            "phrase_completion_rate": float(completed.mean()),
            "median_completion_time_s": (
                float(np.median(times)) if len(times) else np.nan
            ),
            "maximum_completion_time_s": (
                float(np.max(times)) if len(times) else np.nan
            ),
            "total_attempt_time_s": float(
                pd.to_numeric(group["simulated_attempt_time_s"]).sum()
            ),
            "normalized_auc_180": float(
                np.maximum(AUC_HORIZON_S - times, 0.0).sum()
                / (attempts * AUC_HORIZON_S)
            ),
        }
        for horizon in TIME_HORIZONS_S:
            row[f"completion_by_{int(horizon)}s"] = float(
                (times <= horizon).sum() / attempts
            )
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary["pareto_nondominated"] = False
    summary["auc_180_rank"] = np.nan
    summary["final_completion_rank"] = np.nan
    for user_id, indices in summary.groupby("user_id", sort=False).groups.items():
        user_indices = list(indices)
        for index in user_indices:
            row = summary.loc[index]
            dominated = False
            for other_index in user_indices:
                if other_index == index:
                    continue
                other = summary.loc[other_index]
                no_worse = (
                    other["phrase_completion_rate"] >= row["phrase_completion_rate"]
                    and other["normalized_auc_180"] >= row["normalized_auc_180"]
                )
                strictly_better = (
                    other["phrase_completion_rate"] > row["phrase_completion_rate"]
                    or other["normalized_auc_180"] > row["normalized_auc_180"]
                )
                if no_worse and strictly_better:
                    dominated = True
                    break
            summary.loc[index, "pareto_nondominated"] = not dominated
        summary.loc[user_indices, "auc_180_rank"] = (
            summary.loc[user_indices, "normalized_auc_180"]
            .rank(method="min", ascending=False)
            .to_numpy()
        )
        summary.loc[user_indices, "final_completion_rank"] = (
            summary.loc[user_indices, "phrase_completion_rate"]
            .rank(method="min", ascending=False)
            .to_numpy()
        )
    return summary

def build_curve_points(phrase_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for user_id, user_frame in phrase_results.groupby("user_id", sort=False):
        user_times = _completion_times(user_frame)
        endpoint_s = (
            max(
                AUC_HORIZON_S,
                math.ceil(float(user_times.max()) / 30.0) * 30.0,
            )
            if len(user_times)
            else AUC_HORIZON_S
        )
        group_columns = [
            "combo_id",
            "space_period_index",
            "space_period_s",
            "enter_period_index",
            "enter_period_s",
            "combo_label",
            "is_diagonal",
        ]
        for key, group in user_frame.groupby(group_columns, sort=False):
            values = dict(zip(group_columns, key))
            attempts = int(len(group))
            times = _completion_times(group)
            rows.append(
                {
                    "user_id": user_id,
                    **values,
                    "simulated_time_s": 0.0,
                    "cumulative_completed": 0,
                    "cumulative_completion_rate": 0.0,
                    "phrase_attempts": attempts,
                    "event_type": "start",
                    "plot_endpoint_s": endpoint_s,
                }
            )
            for rank, completion_time in enumerate(times, start=1):
                rows.append(
                    {
                        "user_id": user_id,
                        **values,
                        "simulated_time_s": float(completion_time),
                        "cumulative_completed": rank,
                        "cumulative_completion_rate": rank / attempts,
                        "phrase_attempts": attempts,
                        "event_type": "completion",
                        "plot_endpoint_s": endpoint_s,
                    }
                )
            rows.append(
                {
                    "user_id": user_id,
                    **values,
                    "simulated_time_s": endpoint_s,
                    "cumulative_completed": int(len(times)),
                    "cumulative_completion_rate": len(times) / attempts,
                    "phrase_attempts": attempts,
                    "event_type": "endpoint",
                    "plot_endpoint_s": endpoint_s,
                }
            )
    return pd.DataFrame(rows)

def build_failure_distribution(phrase_results: pd.DataFrame) -> pd.DataFrame:
    failed = phrase_results[
        ~phrase_results["phrase_completed"].fillna(False).astype(bool)
    ].copy()
    group_columns = [
        "user_id",
        "combo_id",
        "space_period_index",
        "enter_period_index",
        "phrase_failure_reason",
        "phrase_failure_stage",
    ]
    if failed.empty:
        return pd.DataFrame(
            columns=group_columns + ["failure_count", "failure_share_of_attempts"]
        )
    counts = (
        failed.groupby(group_columns, dropna=False, sort=False)
        .size()
        .rename("failure_count")
        .reset_index()
    )
    attempts = (
        phrase_results.groupby(["user_id", "combo_id"])
        .size()
        .rename("phrase_attempts")
        .reset_index()
    )
    counts = counts.merge(attempts, on=["user_id", "combo_id"], validate="many_to_one")
    counts["failure_share_of_attempts"] = (
        counts["failure_count"] / counts["phrase_attempts"]
    )
    return counts

def _select_best(group: pd.DataFrame) -> pd.Series:
    return group.sort_values(
        [
            "normalized_auc_180",
            "phrase_completion_rate",
            "completion_by_120s",
            "combo_id",
        ],
        ascending=[False, False, False, True],
        kind="stable",
    ).iloc[0]

def build_best_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metric_columns = [
        "combo_id",
        "space_period_index",
        "space_period_s",
        "enter_period_index",
        "enter_period_s",
        "phrase_completion_rate",
        "completion_by_60s",
        "completion_by_120s",
        "completion_by_180s",
        "normalized_auc_180",
        "median_completion_time_s",
    ]
    for user_id, user_summary in summary.groupby("user_id", sort=False):
        diagonal = _select_best(user_summary[user_summary["is_diagonal"].astype(bool)])
        off_diagonal = _select_best(
            user_summary[~user_summary["is_diagonal"].astype(bool)]
        )
        row: dict[str, Any] = {
            "user_id": user_id,
            "selection_basis": "maximum normalized AUC through 180 seconds",
        }
        for prefix, candidate in [
            ("best_diagonal", diagonal),
            ("best_off_diagonal", off_diagonal),
        ]:
            for column in metric_columns:
                row[f"{prefix}_{column}"] = candidate[column]
        for metric in [
            "phrase_completion_rate",
            "completion_by_60s",
            "completion_by_120s",
            "completion_by_180s",
            "normalized_auc_180",
        ]:
            row[f"off_minus_diagonal_{metric}"] = (
                off_diagonal[metric] - diagonal[metric]
            )
        rows.append(row)
    return pd.DataFrame(rows)

def validate_final_outputs(
    phrase_results: pd.DataFrame,
    summary: pd.DataFrame,
    curves: pd.DataFrame,
    users: list[str],
    combos: list[dict[str, Any]],
    trials: int,
    phrase_count: int,
) -> None:
    expected_per_combo = trials * phrase_count
    expected_per_user = len(combos) * expected_per_combo
    user_counts = phrase_results.groupby("user_id").size()
    if any(int(user_counts.get(user_id, 0)) != expected_per_user for user_id in users):
        raise ValueError("not every user has the expected number of phrase attempts")
    combo_counts = phrase_results.groupby(["user_id", "combo_id"]).size()
    if (combo_counts != expected_per_combo).any():
        raise ValueError("not every combination has the expected phrase attempts")
    if phrase_results["simulated_attempt_time_s"].isna().any():
        raise ValueError("phrase results contain missing simulated attempt times")
    completed = phrase_results["phrase_completed"].fillna(False).astype(bool)
    typed = phrase_results["Typed Text"].fillna("").astype(str).str.rstrip()
    target = phrase_results["Target Phrase"].fillna("").astype(str).str.rstrip()
    if not typed[completed].eq(target[completed]).all():
        raise ValueError("completed phrase text does not exactly match its target")
    validate_phrase_timing(phrase_results)

    endpoints = curves[curves["event_type"] == "endpoint"]
    merged = summary.merge(
        endpoints[
            ["user_id", "combo_id", "cumulative_completion_rate"]
        ],
        on=["user_id", "combo_id"],
        how="left",
        validate="one_to_one",
    )
    if not np.allclose(
        merged["phrase_completion_rate"],
        merged["cumulative_completion_rate"],
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("curve plateaus do not equal phrase completion rates")
    metric_columns = [
        "phrase_completion_rate",
        "completion_by_60s",
        "completion_by_120s",
        "completion_by_180s",
        "normalized_auc_180",
    ]
    for column in metric_columns:
        values = pd.to_numeric(summary[column], errors="raise")
        if ((values < 0.0) | (values > 1.0) | ~np.isfinite(values)).any():
            raise ValueError(f"summary metric is outside [0, 1]: {column}")

def validate_diagonal_regression(
    run_dir: Path,
    baseline_run_dir: Path,
    users: list[str],
    phrase_ids: set[str],
) -> pd.DataFrame:
    rows = []
    columns = [
        "phrase_completed",
        "Typed Text",
        "phrase_failure_reason",
        "simulated_attempt_time_s",
        "simulated_completion_time_s",
        "letter_clock_time_s",
        "target_enter_clock_time_s",
        "undo_clock_time_s",
    ]
    for user_id in users:
        for period_index in [6, 10, 14]:
            current_path = condition_path(
                run_dir,
                user_id,
                period_index,
                period_index,
                0,
            )
            baseline_path = (
                baseline_run_dir
                / "conditions"
                / f"user_{user_id}"
                / f"period_{period_index:02d}"
                / "trial_00.csv"
            )
            if not current_path.is_file() or not baseline_path.is_file():
                continue
            current = pd.read_csv(current_path)
            baseline = pd.read_csv(baseline_path)
            current = current[
                current["Comparison Phrase ID"].astype(str).isin(phrase_ids)
            ].set_index("Comparison Phrase ID")
            baseline = baseline[
                baseline["Comparison Phrase ID"].astype(str).isin(phrase_ids)
            ].set_index("Comparison Phrase ID")
            current = current.sort_index()
            baseline = baseline.sort_index()
            passed = bool(current.index.equals(baseline.index))
            for column in columns:
                if not passed:
                    break
                if column not in current or column not in baseline:
                    passed = False
                    break
                if pd.api.types.is_numeric_dtype(current[column]):
                    left = pd.to_numeric(current[column], errors="coerce")
                    right = pd.to_numeric(baseline[column], errors="coerce")
                    passed = bool(
                        np.allclose(
                            left,
                            right,
                            rtol=0.0,
                            atol=1e-10,
                            equal_nan=True,
                        )
                    )
                else:
                    passed = bool(
                        current[column].fillna("").astype(str).equals(
                            baseline[column].fillna("").astype(str)
                        )
                    )
            rows.append(
                {
                    "user_id": user_id,
                    "period_index": period_index,
                    "phrase_count": int(len(current)),
                    "regression_passed": passed,
                    "baseline_condition": str(baseline_path.resolve()),
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty and not result["regression_passed"].all():
        raise ValueError("equal-period specialized runs do not reproduce the baseline")
    return result

def _heatmap_matrix(
    user_summary: pd.DataFrame,
    period_indices: list[int],
    metric: str,
) -> np.ndarray:
    positions = {period_index: position for position, period_index in enumerate(period_indices)}
    matrix = np.full((len(period_indices), len(period_indices)), np.nan)
    for row in user_summary.to_dict("records"):
        matrix[
            positions[int(row["space_period_index"])],
            positions[int(row["enter_period_index"])],
        ] = float(row[metric])
    return matrix

def create_heatmaps(
    run_dir: Path,
    summary: pd.DataFrame,
    users: list[str],
    period_indices: list[int],
) -> list[Path]:
    configure_plot_style()
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    metric_specs = [
        ("completion_by_60s", "Completed by 60 s"),
        ("completion_by_120s", "Completed by 120 s"),
        ("completion_by_180s", "Completed by 180 s"),
        ("phrase_completion_rate", "Final completion"),
    ]
    period_labels = [
        f"{saved_period_seconds(summary, index):.1f}" for index in period_indices
    ]
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#E5E7EB")

    for user_id in users:
        user_summary = summary[summary["user_id"] == user_id]
        figure, axes = plt.subplots(2, 2, figsize=(11, 9.5))
        figure.subplots_adjust(
            left=0.09,
            right=0.9,
            top=0.91,
            bottom=0.08,
            hspace=0.28,
            wspace=0.25,
        )
        images = []
        for axis, (metric, title) in zip(axes.flat, metric_specs):
            matrix = _heatmap_matrix(user_summary, period_indices, metric)
            image = axis.imshow(
                np.ma.masked_invalid(matrix),
                cmap=cmap,
                vmin=0.0,
                vmax=1.0,
                aspect="equal",
            )
            images.append(image)
            axis.set_title(title)
            axis.set_xticks(range(len(period_indices)), period_labels)
            axis.set_yticks(range(len(period_indices)), period_labels)
            axis.set_xlabel("Enter / Undo period (s)")
            axis.set_ylabel("Space period (s)")
            axis.tick_params(labelsize=8)
            for row in range(matrix.shape[0]):
                for column in range(matrix.shape[1]):
                    value = matrix[row, column]
                    if np.isnan(value):
                        continue
                    text_color = "white" if value < 0.35 or value > 0.75 else "#111827"
                    axis.text(
                        column,
                        row,
                        f"{value * 100:.0f}%",
                        ha="center",
                        va="center",
                        fontsize=7,
                        fontweight="bold",
                        color=text_color,
                    )
        figure.suptitle(f"User {user_id} — Space/Enter Phase 1")
        color_axis = figure.add_axes([0.925, 0.18, 0.018, 0.62])
        colorbar = figure.colorbar(images[-1], cax=color_axis)
        colorbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        colorbar.set_label("Phrase trials completed")
        png_path = plot_dir / f"user_{user_id}_space_enter_heatmaps.png"
        pdf_path = plot_dir / f"user_{user_id}_space_enter_heatmaps.pdf"
        figure.savefig(png_path, dpi=200, bbox_inches="tight", facecolor="white")
        figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        outputs.extend([png_path, pdf_path])
    return outputs

def _plot_curve_subset(
    curves: pd.DataFrame,
    summary: pd.DataFrame,
    user_id: str,
    combo_ids: list[str],
    title: str,
    output_stem: Path,
) -> list[Path]:
    configure_plot_style()
    figure, axis = plt.subplots(figsize=(10.5, 6.2))
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 0.9, max(len(combo_ids), 1)))
    user_summary = summary[summary["user_id"] == user_id].set_index("combo_id")
    for color, combo_id in zip(colors, combo_ids):
        curve = curves[
            (curves["user_id"] == user_id) & (curves["combo_id"] == combo_id)
        ].sort_values(["simulated_time_s", "cumulative_completed"], kind="stable")
        if curve.empty:
            continue
        row = user_summary.loc[combo_id]
        label = (
            f"S {row['space_period_s']:.1f}s / E {row['enter_period_s']:.1f}s "
            f"— {row['phrase_completion_rate'] * 100:.0f}%"
        )
        axis.step(
            curve["simulated_time_s"],
            curve["cumulative_completion_rate"],
            where="post",
            linewidth=2.0,
            color=color,
            label=label,
        )
    endpoint_s = float(
        curves.loc[curves["user_id"] == user_id, "plot_endpoint_s"].max()
    )
    axis.set_xlim(0.0, endpoint_s)
    axis.set_ylim(0.0, 1.05)
    axis.xaxis.set_major_locator(MultipleLocator(30))
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    axis.grid(alpha=0.18)
    axis.set_title(title)
    axis.set_xlabel("Simulated clock-interaction time (s)")
    axis.set_ylabel("Phrase trials completed by time")
    axis.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=8,
    )
    figure.subplots_adjust(left=0.09, right=0.7, top=0.91, bottom=0.12)
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    figure.savefig(png_path, dpi=200, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return [png_path, pdf_path]

def create_curve_plots(
    run_dir: Path,
    curves: pd.DataFrame,
    summary: pd.DataFrame,
    users: list[str],
) -> list[Path]:
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for user_id in users:
        user_summary = summary[summary["user_id"] == user_id].copy()
        diagonal_ids = user_summary[
            user_summary["is_diagonal"].astype(bool)
        ].sort_values("space_period_s", ascending=False)["combo_id"].tolist()
        off_diagonal = user_summary[
            ~user_summary["is_diagonal"].astype(bool)
            & user_summary["pareto_nondominated"].astype(bool)
        ].sort_values(
            ["normalized_auc_180", "phrase_completion_rate"],
            ascending=False,
        )
        if off_diagonal.empty:
            off_diagonal = user_summary[
                ~user_summary["is_diagonal"].astype(bool)
            ].sort_values(
                ["normalized_auc_180", "phrase_completion_rate"],
                ascending=False,
            )
        pareto_ids = off_diagonal.head(6)["combo_id"].tolist()
        outputs.extend(
            _plot_curve_subset(
                curves,
                summary,
                user_id,
                diagonal_ids,
                f"User {user_id} — diagonal Space = Enter conditions",
                plot_dir / f"user_{user_id}_diagonal_completion_curves",
            )
        )
        outputs.extend(
            _plot_curve_subset(
                curves,
                summary,
                user_id,
                pareto_ids,
                f"User {user_id} — strongest off-diagonal candidates",
                plot_dir / f"user_{user_id}_off_diagonal_pareto_curves",
            )
        )
    return outputs

def create_plots(
    run_dir: Path,
    summary: pd.DataFrame,
    curves: pd.DataFrame,
    users: list[str],
    period_indices: list[int],
) -> list[Path]:
    return create_heatmaps(run_dir, summary, users, period_indices) + create_curve_plots(
        run_dir,
        curves,
        summary,
        users,
    )

def saved_period_seconds(summary, index):
    """Use recorded periods, never an engine/default speed table."""
    values = []
    for group in ("space", "enter"):
        values.extend(summary.loc[summary[f"{group}_period_index"] == index, f"{group}_period_s"].dropna().unique())
    if not values or not np.allclose(values, values[0], rtol=0, atol=1e-12):
        raise ValueError(f"Missing or inconsistent recorded period for index {index}")
    return float(values[0])
