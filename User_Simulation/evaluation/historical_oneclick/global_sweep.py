"""Historical QuickClick data/analysis extracted from evaluation_oneclick_global_space_enter_sweep.

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

from matplotlib.patches import Patch, Rectangle

from matplotlib.ticker import MultipleLocator, PercentFormatter

from User_Simulation.evaluation.historical_oneclick.speed import configure_plot_style, validate_phrase_timing

from User_Simulation.evaluation.historical_oneclick.phase1 import AUC_HORIZON_S, TIME_HORIZONS_S, build_combo_summary, saved_period_seconds

from User_Simulation.evaluation.historical_oneclick.phase2 import build_trial_summary

def build_global_summary(
    phrase_results: pd.DataFrame,
    reliability_floor: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_user = build_combo_summary(phrase_results)
    global_rows = []
    for combo_id, group in per_user.groupby("combo_id", sort=False):
        first = group.iloc[0]
        values: dict[str, Any] = {
            "combo_id": combo_id,
            "space_period_index": int(first["space_period_index"]),
            "space_period_s": float(first["space_period_s"]),
            "space_period_label": first["space_period_label"],
            "enter_period_index": int(first["enter_period_index"]),
            "enter_period_s": float(first["enter_period_s"]),
            "enter_period_label": first["enter_period_label"],
            "combo_label": first["combo_label"],
            "user_count": int(group["user_id"].nunique()),
            "users_meeting_reliability_floor": int(
                (group["phrase_completion_rate"] >= reliability_floor).sum()
            ),
            "worst_user_completion_rate": float(
                group["phrase_completion_rate"].min()
            ),
            "macro_phrase_completion_rate": float(
                group["phrase_completion_rate"].mean()
            ),
            "macro_normalized_auc_180": float(
                group["normalized_auc_180"].mean()
            ),
            "reliability_floor": reliability_floor,
        }
        for horizon in TIME_HORIZONS_S:
            column = f"completion_by_{int(horizon)}s"
            values[f"macro_{column}"] = float(group[column].mean())
        global_rows.append(values)
    summary = pd.DataFrame(global_rows)

    failed = phrase_results[
        ~phrase_results["phrase_completed"].fillna(False).astype(bool)
    ]
    failure_counts = (
        failed.groupby(["combo_id", "phrase_failure_reason"], dropna=False)
        .size()
        .rename("failure_count")
        .reset_index()
    )
    dominant: dict[str, tuple[str, int]] = {}
    for combo_id, group in failure_counts.groupby("combo_id", sort=False):
        row = group.assign(
            reason_sort=group["phrase_failure_reason"].fillna("").astype(str)
        ).sort_values(
            ["failure_count", "reason_sort"],
            ascending=[False, True],
            kind="stable",
        ).iloc[0]
        dominant[str(combo_id)] = (
            str(row["phrase_failure_reason"]),
            int(row["failure_count"]),
        )
    summary["dominant_failure_reason"] = summary["combo_id"].map(
        lambda value: dominant.get(str(value), ("none", 0))[0]
    )
    summary["dominant_failure_count"] = summary["combo_id"].map(
        lambda value: dominant.get(str(value), ("none", 0))[1]
    )
    summary = rank_global_summary(summary)
    return per_user, summary

def rank_global_summary(summary: pd.DataFrame) -> pd.DataFrame:
    result = summary.sort_values(
        [
            "users_meeting_reliability_floor",
            "worst_user_completion_rate",
            "macro_phrase_completion_rate",
            "macro_normalized_auc_180",
            "macro_completion_by_120s",
            "space_period_index",
            "enter_period_index",
        ],
        ascending=[False, False, False, False, False, True, True],
        kind="stable",
    ).reset_index(drop=True)
    result["global_rank"] = np.arange(1, len(result) + 1)
    return result

def build_global_curve_points(phrase_results: pd.DataFrame) -> pd.DataFrame:
    completed = phrase_results[
        phrase_results["phrase_completed"].fillna(False).astype(bool)
    ]
    all_times = pd.to_numeric(
        completed["simulated_completion_time_s"],
        errors="coerce",
    ).dropna()
    endpoint = (
        max(AUC_HORIZON_S, math.ceil(float(all_times.max()) / 30.0) * 30.0)
        if len(all_times)
        else AUC_HORIZON_S
    )
    rows = []
    combo_columns = [
        "combo_id",
        "space_period_index",
        "space_period_s",
        "enter_period_index",
        "enter_period_s",
        "combo_label",
    ]
    for key, group in phrase_results.groupby(combo_columns, sort=False):
        values = dict(zip(combo_columns, key))
        attempts = len(group)
        times = (
            pd.to_numeric(
                group.loc[
                    group["phrase_completed"].fillna(False).astype(bool),
                    "simulated_completion_time_s",
                ],
                errors="coerce",
            )
            .dropna()
            .sort_values()
            .to_numpy(float)
        )
        rows.append(
            {
                **values,
                "simulated_time_s": 0.0,
                "cumulative_completed": 0,
                "cumulative_completion_rate": 0.0,
                "phrase_attempts": attempts,
                "event_type": "start",
                "plot_endpoint_s": endpoint,
            }
        )
        for count, value in enumerate(times, start=1):
            rows.append(
                {
                    **values,
                    "simulated_time_s": float(value),
                    "cumulative_completed": count,
                    "cumulative_completion_rate": count / attempts,
                    "phrase_attempts": attempts,
                    "event_type": "completion",
                    "plot_endpoint_s": endpoint,
                }
            )
        rows.append(
            {
                **values,
                "simulated_time_s": endpoint,
                "cumulative_completed": len(times),
                "cumulative_completion_rate": len(times) / attempts,
                "phrase_attempts": attempts,
                "event_type": "endpoint",
                "plot_endpoint_s": endpoint,
            }
        )
    return pd.DataFrame(rows)

def build_global_trial_summary(
    phrase_results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_user_trial = build_trial_summary(phrase_results)
    rows = []
    for (combo_id, trial), group in per_user_trial.groupby(
        ["combo_id", "trial"],
        sort=False,
    ):
        first = group.iloc[0]
        row = {
            "combo_id": combo_id,
            "trial": int(trial),
            "space_period_index": int(first["space_period_index"]),
            "space_period_s": float(first["space_period_s"]),
            "enter_period_index": int(first["enter_period_index"]),
            "enter_period_s": float(first["enter_period_s"]),
            "combo_label": first["combo_label"],
            "user_count": int(group["user_id"].nunique()),
            "worst_user_completion_rate": float(
                group["phrase_completion_rate"].min()
            ),
            "macro_phrase_completion_rate": float(
                group["phrase_completion_rate"].mean()
            ),
            "macro_normalized_auc_180": float(
                group["normalized_auc_180"].mean()
            ),
        }
        for horizon in TIME_HORIZONS_S:
            column = f"completion_by_{int(horizon)}s"
            row[f"macro_{column}"] = float(group[column].mean())
        rows.append(row)
    return per_user_trial, pd.DataFrame(rows)

def build_paired_comparisons(
    phrase_results: pd.DataFrame,
    global_summary: pd.DataFrame,
    selected_combo_id: str,
) -> pd.DataFrame:
    keys = ["user_id", "trial", "Comparison Phrase ID"]
    selected = phrase_results[
        phrase_results["combo_id"] == selected_combo_id
    ][keys + ["phrase_completed"]].rename(
        columns={"phrase_completed": "selected_completed"}
    )
    selected["selected_completed"] = selected["selected_completed"].astype(bool)
    selected_summary = global_summary[
        global_summary["combo_id"] == selected_combo_id
    ].iloc[0]
    rows = []
    for combo_id, candidate in phrase_results.groupby("combo_id", sort=False):
        candidate_pairs = candidate[keys + ["phrase_completed"]].rename(
            columns={"phrase_completed": "candidate_completed"}
        )
        paired = candidate_pairs.merge(
            selected,
            on=keys,
            how="inner",
            validate="one_to_one",
        )
        candidate_completed = paired["candidate_completed"].astype(bool)
        selected_completed = paired["selected_completed"].astype(bool)
        candidate_summary = global_summary[
            global_summary["combo_id"] == combo_id
        ].iloc[0]
        rows.append(
            {
                "selected_combo_id": selected_combo_id,
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
                "candidate_minus_selected_macro_completion": float(
                    candidate_summary["macro_phrase_completion_rate"]
                    - selected_summary["macro_phrase_completion_rate"]
                ),
                "candidate_minus_selected_worst_user_completion": float(
                    candidate_summary["worst_user_completion_rate"]
                    - selected_summary["worst_user_completion_rate"]
                ),
                "candidate_minus_selected_macro_auc_180": float(
                    candidate_summary["macro_normalized_auc_180"]
                    - selected_summary["macro_normalized_auc_180"]
                ),
            }
        )
    return pd.DataFrame(rows)

def _metric_matrix(
    summary: pd.DataFrame,
    period_indices: list[int],
    metric: str,
) -> np.ndarray:
    position = {
        period_index: offset
        for offset, period_index in enumerate(period_indices)
    }
    matrix = np.full((len(period_indices), len(period_indices)), np.nan)
    for row in summary.to_dict("records"):
        matrix[
            position[int(row["space_period_index"])],
            position[int(row["enter_period_index"])],
        ] = float(row[metric])
    return matrix

def create_screen_heatmaps(
    run_dir: Path,
    summary: pd.DataFrame,
    shortlist: pd.DataFrame,
    period_indices: list[int],
) -> list[Path]:
    configure_plot_style()
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    labels = [
        f"{saved_period_seconds(summary, index):.1f}"
        for index in period_indices
    ]
    shortlist_ids = set(shortlist["combo_id"].astype(str))
    specs = [
        ("macro_phrase_completion_rate", "Final completion", 1.0),
        ("macro_completion_by_60s", "Completed by 60 s", 1.0),
        ("macro_completion_by_120s", "Completed by 120 s", 1.0),
        ("macro_completion_by_180s", "Completed by 180 s", 1.0),
        ("users_meeting_reliability_floor", "Users at or above 80%", 6.0),
    ]
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#E5E7EB")
    figure = plt.figure(figsize=(15.5, 9.5))
    grid = figure.add_gridspec(2, 6)
    axes = [
        figure.add_subplot(grid[0, 0:2]),
        figure.add_subplot(grid[0, 2:4]),
        figure.add_subplot(grid[0, 4:6]),
        figure.add_subplot(grid[1, 1:3]),
        figure.add_subplot(grid[1, 3:5]),
    ]
    figure.subplots_adjust(
        left=0.06,
        right=0.94,
        top=0.89,
        bottom=0.13,
        hspace=0.38,
        wspace=0.65,
    )
    for axis, (metric, title, maximum) in zip(axes, specs):
        matrix = _metric_matrix(summary, period_indices, metric)
        image = axis.imshow(
            np.ma.masked_invalid(matrix),
            cmap=cmap,
            vmin=0.0,
            vmax=maximum,
            aspect="equal",
        )
        axis.set_title(title)
        axis.set_xticks(range(len(period_indices)), labels)
        axis.set_yticks(range(len(period_indices)), labels)
        axis.set_xlabel("Enter / Undo period (s)")
        axis.set_ylabel("Space period (s)")
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = matrix[row_index, column_index]
                if np.isnan(value):
                    continue
                if maximum == 1.0:
                    label = f"{value * 100:.0f}%"
                    color = "white" if value < 0.35 or value > 0.75 else "#111827"
                else:
                    label = f"{int(value)}/6"
                    color = "white" if value < 2 or value > 4 else "#111827"
                axis.text(
                    column_index,
                    row_index,
                    label,
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    color=color,
                )
        for record in summary.to_dict("records"):
            if str(record["combo_id"]) not in shortlist_ids:
                continue
            row_index = period_indices.index(int(record["space_period_index"]))
            column_index = period_indices.index(int(record["enter_period_index"]))
            axis.add_patch(
                Rectangle(
                    (column_index - 0.5, row_index - 0.5),
                    1.0,
                    1.0,
                    fill=False,
                    edgecolor="#EF4444",
                    linewidth=2.0,
                )
            )
        colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        if maximum == 1.0:
            colorbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        else:
            colorbar.set_ticks(range(0, 7))
    figure.legend(
        handles=[
            Patch(
                facecolor="none",
                edgecolor="#EF4444",
                linewidth=2.0,
                label="Frozen top-five cell",
            )
        ],
        loc="lower center",
        frameon=False,
    )
    figure.suptitle("Global OneClick Space/Enter screening heatmaps")
    outputs = []
    for suffix in ["png", "pdf"]:
        path = plot_dir / f"global_screen_heatmaps.{suffix}"
        figure.savefig(
            path,
            dpi=200 if suffix == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
        outputs.append(path)
    plt.close(figure)
    return outputs

def create_failure_reason_heatmap(
    run_dir: Path,
    summary: pd.DataFrame,
    period_indices: list[int],
) -> list[Path]:
    configure_plot_style()
    plot_dir = run_dir / "plots"
    labels = [
        f"{saved_period_seconds(summary, index):.1f}"
        for index in period_indices
    ]
    reasons = sorted(summary["dominant_failure_reason"].astype(str).unique())
    reason_index = {reason: index for index, reason in enumerate(reasons)}
    matrix = np.full((len(period_indices), len(period_indices)), np.nan)
    position = {
        period_index: offset
        for offset, period_index in enumerate(period_indices)
    }
    for row in summary.to_dict("records"):
        matrix[
            position[int(row["space_period_index"])],
            position[int(row["enter_period_index"])],
        ] = reason_index[str(row["dominant_failure_reason"])]
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 0.9, max(len(reasons), 1)))
    from matplotlib.colors import ListedColormap

    figure, axis = plt.subplots(figsize=(9.5, 7.2))
    axis.imshow(
        np.ma.masked_invalid(matrix),
        cmap=ListedColormap(colors),
        vmin=-0.5,
        vmax=max(len(reasons) - 0.5, 0.5),
        aspect="equal",
    )
    axis.set_xticks(range(len(period_indices)), labels)
    axis.set_yticks(range(len(period_indices)), labels)
    axis.set_xlabel("Enter / Undo period (s)")
    axis.set_ylabel("Space period (s)")
    axis.set_title("Dominant global failure reason by clock combination")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            if np.isnan(matrix[row_index, column_index]):
                continue
            reason = reasons[int(matrix[row_index, column_index])]
            abbreviation = "".join(
                token[0].upper() for token in reason.split("_") if token
            )
            axis.text(
                column_index,
                row_index,
                abbreviation,
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="white",
            )
    axis.legend(
        handles=[
            Patch(facecolor=colors[index], label=reason)
            for reason, index in reason_index.items()
        ],
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
        fontsize=8,
    )
    figure.subplots_adjust(left=0.1, right=0.72, top=0.9, bottom=0.1)
    outputs = []
    for suffix in ["png", "pdf"]:
        path = plot_dir / f"global_screen_dominant_failures.{suffix}"
        figure.savefig(
            path,
            dpi=200 if suffix == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
        outputs.append(path)
    plt.close(figure)
    return outputs

def create_confirmation_plots(
    run_dir: Path,
    phrase_results: pd.DataFrame,
    per_user_summary: pd.DataFrame,
    global_summary: pd.DataFrame,
    global_curves: pd.DataFrame,
    global_trial_summary: pd.DataFrame,
    reliability_floor: float,
) -> list[Path]:
    configure_plot_style()
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    ranked = rank_global_summary(global_summary)
    combo_ids = ranked["combo_id"].astype(str).tolist()
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 0.8, len(combo_ids)))

    figure, axis = plt.subplots(figsize=(10.5, 6.2))
    for color, combo_id in zip(colors, combo_ids):
        curve = global_curves[
            global_curves["combo_id"].astype(str) == combo_id
        ].sort_values(["simulated_time_s", "cumulative_completed"])
        row = ranked[ranked["combo_id"].astype(str) == combo_id].iloc[0]
        axis.step(
            curve["simulated_time_s"],
            curve["cumulative_completion_rate"],
            where="post",
            linewidth=2.0,
            color=color,
            label=(
                f"S {row['space_period_s']:.1f}s / "
                f"E {row['enter_period_s']:.1f}s — "
                f"{row['macro_phrase_completion_rate'] * 100:.0f}%"
            ),
        )
    endpoint = float(global_curves["plot_endpoint_s"].max())
    axis.set_xlim(0.0, endpoint)
    axis.set_ylim(0.0, 1.05)
    axis.xaxis.set_major_locator(MultipleLocator(30))
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    axis.set_xlabel("Simulated clock-interaction time (s)")
    axis.set_ylabel("All user/phrase trials completed by time")
    axis.set_title("Global confirmed finalists — completion by time")
    axis.grid(alpha=0.18)
    axis.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=8,
    )
    figure.subplots_adjust(left=0.1, right=0.72, top=0.9, bottom=0.12)
    for suffix in ["png", "pdf"]:
        path = plot_dir / f"global_confirmed_completion_curves.{suffix}"
        figure.savefig(
            path,
            dpi=200 if suffix == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
        outputs.append(path)
    plt.close(figure)

    labels = [
        f"S {row.space_period_s:.1f} / E {row.enter_period_s:.1f} s"
        for row in ranked.itertuples()
    ]
    positions = np.arange(len(ranked))
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    axes[0].bar(
        positions,
        ranked["macro_phrase_completion_rate"],
        color="#4C9F70",
    )
    axes[0].plot(
        positions,
        ranked["worst_user_completion_rate"],
        color="#B91C1C",
        marker="o",
        linewidth=2.0,
        label="Worst user",
    )
    axes[0].axhline(
        reliability_floor,
        color="#D97706",
        linestyle="--",
        linewidth=1.5,
        label="Reliability floor",
    )
    axes[0].set_title("Macro and worst-user completion")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].bar(
        positions,
        ranked["macro_normalized_auc_180"],
        color="#3979A8",
    )
    axes[1].set_title("Macro completion-by-time AUC")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    for axis in axes:
        axis.set_xticks(positions, labels, rotation=22, ha="right")
        axis.grid(axis="y", alpha=0.18)
    figure.suptitle("Global confirmed OneClick candidates")
    figure.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.86,
        bottom=0.25,
        wspace=0.25,
    )
    for suffix in ["png", "pdf"]:
        path = plot_dir / f"global_confirmed_candidate_comparison.{suffix}"
        figure.savefig(
            path,
            dpi=200 if suffix == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
        outputs.append(path)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10.5, 5.8))
    for position, (color, combo_id) in enumerate(zip(colors, combo_ids)):
        values = global_trial_summary[
            global_trial_summary["combo_id"].astype(str) == combo_id
        ].sort_values("trial")["macro_phrase_completion_rate"].to_numpy(float)
        offsets = np.linspace(-0.1, 0.1, len(values))
        axis.scatter(
            np.full(len(values), position) + offsets,
            values,
            color=color,
            s=42,
            zorder=3,
        )
        axis.hlines(
            values.mean(),
            position - 0.22,
            position + 0.22,
            color="#111827",
            linewidth=2.0,
        )
    axis.set_xticks(positions, labels, rotation=22, ha="right")
    axis.set_ylim(0.0, 1.05)
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    axis.set_ylabel("Macro completion per paired trial")
    axis.set_title("Global finalist stability across five trials")
    axis.grid(axis="y", alpha=0.18)
    figure.subplots_adjust(left=0.1, right=0.98, top=0.9, bottom=0.24)
    for suffix in ["png", "pdf"]:
        path = plot_dir / f"global_confirmed_trial_stability.{suffix}"
        figure.savefig(
            path,
            dpi=200 if suffix == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
        outputs.append(path)
    plt.close(figure)

    for user_id, user_summary in per_user_summary.groupby("user_id", sort=False):
        ordered = ranked[["combo_id"]].merge(
            user_summary,
            on="combo_id",
            how="left",
            validate="one_to_one",
        )
        user_labels = [
            f"S {row.space_period_s:.1f} / E {row.enter_period_s:.1f} s"
            for row in ordered.itertuples()
        ]
        figure, axes = plt.subplots(1, 2, figsize=(12, 5.2))
        axes[0].bar(
            positions,
            ordered["phrase_completion_rate"],
            color="#4C9F70",
        )
        axes[0].axhline(
            reliability_floor,
            color="#D97706",
            linestyle="--",
            linewidth=1.5,
        )
        axes[0].set_title("Final phrase completion")
        axes[0].set_ylim(0.0, 1.05)
        axes[0].yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        axes[1].bar(
            positions,
            ordered["normalized_auc_180"],
            color="#3979A8",
        )
        axes[1].set_title("Completion-by-time AUC")
        axes[1].set_ylim(0.0, 1.0)
        axes[1].yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        for axis in axes:
            axis.set_xticks(positions, user_labels, rotation=22, ha="right")
            axis.grid(axis="y", alpha=0.18)
        figure.suptitle(f"User {user_id} — confirmed global finalists")
        figure.subplots_adjust(
            left=0.08,
            right=0.98,
            top=0.86,
            bottom=0.25,
            wspace=0.25,
        )
        for suffix in ["png", "pdf"]:
            path = plot_dir / f"user_{user_id}_global_finalists.{suffix}"
            figure.savefig(
                path,
                dpi=200 if suffix == "png" else None,
                bbox_inches="tight",
                facecolor="white",
            )
            outputs.append(path)
        plt.close(figure)

        failed = phrase_results[
            (phrase_results["user_id"] == user_id)
            & ~phrase_results["phrase_completed"].fillna(False).astype(bool)
        ]
        table = (
            failed.groupby(["combo_id", "phrase_failure_reason"], dropna=False)
            .size()
            .unstack(fill_value=0)
            .reindex(combo_ids, fill_value=0)
        )
        attempts_by_combo = (
            phrase_results[phrase_results["user_id"] == user_id]
            .groupby("combo_id")
            .size()
            .reindex(combo_ids)
        )
        table = table.div(attempts_by_combo, axis=0)
        figure, axis = plt.subplots(figsize=(10.5, 5.8))
        bottom = np.zeros(len(table))
        failure_colors = plt.get_cmap("Set2")(
            np.linspace(0.0, 0.9, max(len(table.columns), 1))
        )
        for color, reason in zip(failure_colors, table.columns):
            values = table[reason].to_numpy(float)
            axis.bar(
                positions,
                values,
                bottom=bottom,
                label=str(reason),
                color=color,
            )
            bottom += values
        axis.set_xticks(positions, user_labels, rotation=22, ha="right")
        axis.set_ylim(0.0, 1.0)
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        axis.set_ylabel("Share of phrase trials")
        axis.set_title(f"User {user_id} — confirmed finalist failure reasons")
        if len(table.columns):
            axis.legend(
                frameon=False,
                fontsize=8,
                loc="upper left",
                bbox_to_anchor=(1.01, 1.0),
            )
        axis.grid(axis="y", alpha=0.18)
        figure.subplots_adjust(left=0.1, right=0.72, top=0.9, bottom=0.24)
        for suffix in ["png", "pdf"]:
            path = plot_dir / f"user_{user_id}_global_finalist_failures.{suffix}"
            figure.savefig(
                path,
                dpi=200 if suffix == "png" else None,
                bbox_inches="tight",
                facecolor="white",
            )
            outputs.append(path)
        plt.close(figure)
    return outputs

def validate_stage_outputs(
    phrase_results: pd.DataFrame,
    per_user_summary: pd.DataFrame,
    global_summary: pd.DataFrame,
    global_curves: pd.DataFrame,
    manifest: pd.DataFrame,
    users: list[str],
    combos: list[dict[str, Any]],
    trials: list[int],
    phrase_count: int,
) -> None:
    expected_conditions = len(users) * len(combos) * len(trials)
    if len(manifest) != expected_conditions:
        raise ValueError("Manifest condition count is incorrect")
    if not manifest["status"].eq("completed").all():
        raise ValueError("Manifest contains incomplete conditions")
    if len(phrase_results) != expected_conditions * phrase_count:
        raise ValueError("Phrase result row count is incorrect")
    counts = phrase_results.groupby(["user_id", "combo_id", "trial"]).size()
    if (counts != phrase_count).any():
        raise ValueError("A condition does not contain the shared phrase count")
    completed = phrase_results["phrase_completed"].fillna(False).astype(bool)
    typed = phrase_results["Typed Text"].fillna("").astype(str).str.rstrip()
    target = phrase_results["Target Phrase"].fillna("").astype(str).str.rstrip()
    if not typed[completed].eq(target[completed]).all():
        raise ValueError("Completed phrase text does not match the target")
    validate_phrase_timing(phrase_results)
    schedule_counts = phrase_results.groupby(
        ["user_id", "trial"]
    ).paired_click_schedule_id.nunique()
    if (schedule_counts != 1).any():
        raise ValueError("Conditions do not share one schedule per user/trial")
    per_user_counts = per_user_summary.groupby("combo_id").user_id.nunique()
    if (per_user_counts != len(users)).any():
        raise ValueError("Global cells do not contain every user")
    if len(global_summary) != len(combos):
        raise ValueError("Global summary does not contain every cell")
    endpoints = global_curves[global_curves["event_type"] == "endpoint"][
        ["combo_id", "cumulative_completion_rate"]
    ]
    pooled = (
        phrase_results.groupby("combo_id").phrase_completed.mean().rename(
            "pooled_completion"
        )
    )
    merged = endpoints.merge(
        pooled,
        on="combo_id",
        validate="one_to_one",
    )
    if not np.allclose(
        merged["cumulative_completion_rate"],
        merged["pooled_completion"],
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("Global curve plateau does not match completion rate")
    recomputed_macro = (
        per_user_summary.groupby("combo_id").phrase_completion_rate.mean()
    )
    recorded_macro = global_summary.set_index(
        "combo_id"
    ).macro_phrase_completion_rate
    if not np.allclose(
        recomputed_macro.sort_index(),
        recorded_macro.sort_index(),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("Macro completion does not equal per-user aggregation")
