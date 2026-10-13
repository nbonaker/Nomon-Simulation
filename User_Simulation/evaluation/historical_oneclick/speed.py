"""Historical QuickClick data/analysis extracted from evaluation_oneclick_clock_speed_tradeoff.

No simulator execution. Saved result semantics are preserved.
"""
from __future__ import annotations

import hashlib

import json

import math

import os

from pathlib import Path

from typing import Any

import numpy as np

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.ticker import MultipleLocator, PercentFormatter

from User_Simulation.evaluation.historical_oneclick.comparison import normalize_system_results

from User_Simulation.evaluation.synthetic_profiles import CLOCK_PERIOD_COL, DEAD_TIME_COL

def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp_path, index=False)
    os.replace(temp_path, path)

def atomic_write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temp_path, path)

def select_common_prediction_reachable_phrases(
    phrase_audit_df: pd.DataFrame,
    phrase_count: int,
    seed: int,
) -> pd.DataFrame:
    """Select one deterministic phrase from each equal-sized length stratum."""
    if phrase_count < 1:
        raise ValueError("phrase_count must be at least 1")
    reachable = phrase_audit_df[
        phrase_audit_df["all_words_prediction_reachable"].fillna(False).astype(bool)
    ].copy()
    reachable["target_character_count"] = pd.to_numeric(
        reachable["target_character_count"],
        errors="raise",
    )
    reachable = reachable.sort_values(
        ["target_character_count", "phrase_id"],
        kind="stable",
    ).reset_index(drop=True)
    if len(reachable) < phrase_count:
        raise ValueError(
            f"Only {len(reachable)} strictly prediction-reachable phrases are "
            f"available; {phrase_count} requested"
        )

    rng = np.random.default_rng(seed)
    strata = np.array_split(np.arange(len(reachable)), phrase_count)
    selected_indices = [int(rng.choice(stratum)) for stratum in strata]
    selected = reachable.iloc[selected_indices].copy()
    selected = selected.sort_values(
        ["target_character_count", "phrase_id"],
        kind="stable",
    ).reset_index(drop=True)
    selected.insert(
        0,
        "Comparison Phrase ID",
        [f"speed_phrase_{index:02d}" for index in range(1, len(selected) + 1)],
    )
    selected["Session Num"] = np.arange(1, len(selected) + 1)
    selected["Phrase Num"] = 1
    selected["Phrase Text"] = selected["phrase_text"]
    return selected

def phrase_set_checksum(phrase_df: pd.DataFrame) -> str:
    payload = "\n".join(
        f"{row['Comparison Phrase ID']}\t{row['Phrase Text']}"
        for row in phrase_df.to_dict("records")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def apply_fixed_period(click_df: pd.DataFrame, period_s: float) -> pd.DataFrame:
    result = click_df.copy()
    result[CLOCK_PERIOD_COL] = float(period_s)
    active = result["Session Num"].notna()
    result.loc[active, DEAD_TIME_COL] = 0.0
    return result

def normalize_condition_results(
    raw_results: pd.DataFrame,
    user_id: str,
    trial: int,
    period: dict[str, Any],
    schedule_id: str,
) -> pd.DataFrame:
    result = normalize_system_results(raw_results, "oneclick", user_id, trial)
    result["clock_period_index"] = int(period["clock_period_index"])
    result["clock_period_s"] = float(period["clock_period_s"])
    result["clock_period_label"] = str(period["clock_period_label"])
    result["paired_click_schedule_id"] = schedule_id
    return result

def validate_phrase_timing(frame: pd.DataFrame) -> None:
    attempted = pd.to_numeric(frame["simulated_attempt_time_s"], errors="coerce")
    completed = frame["phrase_completed"].fillna(False).astype(bool)
    completion = pd.to_numeric(frame["simulated_completion_time_s"], errors="coerce")
    stage_total = (
        pd.to_numeric(frame["letter_clock_time_s"], errors="coerce")
        + pd.to_numeric(frame["target_enter_clock_time_s"], errors="coerce")
        + pd.to_numeric(frame["undo_clock_time_s"], errors="coerce")
    )
    if attempted.isna().any() or (~np.isfinite(attempted)).any() or (attempted < 0).any():
        raise ValueError("condition contains invalid simulated attempt times")
    if completion[completed].isna().any() or completion[~completed].notna().any():
        raise ValueError("completion times must exist only for completed phrases")
    typed = frame["Typed Text"].fillna("").astype(str).str.rstrip()
    target = frame["Target Phrase"].fillna("").astype(str).str.rstrip()
    if not typed[completed].eq(target[completed]).all():
        raise ValueError("completed phrases must exactly match their target text")
    if not np.allclose(attempted, stage_total, rtol=0.0, atol=1e-8):
        raise ValueError("stage time totals do not equal simulated attempt time")

def build_speed_summary(phrase_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = [
        "user_id",
        "clock_period_index",
        "clock_period_s",
        "clock_period_label",
    ]
    for key, group in phrase_results.groupby(group_cols, sort=False):
        user_id, period_index, period_s, period_label = key
        completed = group["phrase_completed"].fillna(False).astype(bool)
        completion_times = pd.to_numeric(
            group.loc[completed, "simulated_completion_time_s"],
            errors="coerce",
        )
        rows.append(
            {
                "user_id": user_id,
                "clock_period_index": int(period_index),
                "clock_period_s": float(period_s),
                "clock_period_label": period_label,
                "phrase_attempts": int(len(group)),
                "completed_phrases": int(completed.sum()),
                "phrase_completion_rate": float(completed.mean()),
                "median_completion_time_s": (
                    float(completion_times.median())
                    if not completion_times.empty
                    else np.nan
                ),
                "maximum_completion_time_s": (
                    float(completion_times.max())
                    if not completion_times.empty
                    else np.nan
                ),
                "total_attempt_time_s": float(
                    pd.to_numeric(group["simulated_attempt_time_s"]).sum()
                ),
            }
        )
    return pd.DataFrame(rows)

def build_curve_points(phrase_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for user_id, user_frame in phrase_results.groupby("user_id", sort=False):
        successful_times = pd.to_numeric(
            user_frame.loc[
                user_frame["phrase_completed"].fillna(False).astype(bool),
                "simulated_completion_time_s",
            ],
            errors="coerce",
        ).dropna()
        endpoint_s = (
            max(30.0, math.ceil(float(successful_times.max()) / 30.0) * 30.0)
            if not successful_times.empty
            else 30.0
        )
        group_cols = [
            "clock_period_index",
            "clock_period_s",
            "clock_period_label",
        ]
        for key, group in user_frame.groupby(group_cols, sort=False):
            period_index, period_s, period_label = key
            attempts = int(len(group))
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
                    "user_id": user_id,
                    "clock_period_index": int(period_index),
                    "clock_period_s": float(period_s),
                    "clock_period_label": period_label,
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
                        "clock_period_index": int(period_index),
                        "clock_period_s": float(period_s),
                        "clock_period_label": period_label,
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
                    "clock_period_index": int(period_index),
                    "clock_period_s": float(period_s),
                    "clock_period_label": period_label,
                    "simulated_time_s": endpoint_s,
                    "cumulative_completed": int(len(times)),
                    "cumulative_completion_rate": len(times) / attempts,
                    "phrase_attempts": attempts,
                    "event_type": "endpoint",
                    "plot_endpoint_s": endpoint_s,
                }
            )
    return pd.DataFrame(rows)

def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
        }
    )

def plot_user_curve(
    axis,
    user_id: str,
    user_curves: pd.DataFrame,
    periods: list[dict[str, Any]],
    detailed: bool,
) -> None:
    endpoint_s = float(user_curves["plot_endpoint_s"].max())
    final_labels = []
    for period in periods:
        period_curve = user_curves[
            user_curves["clock_period_index"] == int(period["clock_period_index"])
        ].sort_values(
            ["simulated_time_s", "cumulative_completed"],
            kind="stable",
        )
        if period_curve.empty:
            continue
        x = period_curve["simulated_time_s"].to_numpy(float)
        y = period_curve["cumulative_completion_rate"].to_numpy(float)
        axis.step(
            x,
            y,
            where="post",
            linewidth=2.2 if detailed else 1.8,
            color=period["color"],
            label=period["clock_period_label"],
        )
        final_labels.append(
            (
                float(y[-1]),
                period["color"],
                f"{float(period['clock_period_s']):.1f}s: {float(y[-1]) * 100:.0f}%",
            )
        )

    axis.set_xlim(0, endpoint_s)
    axis.set_ylim(0, 1.06)
    axis.xaxis.set_major_locator(MultipleLocator(30))
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    axis.grid(axis="both", alpha=0.18)
    axis.set_title(f"User {user_id}")
    axis.set_xlabel("Simulated clock-interaction time (s)")
    axis.set_ylabel("Phrase trials completed by time")
    if detailed:
        axis.legend(frameon=False, loc="upper left")
    for index, (final_rate, color, label) in enumerate(
        sorted(final_labels, key=lambda item: item[0])
    ):
        offset_points = (index - (len(final_labels) - 1) / 2) * 10
        if final_rate < 0.05 and offset_points < 0:
            offset_points = 0
        axis.annotate(
            label,
            xy=(endpoint_s, final_rate),
            xytext=(-6, offset_points),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=8 if detailed else 7,
            fontweight="bold",
            color=color,
        )

def create_plots(
    run_dir: Path,
    curve_points: pd.DataFrame,
    users: list[str],
    periods: list[dict[str, Any]],
) -> list[Path]:
    configure_plot_style()
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    figure, axes = plt.subplots(2, 3, figsize=(16, 9.5), constrained_layout=False)
    figure.subplots_adjust(
        left=0.065,
        right=0.985,
        top=0.97,
        bottom=0.13,
        hspace=0.36,
        wspace=0.24,
    )
    for axis, user_id in zip(axes.flat, users):
        plot_user_curve(
            axis,
            user_id,
            curve_points[curve_points["user_id"] == user_id],
            periods,
            detailed=False,
        )
    for axis in list(axes.flat)[len(users):]:
        axis.set_visible(False)
    handles = [
        plt.Line2D([0], [0], color=period["color"], linewidth=2.5)
        for period in periods
    ]
    labels = [str(period["clock_period_label"]) for period in periods]
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(periods),
        frameon=False,
        bbox_to_anchor=(0.5, 0.015),
    )
    dashboard_png = plot_dir / "oneclick_completion_by_time_dashboard.png"
    dashboard_pdf = plot_dir / "oneclick_completion_by_time_dashboard.pdf"
    figure.savefig(dashboard_png, dpi=200, bbox_inches="tight", facecolor="white")
    figure.savefig(dashboard_pdf, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    outputs.extend([dashboard_png, dashboard_pdf])

    for user_id in users:
        figure, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
        plot_user_curve(
            axis,
            user_id,
            curve_points[curve_points["user_id"] == user_id],
            periods,
            detailed=True,
        )
        png_path = plot_dir / f"user_{user_id}_completion_by_time.png"
        pdf_path = plot_dir / f"user_{user_id}_completion_by_time.pdf"
        figure.savefig(png_path, dpi=200, bbox_inches="tight", facecolor="white")
        figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        outputs.extend([png_path, pdf_path])
    return outputs

def validate_final_outputs(
    phrase_results: pd.DataFrame,
    summary: pd.DataFrame,
    curves: pd.DataFrame,
    users: list[str],
    periods: list[dict[str, Any]],
    trials: int,
    phrase_count: int,
) -> None:
    expected_per_speed = trials * phrase_count
    expected_per_user = len(periods) * expected_per_speed
    user_counts = phrase_results.groupby("user_id").size()
    if any(int(user_counts.get(user_id, 0)) != expected_per_user for user_id in users):
        raise ValueError("not every user has the expected number of phrase attempts")
    speed_counts = phrase_results.groupby(["user_id", "clock_period_index"]).size()
    if (speed_counts != expected_per_speed).any():
        raise ValueError("not every speed has the expected number of phrase attempts")
    if phrase_results["simulated_attempt_time_s"].isna().any():
        raise ValueError("phrase results contain missing simulated attempt times")
    completed = phrase_results["phrase_completed"].fillna(False).astype(bool)
    typed = phrase_results["Typed Text"].fillna("").astype(str).str.rstrip()
    target = phrase_results["Target Phrase"].fillna("").astype(str).str.rstrip()
    if not typed[completed].eq(target[completed]).all():
        raise ValueError("completed phrase text does not exactly match its target")

    endpoints = curves[curves["event_type"] == "endpoint"].copy()
    merged = summary.merge(
        endpoints[
            [
                "user_id",
                "clock_period_index",
                "cumulative_completion_rate",
            ]
        ],
        on=["user_id", "clock_period_index"],
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
