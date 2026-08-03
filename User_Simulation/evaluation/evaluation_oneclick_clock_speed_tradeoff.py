"""Sweep fixed OneClick clock periods and plot completion by simulated time.

Run from the repository root:

    python -m User_Simulation.evaluation.evaluation_oneclick_clock_speed_tradeoff
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter

from OneClick_Core import config as oneclick_config
from User_Simulation.evaluation.evaluation_baseline import (
    REPO_ROOT,
    load_text_click_data,
    load_text_phrase_data,
    normalize_phrase_order,
)
from User_Simulation.evaluation.evaluation_nomon_oneclick_bootstrap_comparison import (
    bootstrap_profile_row,
    build_shared_bootstrap_click_df,
    clean_click_rows,
)
from User_Simulation.evaluation.evaluation_nomon_oneclick_comparison import (
    DEFAULT_USERS,
    normalize_system_results,
    parse_csv_values,
    run_oneclick,
)
from User_Simulation.evaluation.evaluation_oneclick_phrase_audit import (
    CachedOneClickWordClient,
    load_real_phrase_candidates,
    run_audit,
)
from User_Simulation.evaluation.synthetic_profiles import (
    CLICK_OFFSET_COL,
    CLOCK_PERIOD_COL,
    DEAD_TIME_COL,
    estimate_profile,
    select_sessions,
    split_sessions,
)


DEFAULT_PERIOD_INDICES = (14, 10, 6)
DEFAULT_TRIALS = 5
DEFAULT_PHRASES = 20
DEFAULT_SEED = 12345
PERIOD_COLORS = ("#E45756", "#4C78A8", "#54A24B")
REQUIRED_CONDITION_COLUMNS = {
    "user_id",
    "trial",
    "clock_period_index",
    "clock_period_s",
    "Comparison Phrase ID",
    "Target Phrase",
    "Typed Text",
    "phrase_completed",
    "simulated_attempt_time_s",
    "simulated_completion_time_s",
}


def build_output_dir(base_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    output_dir = base_dir / f"oneclick_clock_speed_tradeoff_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def parse_int_values(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


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


def period_records(period_indices: list[int]) -> list[dict[str, Any]]:
    if len(period_indices) != 3:
        raise ValueError("the initial speed sweep requires exactly three periods")
    if len(period_indices) != len(set(period_indices)):
        raise ValueError("clock period indices must be unique")
    records = []
    for color, period_index in zip(PERIOD_COLORS, period_indices):
        if period_index < 0 or period_index >= len(oneclick_config.period_li):
            raise ValueError(f"clock period index out of range: {period_index}")
        period_s = float(oneclick_config.period_li[period_index])
        records.append(
            {
                "clock_period_index": int(period_index),
                "clock_period_s": period_s,
                "clock_period_label": f"{period_s:.1f} s clock",
                "color": color,
            }
        )
    return records


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


def condition_path(
    run_dir: Path,
    user_id: str,
    period_index: int,
    trial: int,
) -> Path:
    return (
        run_dir
        / "conditions"
        / f"user_{user_id}"
        / f"period_{period_index:02d}"
        / f"trial_{trial:02d}.csv"
    )


def schedule_path(run_dir: Path, user_id: str, trial: int) -> Path:
    return run_dir / "paired_click_schedules" / f"user_{user_id}_trial_{trial:02d}.csv"


def condition_is_complete(
    path: Path,
    user_id: str,
    period_index: int,
    trial: int,
    phrase_ids: set[str],
) -> bool:
    if not path.is_file():
        return False
    try:
        frame = pd.read_csv(path)
    except Exception:
        return False
    if not REQUIRED_CONDITION_COLUMNS.issubset(frame.columns):
        return False
    if len(frame) != len(phrase_ids):
        return False
    if set(frame["Comparison Phrase ID"].astype(str)) != phrase_ids:
        return False
    completed = frame["phrase_completed"].fillna(False).astype(bool)
    typed = frame["Typed Text"].fillna("").astype(str).str.rstrip()
    target = frame["Target Phrase"].fillna("").astype(str).str.rstrip()
    if not typed[completed].eq(target[completed]).all():
        return False
    return bool(
        (frame["user_id"].astype(str) == str(user_id)).all()
        and (pd.to_numeric(frame["clock_period_index"]) == period_index).all()
        and (pd.to_numeric(frame["trial"]) == trial).all()
    )


def build_manifest(
    run_dir: Path,
    users: list[str],
    periods: list[dict[str, Any]],
    trials: int,
    phrase_ids: set[str],
) -> pd.DataFrame:
    rows = []
    for user_id in users:
        for period in periods:
            period_index = int(period["clock_period_index"])
            for trial in range(trials):
                path = condition_path(run_dir, user_id, period_index, trial)
                rows.append(
                    {
                        "user_id": user_id,
                        "clock_period_index": period_index,
                        "clock_period_s": float(period["clock_period_s"]),
                        "trial": trial,
                        "status": (
                            "completed"
                            if condition_is_complete(
                                path,
                                user_id,
                                period_index,
                                trial,
                                phrase_ids,
                            )
                            else "pending"
                        ),
                        "condition_file": str(path.relative_to(run_dir)),
                    }
                )
    return pd.DataFrame(rows)


def load_or_build_schedule(
    run_dir: Path,
    user_id: str,
    trial: int,
    profile: dict[str, Any],
    phrase_df: pd.DataFrame,
    train_click_df: pd.DataFrame,
    seed: int,
    clicks_per_phrase: int,
    calibration_clicks: int,
) -> pd.DataFrame:
    path = schedule_path(run_dir, user_id, trial)
    if path.is_file():
        return pd.read_csv(path)
    click_df = build_shared_bootstrap_click_df(
        profile=profile,
        phrase_df=phrase_df,
        train_click_df=train_click_df,
        trial=trial,
        seed=seed,
        clicks_per_phrase=clicks_per_phrase,
        calibration_clicks=calibration_clicks,
    )
    active = click_df["Session Num"].notna()
    click_df.loc[active, DEAD_TIME_COL] = 0.0
    atomic_write_csv(click_df, path)
    return click_df


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


def build_common_phrase_files(
    run_dir: Path,
    phrase_count: int,
    seed: int,
    cache_dir: Path,
    audit_workers: int,
) -> pd.DataFrame:
    phrase_set_path = run_dir / "common_phrase_set.csv"
    if phrase_set_path.is_file():
        return pd.read_csv(phrase_set_path)

    candidates = load_real_phrase_candidates()
    client = CachedOneClickWordClient(cache_dir)
    phrase_audit, word_audit = run_audit(
        candidates,
        client,
        workers=audit_workers,
    )
    selected = select_common_prediction_reachable_phrases(
        phrase_audit,
        phrase_count,
        seed,
    )
    atomic_write_csv(phrase_audit, run_dir / "common_phrase_reachability_audit.csv")
    atomic_write_csv(word_audit, run_dir / "common_phrase_word_reachability_audit.csv")
    atomic_write_csv(selected, phrase_set_path)
    return selected


def load_user_profile(
    user_id: str,
    validation_fraction: float,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    real_click_df = clean_click_rows(load_text_click_data(user_id))
    real_phrase_df = normalize_phrase_order(load_text_phrase_data(user_id))
    train_sessions, validation_sessions = split_sessions(
        real_click_df,
        validation_fraction,
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
    profile = estimate_profile(
        user_id,
        train_click_df,
        train_phrase_df,
        validation_click_df,
        validation_phrase_df,
        train_sessions,
        validation_sessions,
    )
    profile_row = bootstrap_profile_row(
        user_id,
        profile,
        train_sessions,
        train_click_df,
    )
    split_config = {
        "train_sessions": [int(value) for value in train_sessions],
        "validation_sessions": [int(value) for value in validation_sessions],
    }
    return profile, train_click_df, {**profile_row, **split_config}


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--users", default=",".join(DEFAULT_USERS))
    parser.add_argument(
        "--period-indices",
        default=",".join(str(value) for value in DEFAULT_PERIOD_INDICES),
    )
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--phrases", type=int, default=DEFAULT_PHRASES)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--clicks-per-phrase", type=int, default=500)
    parser.add_argument("--calibration-clicks", type=int, default=200)
    parser.add_argument("--max-word-attempts", type=int, default=5)
    parser.add_argument("--max-enter-attempts", type=int, default=5)
    parser.add_argument("--max-clicks-per-word", type=int, default=30)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--audit-workers", type=int, default=1)
    parser.add_argument(
        "--oneclick-cache-dir",
        type=Path,
        default=REPO_ROOT / ".cache" / "oneclick_phrase_audit",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
    )
    parser.add_argument("--resume-run-dir", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def run_sweep(args: argparse.Namespace) -> Path:
    if args.resume_run_dir is not None:
        run_dir = args.resume_run_dir.resolve()
        config_path = run_dir / "run_config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Resume directory lacks run_config.json: {run_dir}")
        saved = json.loads(config_path.read_text(encoding="utf-8"))
        users = [str(value) for value in saved["users"]]
        period_indices = [int(value) for value in saved["period_indices"]]
        trials = int(saved["trials"])
        phrase_count = int(saved["phrase_count"])
        validation_fraction = float(saved["validation_fraction"])
        clicks_per_phrase = int(saved["clicks_per_phrase"])
        calibration_clicks = int(saved["calibration_clicks"])
        max_word_attempts = int(saved["max_word_attempts"])
        max_enter_attempts = int(saved["max_enter_attempts"])
        max_clicks_per_word = int(saved["max_clicks_per_word"])
        seed = int(saved["seed"])
        cache_dir = Path(saved["oneclick_cache_dir"])
    else:
        users = parse_csv_values(args.users)
        period_indices = parse_int_values(args.period_indices)
        trials = int(args.trials)
        phrase_count = int(args.phrases)
        validation_fraction = float(args.validation_fraction)
        clicks_per_phrase = int(args.clicks_per_phrase)
        calibration_clicks = int(args.calibration_clicks)
        max_word_attempts = int(args.max_word_attempts)
        max_enter_attempts = int(args.max_enter_attempts)
        max_clicks_per_word = int(args.max_clicks_per_word)
        seed = int(args.seed)
        cache_dir = args.oneclick_cache_dir.resolve()
        run_dir = build_output_dir(args.output_dir)

    if not users or trials < 1 or phrase_count < 1:
        raise ValueError("users, trials, and phrases must be non-empty/positive")
    if args.audit_workers < 1:
        raise ValueError("--audit-workers must be at least 1")
    periods = period_records(period_indices)
    phrase_set = build_common_phrase_files(
        run_dir,
        phrase_count,
        seed,
        cache_dir,
        args.audit_workers,
    )
    if len(phrase_set) != phrase_count:
        raise ValueError("saved common phrase set does not match configured phrase count")
    phrase_ids = set(phrase_set["Comparison Phrase ID"].astype(str))

    run_config = {
        "experiment": "oneclick_clock_speed_completion_by_time",
        "users": users,
        "period_indices": period_indices,
        "periods": periods,
        "trials": trials,
        "phrase_count": phrase_count,
        "phrase_attempts_per_speed": trials * phrase_count,
        "phrase_attempts_per_user": len(periods) * trials * phrase_count,
        "total_phrase_attempts": len(users) * len(periods) * trials * phrase_count,
        "phrase_set_checksum": phrase_set_checksum(phrase_set),
        "phrase_policy": "all_words_prediction_reachable",
        "validation_fraction": validation_fraction,
        "clicks_per_phrase": clicks_per_phrase,
        "calibration_clicks": calibration_clicks,
        "max_word_attempts": max_word_attempts,
        "max_enter_attempts": max_enter_attempts,
        "max_clicks_per_word": max_clicks_per_word,
        "undo_mode": "protected",
        "fixed_period_for_space_and_enter": True,
        "offset_transfer": "paired_absolute_seconds",
        "dead_time_mode": "zero_active_dead_time",
        "phrase_time_ceiling_s": None,
        "seed": seed,
        "oneclick_cache_dir": str(cache_dir),
    }
    atomic_write_json(run_config, run_dir / "run_config.json")

    manifest = build_manifest(run_dir, users, periods, trials, phrase_ids)
    atomic_write_csv(manifest, run_dir / "condition_manifest.csv")
    profile_rows = []

    simulation_phrase_df = phrase_set[
        [
            "Session Num",
            "Phrase Num",
            "Phrase Text",
            "Comparison Phrase ID",
        ]
    ].copy()

    for user_id in users:
        profile, train_click_df, profile_row = load_user_profile(
            user_id,
            validation_fraction,
        )
        profile_rows.append(profile_row)
        for trial in range(trials):
            trial_seed = seed + trial + sum(user_id.encode("utf-8")) * 1000
            base_click_df = load_or_build_schedule(
                run_dir,
                user_id,
                trial,
                profile,
                simulation_phrase_df,
                train_click_df,
                trial_seed,
                clicks_per_phrase,
                calibration_clicks,
            )
            schedule_id = f"{user_id}_trial_{trial:02d}"
            for period in periods:
                period_index = int(period["clock_period_index"])
                path = condition_path(run_dir, user_id, period_index, trial)
                if condition_is_complete(
                    path,
                    user_id,
                    period_index,
                    trial,
                    phrase_ids,
                ):
                    print(
                        f"Skipping completed condition: user {user_id}, "
                        f"{period['clock_period_label']}, trial {trial + 1}/{trials}"
                    )
                    continue
                print(
                    f"Running user {user_id}, {period['clock_period_label']}, "
                    f"trial {trial + 1}/{trials}"
                )
                click_df = apply_fixed_period(
                    base_click_df,
                    float(period["clock_period_s"]),
                )
                raw_results = run_oneclick(
                    click_df=click_df,
                    phrase_df=simulation_phrase_df,
                    max_word_attempts=max_word_attempts,
                    max_enter_attempts=max_enter_attempts,
                    max_clicks_per_word=max_clicks_per_word,
                    undo_mode="protected",
                    oneclick_cache_dir=cache_dir,
                    perfect_letter_observations=False,
                    verbose=args.verbose,
                    fixed_clock_period_s=float(period["clock_period_s"]),
                )
                condition = normalize_condition_results(
                    raw_results,
                    user_id,
                    trial,
                    period,
                    schedule_id,
                )
                validate_phrase_timing(condition)
                if set(condition["Comparison Phrase ID"].astype(str)) != phrase_ids:
                    raise ValueError("condition did not return the common phrase set")
                atomic_write_csv(condition, path)
                manifest = build_manifest(
                    run_dir,
                    users,
                    periods,
                    trials,
                    phrase_ids,
                )
                atomic_write_csv(manifest, run_dir / "condition_manifest.csv")

    atomic_write_csv(pd.DataFrame(profile_rows), run_dir / "user_bootstrap_profiles.csv")
    manifest = build_manifest(run_dir, users, periods, trials, phrase_ids)
    incomplete = manifest[manifest["status"] != "completed"]
    atomic_write_csv(manifest, run_dir / "condition_manifest.csv")
    if not incomplete.empty:
        raise RuntimeError(f"{len(incomplete)} sweep conditions are incomplete")

    condition_frames = [
        pd.read_csv(run_dir / relative_path)
        for relative_path in manifest["condition_file"]
    ]
    phrase_results = pd.concat(condition_frames, ignore_index=True)
    summary = build_speed_summary(phrase_results)
    curves = build_curve_points(phrase_results)
    validate_final_outputs(
        phrase_results,
        summary,
        curves,
        users,
        periods,
        trials,
        phrase_count,
    )
    atomic_write_csv(phrase_results, run_dir / "clock_speed_phrase_results.csv")
    atomic_write_csv(summary, run_dir / "clock_speed_summary.csv")
    atomic_write_csv(curves, run_dir / "clock_speed_curve_points.csv")
    plot_outputs = create_plots(run_dir, curves, users, periods)
    print(f"Saved OneClick clock-speed sweep to: {run_dir}")
    print(summary.to_string(index=False))
    for output in plot_outputs:
        print(output)
    return run_dir


def main() -> None:
    run_sweep(parse_args())


if __name__ == "__main__":
    main()
