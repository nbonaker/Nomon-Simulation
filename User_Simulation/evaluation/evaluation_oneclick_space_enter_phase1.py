"""Phase 1 screening of independent OneClick Space and Enter clock periods.

Run from the repository root:

    python -m User_Simulation.evaluation.evaluation_oneclick_space_enter_phase1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
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
from User_Simulation.evaluation.evaluation_baseline import REPO_ROOT
from User_Simulation.evaluation.evaluation_nomon_oneclick_comparison import (
    normalize_system_results,
    parse_csv_values,
    run_oneclick,
)
from User_Simulation.evaluation.evaluation_oneclick_clock_speed_tradeoff import (
    atomic_write_csv,
    atomic_write_json,
    configure_plot_style,
    phrase_set_checksum,
    validate_phrase_timing,
)
from User_Simulation.evaluation.synthetic_profiles import (
    CLICK_OFFSET_COL,
    DEAD_TIME_COL,
)


DEFAULT_USERS = ("A", "C")
DEFAULT_PERIOD_INDICES = (0, 2, 4, 6, 8, 10, 12, 14, 16, 18)
DEFAULT_OFF_DIAGONAL_PAIRS = (
    (6, 10),
    (10, 6),
    (10, 14),
    (14, 10),
    (6, 14),
    (14, 6),
    (0, 18),
    (18, 0),
    (2, 16),
    (16, 2),
    (4, 12),
    (12, 4),
    (0, 10),
    (10, 0),
    (16, 8),
)
DEFAULT_TRIALS = 1
DEFAULT_PHRASES = 20
DEFAULT_SEED = 12345
TIME_HORIZONS_S = (60.0, 120.0, 180.0)
AUC_HORIZON_S = 180.0

REQUIRED_CONDITION_COLUMNS = {
    "user_id",
    "trial",
    "space_period_index",
    "space_period_s",
    "enter_period_index",
    "enter_period_s",
    "Comparison Phrase ID",
    "Target Phrase",
    "Typed Text",
    "phrase_completed",
    "simulated_attempt_time_s",
    "simulated_completion_time_s",
}


def build_output_dir(base_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    output_dir = base_dir / f"oneclick_space_enter_phase1_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def parse_int_values(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_combo_pairs(value: str | None) -> list[tuple[int, int]] | None:
    if value is None:
        return None
    pairs: list[tuple[int, int]] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        pieces = item.split(":")
        if len(pieces) != 2:
            raise ValueError(
                "combo pairs must use SPACE_INDEX:ENTER_INDEX syntax"
            )
        pairs.append((int(pieces[0]), int(pieces[1])))
    if not pairs:
        raise ValueError("at least one combo pair is required")
    return pairs


def period_record(period_index: int) -> dict[str, Any]:
    if period_index < 0 or period_index >= len(oneclick_config.period_li):
        raise ValueError(f"clock period index out of range: {period_index}")
    period_s = float(oneclick_config.period_li[period_index])
    return {
        "period_index": int(period_index),
        "period_s": period_s,
        "period_label": f"{period_s:.1f} s",
    }


def build_combo_records(
    period_indices: list[int],
    explicit_pairs: list[tuple[int, int]] | None = None,
) -> list[dict[str, Any]]:
    if len(period_indices) != len(set(period_indices)):
        raise ValueError("clock period indices must be unique")
    if not period_indices:
        raise ValueError("at least one clock period index is required")
    allowed = set(period_indices)
    pairs = (
        [(index, index) for index in period_indices]
        + list(DEFAULT_OFF_DIAGONAL_PAIRS)
        if explicit_pairs is None
        else list(explicit_pairs)
    )
    if len(pairs) != len(set(pairs)):
        raise ValueError("Space/Enter combinations must be unique")
    invalid = [
        pair for pair in pairs if pair[0] not in allowed or pair[1] not in allowed
    ]
    if invalid:
        raise ValueError(
            f"combination indices are absent from the period set: {invalid}"
        )
    if explicit_pairs is None and (
        tuple(period_indices) != DEFAULT_PERIOD_INDICES or len(pairs) != 25
    ):
        raise ValueError(
            "the default Phase 1 design requires period indices "
            f"{list(DEFAULT_PERIOD_INDICES)}"
        )

    records = []
    for space_index, enter_index in pairs:
        space = period_record(space_index)
        enter = period_record(enter_index)
        records.append(
            {
                "combo_id": f"s{space_index:02d}_e{enter_index:02d}",
                "space_period_index": space_index,
                "space_period_s": space["period_s"],
                "space_period_label": space["period_label"],
                "enter_period_index": enter_index,
                "enter_period_s": enter["period_s"],
                "enter_period_label": enter["period_label"],
                "combo_label": (
                    f"Space {space['period_s']:.1f}s / "
                    f"Enter {enter['period_s']:.1f}s"
                ),
                "is_diagonal": bool(space_index == enter_index),
            }
        )
    return records


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


def schedule_path(run_dir: Path, user_id: str, trial: int) -> Path:
    return run_dir / "paired_click_schedules" / f"user_{user_id}_trial_{trial:02d}.csv"


def condition_is_complete(
    path: Path,
    user_id: str,
    combo: dict[str, Any],
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
        and (
            pd.to_numeric(frame["space_period_index"])
            == int(combo["space_period_index"])
        ).all()
        and (
            pd.to_numeric(frame["enter_period_index"])
            == int(combo["enter_period_index"])
        ).all()
        and (pd.to_numeric(frame["trial"]) == trial).all()
    )


def build_manifest(
    run_dir: Path,
    users: list[str],
    combos: list[dict[str, Any]],
    trials: int,
    phrase_ids: set[str],
) -> pd.DataFrame:
    rows = []
    for user_id in users:
        for combo in combos:
            for trial in range(trials):
                path = condition_path(
                    run_dir,
                    user_id,
                    int(combo["space_period_index"]),
                    int(combo["enter_period_index"]),
                    trial,
                )
                rows.append(
                    {
                        "user_id": user_id,
                        **combo,
                        "trial": trial,
                        "status": (
                            "completed"
                            if condition_is_complete(
                                path,
                                user_id,
                                combo,
                                trial,
                                phrase_ids,
                            )
                            else "pending"
                        ),
                        "condition_file": str(path.relative_to(run_dir)),
                    }
                )
    return pd.DataFrame(rows)


def find_latest_baseline_run(
    output_dir: Path,
    users: list[str],
    trials: int,
) -> Path:
    candidates = sorted(
        output_dir.glob("oneclick_clock_speed_tradeoff_*"),
        reverse=True,
    )
    for candidate in candidates:
        if not (candidate / "common_phrase_set.csv").is_file():
            continue
        if all(
            (
                candidate
                / "paired_click_schedules"
                / f"user_{user_id}_trial_{trial:02d}.csv"
            ).is_file()
            for user_id in users
            for trial in range(trials)
        ):
            return candidate.resolve()
    raise FileNotFoundError(
        "No completed clock-speed run contains the required phrase set and "
        "paired schedules; pass --baseline-run-dir explicitly"
    )


def prepare_baseline_phrase_set(
    run_dir: Path,
    baseline_run_dir: Path,
    phrase_count: int,
) -> pd.DataFrame:
    destination = run_dir / "common_phrase_set.csv"
    if destination.is_file():
        phrase_set = pd.read_csv(destination)
    else:
        source = baseline_run_dir / "common_phrase_set.csv"
        phrase_set = pd.read_csv(source)
        if phrase_count > len(phrase_set):
            raise ValueError(
                f"baseline has {len(phrase_set)} phrases; {phrase_count} requested"
            )
        phrase_set = phrase_set.iloc[:phrase_count].copy()
        atomic_write_csv(phrase_set, destination)
        for audit_name in [
            "common_phrase_reachability_audit.csv",
            "common_phrase_word_reachability_audit.csv",
        ]:
            audit_source = baseline_run_dir / audit_name
            if audit_source.is_file():
                shutil.copy2(audit_source, run_dir / audit_name)
    if len(phrase_set) != phrase_count:
        raise ValueError("saved phrase set does not match configured phrase count")
    if phrase_set["Comparison Phrase ID"].astype(str).duplicated().any():
        raise ValueError("common phrase identifiers must be unique")
    if "all_words_prediction_reachable" in phrase_set and not phrase_set[
        "all_words_prediction_reachable"
    ].fillna(False).astype(bool).all():
        raise ValueError("common phrase set contains a prediction-unreachable phrase")
    return phrase_set


def _frame_checksum(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(index=False, float_format="%.17g")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_or_copy_schedule(
    run_dir: Path,
    baseline_run_dir: Path,
    user_id: str,
    trial: int,
    phrase_sessions: set[int],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    destination = schedule_path(run_dir, user_id, trial)
    source = (
        baseline_run_dir
        / "paired_click_schedules"
        / f"user_{user_id}_trial_{trial:02d}.csv"
    )
    source_frame = pd.read_csv(source)
    calibration = source_frame["Session Num"].isna()
    active_sessions = pd.to_numeric(
        source_frame["Session Num"],
        errors="coerce",
    )
    keep = calibration | active_sessions.isin(phrase_sessions)
    expected = source_frame.loc[keep].copy().reset_index(drop=True)
    active = expected["Session Num"].notna()
    expected.loc[active, DEAD_TIME_COL] = 0.0

    if destination.is_file():
        saved = pd.read_csv(destination)
        if _frame_checksum(saved) != _frame_checksum(expected):
            raise ValueError(
                f"saved schedule differs from baseline for user {user_id}, trial {trial}"
            )
        frame = saved
    else:
        atomic_write_csv(expected, destination)
        frame = expected

    if not np.allclose(
        pd.to_numeric(frame[CLICK_OFFSET_COL], errors="raise"),
        pd.to_numeric(expected[CLICK_OFFSET_COL], errors="raise"),
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError("paired absolute-second offsets changed during schedule copy")
    if (
        pd.to_numeric(frame.loc[frame["Session Num"].notna(), DEAD_TIME_COL])
        != 0.0
    ).any():
        raise ValueError("active dead time must be zero")
    return frame, {
        "user_id": user_id,
        "trial": trial,
        "schedule_id": f"{user_id}_trial_{trial:02d}",
        "source_schedule": str(source.resolve()),
        "row_count": int(len(frame)),
        "active_row_count": int(frame["Session Num"].notna().sum()),
        "offset_checksum": _frame_checksum(
            frame[["Session Num", CLICK_OFFSET_COL]].copy()
        ),
    }


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
        f"{float(oneclick_config.period_li[index]):.1f}" for index in period_indices
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--users", default=",".join(DEFAULT_USERS))
    parser.add_argument(
        "--period-indices",
        default=",".join(str(value) for value in DEFAULT_PERIOD_INDICES),
    )
    parser.add_argument(
        "--combo-pairs",
        default=None,
        help="Optional comma-separated SPACE_INDEX:ENTER_INDEX pairs for smoke tests",
    )
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--phrases", type=int, default=DEFAULT_PHRASES)
    parser.add_argument("--max-word-attempts", type=int, default=5)
    parser.add_argument("--max-enter-attempts", type=int, default=5)
    parser.add_argument("--max-clicks-per-word", type=int, default=30)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
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
    parser.add_argument("--baseline-run-dir", type=Path, default=None)
    parser.add_argument("--resume-run-dir", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def run_phase1(args: argparse.Namespace) -> Path:
    if args.resume_run_dir is not None:
        run_dir = args.resume_run_dir.resolve()
        config_path = run_dir / "run_config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Resume directory lacks run_config.json: {run_dir}")
        saved = json.loads(config_path.read_text(encoding="utf-8"))
        users = [str(value) for value in saved["users"]]
        period_indices = [int(value) for value in saved["period_indices"]]
        combo_pairs = [
            (int(value[0]), int(value[1])) for value in saved["combo_pairs"]
        ]
        trials = int(saved["trials"])
        phrase_count = int(saved["phrase_count"])
        max_word_attempts = int(saved["max_word_attempts"])
        max_enter_attempts = int(saved["max_enter_attempts"])
        max_clicks_per_word = int(saved["max_clicks_per_word"])
        seed = int(saved["seed"])
        cache_dir = Path(saved["oneclick_cache_dir"])
        baseline_run_dir = Path(saved["baseline_run_dir"])
        combos = build_combo_records(period_indices, combo_pairs)
    else:
        users = parse_csv_values(args.users)
        period_indices = parse_int_values(args.period_indices)
        explicit_pairs = parse_combo_pairs(args.combo_pairs)
        combos = build_combo_records(period_indices, explicit_pairs)
        combo_pairs = [
            (
                int(combo["space_period_index"]),
                int(combo["enter_period_index"]),
            )
            for combo in combos
        ]
        trials = int(args.trials)
        phrase_count = int(args.phrases)
        max_word_attempts = int(args.max_word_attempts)
        max_enter_attempts = int(args.max_enter_attempts)
        max_clicks_per_word = int(args.max_clicks_per_word)
        seed = int(args.seed)
        cache_dir = args.oneclick_cache_dir.resolve()
        baseline_run_dir = (
            args.baseline_run_dir.resolve()
            if args.baseline_run_dir is not None
            else find_latest_baseline_run(args.output_dir.resolve(), users, trials)
        )
        run_dir = build_output_dir(args.output_dir.resolve())

    if not users or trials < 1 or phrase_count < 1:
        raise ValueError("users, trials, and phrases must be non-empty/positive")
    phrase_set = prepare_baseline_phrase_set(
        run_dir,
        baseline_run_dir,
        phrase_count,
    )
    phrase_ids = set(phrase_set["Comparison Phrase ID"].astype(str))
    phrase_sessions = set(
        pd.to_numeric(phrase_set["Session Num"], errors="raise").astype(int)
    )
    simulation_phrase_df = phrase_set[
        ["Session Num", "Phrase Num", "Phrase Text", "Comparison Phrase ID"]
    ].copy()

    run_config = {
        "experiment": "oneclick_space_enter_phase1",
        "users": users,
        "period_indices": period_indices,
        "periods": [period_record(index) for index in period_indices],
        "combo_pairs": [list(pair) for pair in combo_pairs],
        "combination_count": len(combos),
        "trials": trials,
        "phrase_count": phrase_count,
        "phrase_attempts_per_combination": trials * phrase_count,
        "phrase_attempts_per_user": len(combos) * trials * phrase_count,
        "total_phrase_attempts": len(users) * len(combos) * trials * phrase_count,
        "phrase_set_checksum": phrase_set_checksum(phrase_set),
        "phrase_policy": "reused_prediction_reachable_baseline_set",
        "baseline_run_dir": str(baseline_run_dir.resolve()),
        "max_word_attempts": max_word_attempts,
        "max_enter_attempts": max_enter_attempts,
        "max_clicks_per_word": max_clicks_per_word,
        "undo_mode": "protected",
        "enter_period_applies_to_undo": True,
        "offset_transfer": "paired_absolute_seconds",
        "dead_time_mode": "zero_active_dead_time",
        "phrase_time_ceiling_s": None,
        "auc_horizon_s": AUC_HORIZON_S,
        "time_horizons_s": list(TIME_HORIZONS_S),
        "seed": seed,
        "oneclick_cache_dir": str(cache_dir),
        "worker_count": 1,
    }
    atomic_write_json(run_config, run_dir / "run_config.json")

    manifest = build_manifest(run_dir, users, combos, trials, phrase_ids)
    atomic_write_csv(manifest, run_dir / "condition_manifest.csv")
    schedule_rows = []
    for user_id in users:
        for trial in range(trials):
            click_df, schedule_metadata = load_or_copy_schedule(
                run_dir,
                baseline_run_dir,
                user_id,
                trial,
                phrase_sessions,
            )
            schedule_rows.append(schedule_metadata)
            schedule_id = schedule_metadata["schedule_id"]
            for combo in combos:
                path = condition_path(
                    run_dir,
                    user_id,
                    int(combo["space_period_index"]),
                    int(combo["enter_period_index"]),
                    trial,
                )
                if condition_is_complete(
                    path,
                    user_id,
                    combo,
                    trial,
                    phrase_ids,
                ):
                    print(
                        f"Skipping completed condition: user {user_id}, "
                        f"{combo['combo_label']}, trial {trial + 1}/{trials}"
                    )
                    continue
                print(
                    f"Running user {user_id}, {combo['combo_label']}, "
                    f"trial {trial + 1}/{trials}"
                )
                raw_results = run_oneclick(
                    click_df=click_df.copy(),
                    phrase_df=simulation_phrase_df,
                    max_word_attempts=max_word_attempts,
                    max_enter_attempts=max_enter_attempts,
                    max_clicks_per_word=max_clicks_per_word,
                    undo_mode="protected",
                    oneclick_cache_dir=cache_dir,
                    perfect_letter_observations=False,
                    verbose=args.verbose,
                    fixed_space_clock_period_s=float(combo["space_period_s"]),
                    fixed_enter_clock_period_s=float(combo["enter_period_s"]),
                )
                condition = normalize_condition_results(
                    raw_results,
                    user_id,
                    trial,
                    combo,
                    schedule_id,
                )
                validate_phrase_timing(condition)
                if set(condition["Comparison Phrase ID"].astype(str)) != phrase_ids:
                    raise ValueError("condition did not return the common phrase set")
                atomic_write_csv(condition, path)
                manifest = build_manifest(
                    run_dir,
                    users,
                    combos,
                    trials,
                    phrase_ids,
                )
                atomic_write_csv(manifest, run_dir / "condition_manifest.csv")

    atomic_write_csv(pd.DataFrame(schedule_rows), run_dir / "paired_schedule_checksums.csv")
    profile_source = baseline_run_dir / "user_bootstrap_profiles.csv"
    if profile_source.is_file():
        profiles = pd.read_csv(profile_source)
        profiles = profiles[profiles["user_id"].astype(str).isin(users)]
        atomic_write_csv(profiles, run_dir / "user_bootstrap_profiles.csv")

    manifest = build_manifest(run_dir, users, combos, trials, phrase_ids)
    atomic_write_csv(manifest, run_dir / "condition_manifest.csv")
    incomplete = manifest[manifest["status"] != "completed"]
    if not incomplete.empty:
        raise RuntimeError(f"{len(incomplete)} Phase 1 conditions are incomplete")

    condition_frames = [
        pd.read_csv(run_dir / relative_path)
        for relative_path in manifest["condition_file"]
    ]
    phrase_results = pd.concat(condition_frames, ignore_index=True)
    summary = build_combo_summary(phrase_results)
    curves = build_curve_points(phrase_results)
    failures = build_failure_distribution(phrase_results)
    comparison = build_best_comparison(summary)
    validate_final_outputs(
        phrase_results,
        summary,
        curves,
        users,
        combos,
        trials,
        phrase_count,
    )
    diagonal_regression = validate_diagonal_regression(
        run_dir,
        baseline_run_dir,
        users,
        phrase_ids,
    )

    atomic_write_csv(phrase_results, run_dir / "space_enter_phrase_results.csv")
    atomic_write_csv(summary, run_dir / "space_enter_combo_summary.csv")
    atomic_write_csv(curves, run_dir / "space_enter_curve_points.csv")
    atomic_write_csv(failures, run_dir / "space_enter_failure_distribution.csv")
    atomic_write_csv(comparison, run_dir / "best_diagonal_off_diagonal_comparison.csv")
    atomic_write_csv(diagonal_regression, run_dir / "diagonal_baseline_regression.csv")
    plot_outputs = create_plots(
        run_dir,
        summary,
        curves,
        users,
        period_indices,
    )
    print(f"Saved OneClick Space/Enter Phase 1 to: {run_dir}")
    print(comparison.to_string(index=False))
    for output in plot_outputs:
        print(output)
    return run_dir


if __name__ == "__main__":
    run_phase1(parse_args())
