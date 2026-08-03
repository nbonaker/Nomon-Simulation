"""Global two-stage OneClick Space/Enter clock-period experiment.

The screen evaluates a complete Space/Enter grid with one paired trial. The
five strongest cells are frozen and confirmed with four additional trials.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MultipleLocator, PercentFormatter

from User_Simulation.evaluation.evaluation_baseline import REPO_ROOT
from User_Simulation.evaluation.evaluation_nomon_oneclick_comparison import (
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
from User_Simulation.evaluation.evaluation_oneclick_space_enter_phase1 import (
    AUC_HORIZON_S,
    TIME_HORIZONS_S,
    _frame_checksum,
    build_combo_records,
    build_combo_summary,
    build_failure_distribution,
    condition_is_complete,
    condition_path,
    load_or_copy_schedule,
    normalize_condition_results,
    parse_combo_pairs,
    parse_int_values,
    period_record,
    prepare_baseline_phrase_set,
)
from User_Simulation.evaluation.evaluation_oneclick_space_enter_phase2 import (
    build_trial_summary,
)


DEFAULT_USERS = ("A", "B", "C", "D", "F", "G")
DEFAULT_PERIOD_INDICES = (10, 8, 6, 4, 2, 0)
DEFAULT_SCREEN_TRIAL = 0
DEFAULT_CONFIRMATION_TRIALS = 5
DEFAULT_PHRASES = 20
DEFAULT_SHORTLIST_SIZE = 5
DEFAULT_RELIABILITY_FLOOR = 0.80
DEFAULT_SEED = 12345
VALID_PHASES = ("screen", "confirm", "all")

REQUIRED_REUSE_CONFIG = {
    "max_word_attempts": 5,
    "max_enter_attempts": 5,
    "max_clicks_per_word": 30,
    "undo_mode": "protected",
    "dead_time_mode": "zero_active_dead_time",
    "phrase_time_ceiling_s": None,
}


def build_output_dir(base_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    path = base_dir / f"oneclick_global_space_enter_sweep_{timestamp}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def find_latest_run(output_dir: Path, prefix: str, required: Iterable[str]) -> Path:
    for candidate in sorted(output_dir.glob(f"{prefix}_*"), reverse=True):
        if all((candidate / name).is_file() for name in required):
            return candidate.resolve()
    raise FileNotFoundError(
        f"No completed {prefix} run was found under {output_dir}"
    )


def build_grid_combos(
    period_indices: list[int],
    explicit_pairs: list[tuple[int, int]] | None = None,
) -> list[dict[str, Any]]:
    if explicit_pairs is None:
        pairs = [
            (space_index, enter_index)
            for space_index in period_indices
            for enter_index in period_indices
        ]
    else:
        pairs = list(explicit_pairs)
    return build_combo_records(period_indices, pairs)


def _condition_checksum(frame: pd.DataFrame) -> str:
    columns = sorted(frame.columns)
    return _frame_checksum(frame[columns].copy())


def _config_checksum(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_source_run_config(
    source_dir: Path,
    expected_phrase_checksum: str,
) -> dict[str, Any]:
    config_path = source_dir / "run_config.json"
    if not config_path.is_file():
        raise ValueError(f"Reuse source lacks run_config.json: {source_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("phrase_set_checksum") != expected_phrase_checksum:
        raise ValueError(f"Reuse source phrase checksum differs: {source_dir}")
    for key, expected in REQUIRED_REUSE_CONFIG.items():
        if config.get(key) != expected:
            raise ValueError(
                f"Reuse source has incompatible {key}: "
                f"{config.get(key)!r} != {expected!r}"
            )
    return config


def _filter_condition(
    frame: pd.DataFrame,
    phrase_order: list[str],
) -> pd.DataFrame:
    indexed = frame.copy()
    indexed["Comparison Phrase ID"] = indexed["Comparison Phrase ID"].astype(str)
    indexed = indexed.set_index("Comparison Phrase ID", drop=False)
    missing = [phrase_id for phrase_id in phrase_order if phrase_id not in indexed.index]
    if missing:
        raise ValueError(f"Reusable condition lacks phrase IDs: {missing[:3]}")
    filtered = indexed.loc[phrase_order].reset_index(drop=True)
    if len(filtered) != len(phrase_order):
        raise ValueError("Reusable condition contains duplicate phrase IDs")
    return filtered


def _normalize_reused_condition(
    frame: pd.DataFrame,
    user_id: str,
    trial: int,
    combo: dict[str, Any],
) -> pd.DataFrame:
    result = frame.copy()
    result["user_id"] = str(user_id)
    result["trial"] = int(trial)
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
    result["paired_click_schedule_id"] = f"{user_id}_trial_{trial:02d}"
    return result


def reusable_source_candidates(
    phase1_run_dir: Path,
    phase2_run_dir: Path,
    baseline_run_dir: Path,
    user_id: str,
    combo: dict[str, Any],
    trial: int,
) -> list[tuple[str, Path, bool]]:
    space_index = int(combo["space_period_index"])
    enter_index = int(combo["enter_period_index"])
    candidates: list[tuple[str, Path, bool]] = [
        (
            "phase2",
            condition_path(
                phase2_run_dir,
                user_id,
                space_index,
                enter_index,
                trial,
            ),
            False,
        )
    ]
    if trial == 0:
        candidates.append(
            (
                "phase1",
                condition_path(
                    phase1_run_dir,
                    user_id,
                    space_index,
                    enter_index,
                    trial,
                ),
                False,
            )
        )
    if space_index == enter_index:
        candidates.append(
            (
                "clock_speed_baseline",
                baseline_run_dir
                / "conditions"
                / f"user_{user_id}"
                / f"period_{space_index:02d}"
                / f"trial_{trial:02d}.csv",
                True,
            )
        )
    return candidates


def import_reusable_condition(
    run_dir: Path,
    phase1_run_dir: Path,
    phase2_run_dir: Path,
    baseline_run_dir: Path,
    user_id: str,
    combo: dict[str, Any],
    trial: int,
    phrase_order: list[str],
    source_configs: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    destination = condition_path(
        run_dir,
        user_id,
        int(combo["space_period_index"]),
        int(combo["enter_period_index"]),
        trial,
    )
    if condition_is_complete(
        destination,
        user_id,
        combo,
        trial,
        set(phrase_order),
    ):
        saved = pd.read_csv(destination)
        return {
            "user_id": user_id,
            "combo_id": combo["combo_id"],
            "trial": trial,
            "condition_origin": "current_run",
            "source_condition": str(destination.resolve()),
            "destination_condition": str(destination.resolve()),
            "condition_checksum": _condition_checksum(saved),
            "reused": True,
            "reuse_validated": True,
        }

    for origin, source, legacy_diagonal in reusable_source_candidates(
        phase1_run_dir,
        phase2_run_dir,
        baseline_run_dir,
        user_id,
        combo,
        trial,
    ):
        if not source.is_file():
            continue
        source_config = source_configs[origin]
        source_frame = _filter_condition(pd.read_csv(source), phrase_order)
        normalized = _normalize_reused_condition(
            source_frame,
            user_id,
            trial,
            combo,
        )
        if legacy_diagonal:
            source_period = pd.to_numeric(
                source_frame["clock_period_index"],
                errors="raise",
            )
            if not (source_period == int(combo["space_period_index"])).all():
                raise ValueError(f"Legacy diagonal period mismatch: {source}")
        completed = normalized["phrase_completed"].fillna(False).astype(bool)
        typed = normalized["Typed Text"].fillna("").astype(str).str.rstrip()
        target = normalized["Target Phrase"].fillna("").astype(str).str.rstrip()
        if not typed[completed].eq(target[completed]).all():
            raise ValueError(f"Reusable condition has false completion: {source}")
        validate_phrase_timing(normalized)
        destination.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_csv(normalized, destination)
        if not condition_is_complete(
            destination,
            user_id,
            combo,
            trial,
            set(phrase_order),
        ):
            raise ValueError(f"Imported reusable condition is invalid: {source}")
        return {
            "user_id": user_id,
            "combo_id": combo["combo_id"],
            "trial": trial,
            "condition_origin": origin,
            "source_condition": str(source.resolve()),
            "destination_condition": str(destination.resolve()),
            "source_config_checksum": _config_checksum(source_config),
            "condition_checksum": _condition_checksum(normalized),
            "reused": True,
            "reuse_validated": True,
        }
    return None


def build_manifest(
    run_dir: Path,
    users: list[str],
    combos: list[dict[str, Any]],
    trials: list[int],
    phrase_ids: set[str],
    stage: str,
    reuse_audit: pd.DataFrame | None = None,
) -> pd.DataFrame:
    origin_lookup: dict[tuple[str, str, int], str] = {}
    if reuse_audit is not None and not reuse_audit.empty:
        for row in reuse_audit.to_dict("records"):
            origin_lookup[
                (str(row["user_id"]), str(row["combo_id"]), int(row["trial"]))
            ] = str(row["condition_origin"])
    rows = []
    for user_id in users:
        for combo in combos:
            for trial in trials:
                path = condition_path(
                    run_dir,
                    user_id,
                    int(combo["space_period_index"]),
                    int(combo["enter_period_index"]),
                    trial,
                )
                complete = condition_is_complete(
                    path,
                    user_id,
                    combo,
                    trial,
                    phrase_ids,
                )
                rows.append(
                    {
                        "stage": stage,
                        "user_id": user_id,
                        **combo,
                        "trial": trial,
                        "status": "completed" if complete else "pending",
                        "condition_origin": origin_lookup.get(
                            (user_id, combo["combo_id"], trial),
                            "new_simulation" if complete else "pending",
                        ),
                        "condition_file": str(path.relative_to(run_dir)),
                    }
                )
    return pd.DataFrame(rows)


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


def freeze_shortlist(
    run_dir: Path,
    ranked_summary: pd.DataFrame,
    shortlist_size: int,
    screen_config_checksum: str,
) -> pd.DataFrame:
    path = run_dir / "frozen_shortlist.csv"
    expected = ranked_summary.head(shortlist_size).copy()
    expected["shortlist_rank"] = np.arange(1, len(expected) + 1)
    expected["screen_config_checksum"] = screen_config_checksum
    expected["shortlist_frozen"] = True
    if path.is_file():
        saved = pd.read_csv(path)
        expected_ids = expected["combo_id"].astype(str).tolist()
        saved_ids = saved["combo_id"].astype(str).tolist()
        if saved_ids != expected_ids:
            raise ValueError(
                "Frozen shortlist differs from the recomputed screen ranking"
            )
        if not saved["screen_config_checksum"].eq(screen_config_checksum).all():
            raise ValueError("Frozen shortlist belongs to another screen config")
        return saved
    atomic_write_csv(expected, path)
    return expected


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
        f"{float(period_record(index)['period_s']):.1f}"
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
        f"{float(period_record(index)['period_s']):.1f}"
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


def _load_condition_frames(run_dir: Path, manifest: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(
        [
            pd.read_csv(run_dir / relative_path)
            for relative_path in manifest["condition_file"]
        ],
        ignore_index=True,
    )


def _write_stage_outputs(
    run_dir: Path,
    prefix: str,
    phrase_results: pd.DataFrame,
    per_user_summary: pd.DataFrame,
    global_summary: pd.DataFrame,
    global_curves: pd.DataFrame,
) -> None:
    atomic_write_csv(phrase_results, run_dir / f"{prefix}_phrase_results.csv")
    atomic_write_csv(per_user_summary, run_dir / f"{prefix}_per_user_summary.csv")
    atomic_write_csv(global_summary, run_dir / f"{prefix}_global_cell_summary.csv")
    atomic_write_csv(global_curves, run_dir / f"{prefix}_global_curve_points.csv")
    atomic_write_csv(
        build_failure_distribution(phrase_results),
        run_dir / f"{prefix}_failure_distribution.csv",
    )


def _run_conditions(
    run_dir: Path,
    users: list[str],
    combos: list[dict[str, Any]],
    trials: list[int],
    phrase_ids: set[str],
    simulation_phrase_df: pd.DataFrame,
    click_schedules: dict[tuple[str, int], pd.DataFrame],
    cache_dir: Path,
    max_word_attempts: int,
    max_enter_attempts: int,
    max_clicks_per_word: int,
    verbose: bool,
    stage: str,
    reuse_audit: pd.DataFrame,
) -> pd.DataFrame:
    manifest_path = run_dir / f"{stage}_manifest.csv"
    for user_id in users:
        for trial in trials:
            schedule_id = f"{user_id}_trial_{trial:02d}"
            click_df = click_schedules[(user_id, trial)]
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
                        f"Skipping completed {stage} condition: user {user_id}, "
                        f"{combo['combo_label']}, trial {trial + 1}"
                    )
                    continue
                print(
                    f"Running {stage}: user {user_id}, {combo['combo_label']}, "
                    f"trial {trial + 1}"
                )
                raw = run_oneclick(
                    click_df=click_df.copy(),
                    phrase_df=simulation_phrase_df,
                    max_word_attempts=max_word_attempts,
                    max_enter_attempts=max_enter_attempts,
                    max_clicks_per_word=max_clicks_per_word,
                    undo_mode="protected",
                    oneclick_cache_dir=cache_dir,
                    perfect_letter_observations=False,
                    verbose=verbose,
                    fixed_space_clock_period_s=float(combo["space_period_s"]),
                    fixed_enter_clock_period_s=float(combo["enter_period_s"]),
                )
                condition = normalize_condition_results(
                    raw,
                    user_id,
                    trial,
                    combo,
                    schedule_id,
                )
                validate_phrase_timing(condition)
                if set(condition["Comparison Phrase ID"].astype(str)) != phrase_ids:
                    raise ValueError("Condition did not return the common phrase set")
                atomic_write_csv(condition, path)
                manifest = build_manifest(
                    run_dir,
                    users,
                    combos,
                    trials,
                    phrase_ids,
                    stage,
                    reuse_audit,
                )
                atomic_write_csv(manifest, manifest_path)
    manifest = build_manifest(
        run_dir,
        users,
        combos,
        trials,
        phrase_ids,
        stage,
        reuse_audit,
    )
    atomic_write_csv(manifest, manifest_path)
    if not manifest["status"].eq("completed").all():
        raise RuntimeError(f"{stage} contains incomplete conditions")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=VALID_PHASES, default="all")
    parser.add_argument("--users", default=",".join(DEFAULT_USERS))
    parser.add_argument(
        "--period-indices",
        default=",".join(str(value) for value in DEFAULT_PERIOD_INDICES),
    )
    parser.add_argument(
        "--combo-pairs",
        default=None,
        help="Optional SPACE_INDEX:ENTER_INDEX pairs for smoke tests",
    )
    parser.add_argument(
        "--confirmation-trials",
        type=int,
        default=DEFAULT_CONFIRMATION_TRIALS,
    )
    parser.add_argument("--phrases", type=int, default=DEFAULT_PHRASES)
    parser.add_argument("--shortlist-size", type=int, default=DEFAULT_SHORTLIST_SIZE)
    parser.add_argument(
        "--reliability-floor",
        type=float,
        default=DEFAULT_RELIABILITY_FLOOR,
    )
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
    parser.add_argument("--phase1-run-dir", type=Path, default=None)
    parser.add_argument("--phase2-run-dir", type=Path, default=None)
    parser.add_argument("--baseline-run-dir", type=Path, default=None)
    parser.add_argument("--resume-run-dir", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def run_global_sweep(args: argparse.Namespace) -> Path:
    if args.resume_run_dir is not None:
        run_dir = args.resume_run_dir.resolve()
        config_path = run_dir / "run_config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Resume directory lacks config: {run_dir}")
        saved = json.loads(config_path.read_text(encoding="utf-8"))
        users = [str(value) for value in saved["users"]]
        period_indices = [int(value) for value in saved["period_indices"]]
        combo_pairs = [tuple(value) for value in saved["combo_pairs"]]
        combos = build_grid_combos(period_indices, combo_pairs)
        confirmation_trials = int(saved["confirmation_trials"])
        phrase_count = int(saved["phrase_count"])
        shortlist_size = int(saved["shortlist_size"])
        reliability_floor = float(saved["reliability_floor"])
        max_word_attempts = int(saved["max_word_attempts"])
        max_enter_attempts = int(saved["max_enter_attempts"])
        max_clicks_per_word = int(saved["max_clicks_per_word"])
        seed = int(saved["seed"])
        cache_dir = Path(saved["oneclick_cache_dir"])
        phase1_run_dir = Path(saved["phase1_run_dir"])
        phase2_run_dir = Path(saved["phase2_run_dir"])
        baseline_run_dir = Path(saved["baseline_run_dir"])
        configured_phase = str(saved["configured_phase"])
        requested_phase = args.phase
    else:
        users = parse_csv_values(args.users)
        period_indices = parse_int_values(args.period_indices)
        explicit_pairs = parse_combo_pairs(args.combo_pairs)
        combos = build_grid_combos(period_indices, explicit_pairs)
        combo_pairs = [
            (
                int(combo["space_period_index"]),
                int(combo["enter_period_index"]),
            )
            for combo in combos
        ]
        confirmation_trials = int(args.confirmation_trials)
        phrase_count = int(args.phrases)
        shortlist_size = int(args.shortlist_size)
        reliability_floor = float(args.reliability_floor)
        max_word_attempts = int(args.max_word_attempts)
        max_enter_attempts = int(args.max_enter_attempts)
        max_clicks_per_word = int(args.max_clicks_per_word)
        seed = int(args.seed)
        cache_dir = args.oneclick_cache_dir.resolve()
        output_dir = args.output_dir.resolve()
        phase1_run_dir = (
            args.phase1_run_dir.resolve()
            if args.phase1_run_dir is not None
            else find_latest_run(
                output_dir,
                "oneclick_space_enter_phase1",
                ["run_config.json", "space_enter_phrase_results.csv"],
            )
        )
        phase2_run_dir = (
            args.phase2_run_dir.resolve()
            if args.phase2_run_dir is not None
            else find_latest_run(
                output_dir,
                "oneclick_space_enter_phase2",
                ["run_config.json", "phase2_phrase_results.csv"],
            )
        )
        phase1_config = json.loads(
            (phase1_run_dir / "run_config.json").read_text(encoding="utf-8")
        )
        baseline_run_dir = (
            args.baseline_run_dir.resolve()
            if args.baseline_run_dir is not None
            else Path(phase1_config["baseline_run_dir"]).resolve()
        )
        configured_phase = args.phase
        requested_phase = args.phase
        run_dir = build_output_dir(output_dir)

    if not users or not combos or phrase_count < 1:
        raise ValueError("Users, combinations, and phrase count must be positive")
    if confirmation_trials < 1:
        raise ValueError("Confirmation trials must be positive")
    if shortlist_size < 1 or shortlist_size > len(combos):
        raise ValueError("Shortlist size must be within the combination count")
    if not 0.0 <= reliability_floor <= 1.0:
        raise ValueError("Reliability floor must be within [0, 1]")
    if requested_phase == "confirm" and not (
        run_dir / "frozen_shortlist.csv"
    ).is_file():
        raise ValueError("Confirmation requires a completed frozen screen shortlist")

    phrase_set = prepare_baseline_phrase_set(
        run_dir,
        phase1_run_dir,
        phrase_count,
    )
    phrase_order = phrase_set["Comparison Phrase ID"].astype(str).tolist()
    phrase_ids = set(phrase_order)
    phrase_sessions = set(
        pd.to_numeric(phrase_set["Session Num"], errors="raise").astype(int)
    )
    simulation_phrase_df = phrase_set[
        ["Session Num", "Phrase Num", "Phrase Text", "Comparison Phrase ID"]
    ].copy()
    current_phrase_checksum = phrase_set_checksum(phrase_set)

    source_configs = {
        "phase1": validate_source_run_config(
            phase1_run_dir,
            current_phrase_checksum,
        ),
        "phase2": validate_source_run_config(
            phase2_run_dir,
            current_phrase_checksum,
        ),
        "clock_speed_baseline": validate_source_run_config(
            baseline_run_dir,
            current_phrase_checksum,
        ),
    }
    screen_config = {
        "users": users,
        "period_indices": period_indices,
        "combo_pairs": [list(pair) for pair in combo_pairs],
        "phrase_set_checksum": current_phrase_checksum,
        "screen_trial": DEFAULT_SCREEN_TRIAL,
        "reliability_floor": reliability_floor,
        "ranking": [
            "users_meeting_reliability_floor_desc",
            "worst_user_completion_desc",
            "macro_completion_desc",
            "macro_auc_180_desc",
            "macro_completion_by_120s_desc",
            "space_index_asc",
            "enter_index_asc",
        ],
    }
    screen_config_checksum = _config_checksum(screen_config)
    run_config = {
        "experiment": "oneclick_global_space_enter_sweep",
        "configured_phase": configured_phase,
        "users": users,
        "period_indices": period_indices,
        "periods": [period_record(index) for index in period_indices],
        "combo_pairs": [list(pair) for pair in combo_pairs],
        "combination_count": len(combos),
        "screen_trial": DEFAULT_SCREEN_TRIAL,
        "confirmation_trials": confirmation_trials,
        "phrase_count": phrase_count,
        "shortlist_size": shortlist_size,
        "reliability_floor": reliability_floor,
        "phrase_set_checksum": current_phrase_checksum,
        "screen_config_checksum": screen_config_checksum,
        "screen_analyzed_phrase_attempts": (
            len(users) * len(combos) * phrase_count
        ),
        "confirmation_analyzed_phrase_attempts": (
            len(users) * shortlist_size * confirmation_trials * phrase_count
        ),
        "maximum_unique_phrase_attempts": (
            len(users) * len(combos) * phrase_count
            + len(users)
            * shortlist_size
            * max(confirmation_trials - 1, 0)
            * phrase_count
        ),
        "phase1_run_dir": str(phase1_run_dir.resolve()),
        "phase2_run_dir": str(phase2_run_dir.resolve()),
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
        "selection_policy": screen_config["ranking"],
        "seed": seed,
        "oneclick_cache_dir": str(cache_dir),
        "worker_count": 1,
    }
    atomic_write_json(run_config, run_dir / "run_config.json")

    click_schedules: dict[tuple[str, int], pd.DataFrame] = {}
    schedule_rows = []
    for user_id in users:
        for trial in range(confirmation_trials):
            frame, metadata = load_or_copy_schedule(
                run_dir,
                baseline_run_dir,
                user_id,
                trial,
                phrase_sessions,
            )
            click_schedules[(user_id, trial)] = frame
            schedule_rows.append(metadata)
    atomic_write_csv(
        pd.DataFrame(schedule_rows),
        run_dir / "paired_schedule_checksums.csv",
    )

    reuse_rows: list[dict[str, Any]] = []
    reuse_path = run_dir / "reuse_audit.csv"
    if reuse_path.is_file():
        reuse_rows = pd.read_csv(reuse_path).to_dict("records")
    reuse_keys = {
        (str(row["user_id"]), str(row["combo_id"]), int(row["trial"]))
        for row in reuse_rows
    }

    if requested_phase in ("screen", "all") or (
        run_dir / "screen_manifest.csv"
    ).is_file():
        for user_id in users:
            for combo in combos:
                key = (user_id, combo["combo_id"], DEFAULT_SCREEN_TRIAL)
                if key in reuse_keys:
                    continue
                audit = import_reusable_condition(
                    run_dir,
                    phase1_run_dir,
                    phase2_run_dir,
                    baseline_run_dir,
                    user_id,
                    combo,
                    DEFAULT_SCREEN_TRIAL,
                    phrase_order,
                    source_configs,
                )
                if audit is not None:
                    reuse_rows.append(audit)
                    reuse_keys.add(key)
        reuse_audit = pd.DataFrame(reuse_rows)
        atomic_write_csv(reuse_audit, reuse_path)
        screen_manifest = build_manifest(
            run_dir,
            users,
            combos,
            [DEFAULT_SCREEN_TRIAL],
            phrase_ids,
            "screen",
            reuse_audit,
        )
        atomic_write_csv(screen_manifest, run_dir / "screen_manifest.csv")
        if requested_phase in ("screen", "all"):
            screen_manifest = _run_conditions(
                run_dir,
                users,
                combos,
                [DEFAULT_SCREEN_TRIAL],
                phrase_ids,
                simulation_phrase_df,
                click_schedules,
                cache_dir,
                max_word_attempts,
                max_enter_attempts,
                max_clicks_per_word,
                args.verbose,
                "screen",
                reuse_audit,
            )
        if not screen_manifest["status"].eq("completed").all():
            raise RuntimeError("Screen must complete before shortlist selection")
        screen_results = _load_condition_frames(run_dir, screen_manifest)
        screen_per_user, screen_global = build_global_summary(
            screen_results,
            reliability_floor,
        )
        screen_curves = build_global_curve_points(screen_results)
        validate_stage_outputs(
            screen_results,
            screen_per_user,
            screen_global,
            screen_curves,
            screen_manifest,
            users,
            combos,
            [DEFAULT_SCREEN_TRIAL],
            phrase_count,
        )
        shortlist = freeze_shortlist(
            run_dir,
            screen_global,
            shortlist_size,
            screen_config_checksum,
        )
        _write_stage_outputs(
            run_dir,
            "screen",
            screen_results,
            screen_per_user,
            screen_global,
            screen_curves,
        )
        screen_plot_outputs = create_screen_heatmaps(
            run_dir,
            screen_global,
            shortlist,
            period_indices,
        ) + create_failure_reason_heatmap(
            run_dir,
            screen_global,
            period_indices,
        )
    else:
        shortlist = pd.read_csv(run_dir / "frozen_shortlist.csv")
        screen_plot_outputs = []

    confirmation_plot_outputs: list[Path] = []
    if requested_phase in ("confirm", "all"):
        shortlist_pairs = [
            (
                int(row["space_period_index"]),
                int(row["enter_period_index"]),
            )
            for row in shortlist.to_dict("records")
        ]
        confirmation_combos = build_grid_combos(
            period_indices,
            shortlist_pairs,
        )
        confirmation_trial_indices = list(range(confirmation_trials))
        for user_id in users:
            for combo in confirmation_combos:
                for trial in confirmation_trial_indices:
                    key = (user_id, combo["combo_id"], trial)
                    if key in reuse_keys:
                        continue
                    audit = import_reusable_condition(
                        run_dir,
                        phase1_run_dir,
                        phase2_run_dir,
                        baseline_run_dir,
                        user_id,
                        combo,
                        trial,
                        phrase_order,
                        source_configs,
                    )
                    if audit is not None:
                        reuse_rows.append(audit)
                        reuse_keys.add(key)
        reuse_audit = pd.DataFrame(reuse_rows)
        atomic_write_csv(reuse_audit, reuse_path)
        confirmation_manifest = build_manifest(
            run_dir,
            users,
            confirmation_combos,
            confirmation_trial_indices,
            phrase_ids,
            "confirmation",
            reuse_audit,
        )
        atomic_write_csv(
            confirmation_manifest,
            run_dir / "confirmation_manifest.csv",
        )
        confirmation_manifest = _run_conditions(
            run_dir,
            users,
            confirmation_combos,
            confirmation_trial_indices,
            phrase_ids,
            simulation_phrase_df,
            click_schedules,
            cache_dir,
            max_word_attempts,
            max_enter_attempts,
            max_clicks_per_word,
            args.verbose,
            "confirmation",
            reuse_audit,
        )
        confirmation_results = _load_condition_frames(
            run_dir,
            confirmation_manifest,
        )
        confirmed_per_user, confirmed_global = build_global_summary(
            confirmation_results,
            reliability_floor,
        )
        confirmed_global_curves = build_global_curve_points(
            confirmation_results
        )
        per_user_trial, global_trial = build_global_trial_summary(
            confirmation_results
        )
        validate_stage_outputs(
            confirmation_results,
            confirmed_per_user,
            confirmed_global,
            confirmed_global_curves,
            confirmation_manifest,
            users,
            confirmation_combos,
            confirmation_trial_indices,
            phrase_count,
        )
        selected = rank_global_summary(confirmed_global).iloc[[0]].copy()
        selected["universal_reliability_met"] = (
            selected["users_meeting_reliability_floor"] == len(users)
        )
        selected["selection_status"] = np.where(
            selected["universal_reliability_met"],
            "universal reliability threshold met",
            "no tested global cell met universal reliability; best compromise selected",
        )
        paired = build_paired_comparisons(
            confirmation_results,
            confirmed_global,
            str(selected.iloc[0]["combo_id"]),
        )
        _write_stage_outputs(
            run_dir,
            "confirmed",
            confirmation_results,
            confirmed_per_user,
            confirmed_global,
            confirmed_global_curves,
        )
        atomic_write_csv(
            per_user_trial,
            run_dir / "confirmed_per_user_trial_summary.csv",
        )
        atomic_write_csv(
            global_trial,
            run_dir / "confirmed_global_trial_summary.csv",
        )
        atomic_write_csv(selected, run_dir / "global_selection_summary.csv")
        atomic_write_csv(paired, run_dir / "confirmed_paired_comparisons.csv")
        confirmation_plot_outputs = create_confirmation_plots(
            run_dir,
            confirmation_results,
            confirmed_per_user,
            confirmed_global,
            confirmed_global_curves,
            global_trial,
            reliability_floor,
        )

    print(f"Saved global OneClick Space/Enter sweep to: {run_dir}")
    for output in screen_plot_outputs + confirmation_plot_outputs:
        print(output)
    return run_dir


if __name__ == "__main__":
    run_global_sweep(parse_args())
