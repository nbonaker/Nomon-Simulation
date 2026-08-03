"""Targeted Phase 2 confirmation of OneClick Space/Enter clock periods.

The default design reuses trial 0 from the completed Phase 1 screen and runs
four additional paired trials for four candidate conditions per user.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    DEFAULT_PERIOD_INDICES,
    TIME_HORIZONS_S,
    _frame_checksum,
    _plot_curve_subset,
    build_combo_records,
    build_combo_summary,
    build_curve_points,
    build_failure_distribution,
    condition_is_complete,
    condition_path,
    load_or_copy_schedule,
    normalize_condition_results,
    period_record,
    prepare_baseline_phrase_set,
)


DEFAULT_USERS = ("A", "C")
DEFAULT_CANDIDATE_PAIRS = {
    "A": ((10, 10), (0, 0), (14, 10), (10, 0)),
    "C": ((0, 0), (2, 2), (4, 4), (10, 0)),
}
DEFAULT_TRIALS = 5
DEFAULT_PHRASES = 20
DEFAULT_RELIABILITY_FLOOR = 0.80
DEFAULT_SEED = 12345


def build_output_dir(base_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    output_dir = base_dir / f"oneclick_space_enter_phase2_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def find_latest_phase1_run(output_dir: Path, users: list[str]) -> Path:
    for candidate in sorted(
        output_dir.glob("oneclick_space_enter_phase1_*"),
        reverse=True,
    ):
        if not all(
            (candidate / "conditions" / f"user_{user_id}").is_dir()
            for user_id in users
        ):
            continue
        required = [
            "run_config.json",
            "common_phrase_set.csv",
            "space_enter_phrase_results.csv",
        ]
        if all((candidate / name).is_file() for name in required):
            return candidate.resolve()
    raise FileNotFoundError(
        "No completed Phase 1 run was found; pass --phase1-run-dir explicitly"
    )


def parse_candidate_pairs(value: str | None) -> list[tuple[int, int]] | None:
    if value is None:
        return None
    pairs: list[tuple[int, int]] = []
    for item in value.split(","):
        pieces = item.strip().split(":")
        if len(pieces) != 2:
            raise ValueError(
                "candidate pairs must use SPACE_INDEX:ENTER_INDEX syntax"
            )
        pairs.append((int(pieces[0]), int(pieces[1])))
    if not pairs or len(pairs) != len(set(pairs)):
        raise ValueError("candidate pairs must be non-empty and unique")
    return pairs


def build_candidate_map(
    users: list[str],
    shared_pairs: list[tuple[int, int]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for user_id in users:
        if shared_pairs is None:
            if user_id not in DEFAULT_CANDIDATE_PAIRS:
                raise ValueError(f"No default Phase 2 candidates for user {user_id}")
            pairs = list(DEFAULT_CANDIDATE_PAIRS[user_id])
        else:
            pairs = list(shared_pairs)
        result[user_id] = build_combo_records(
            list(DEFAULT_PERIOD_INDICES),
            pairs,
        )
    return result


def build_manifest(
    run_dir: Path,
    candidate_map: dict[str, list[dict[str, Any]]],
    trials: int,
    phrase_ids: set[str],
) -> pd.DataFrame:
    rows = []
    for user_id, combos in candidate_map.items():
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
                        "condition_origin": (
                            "phase1_reused" if trial == 0 else "phase2_new"
                        ),
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


def reuse_phase1_trial_zero(
    run_dir: Path,
    phase1_run_dir: Path,
    candidate_map: dict[str, list[dict[str, Any]]],
    phrase_ids: set[str],
) -> pd.DataFrame:
    rows = []
    for user_id, combos in candidate_map.items():
        for combo in combos:
            source = condition_path(
                phase1_run_dir,
                user_id,
                int(combo["space_period_index"]),
                int(combo["enter_period_index"]),
                0,
            )
            destination = condition_path(
                run_dir,
                user_id,
                int(combo["space_period_index"]),
                int(combo["enter_period_index"]),
                0,
            )
            if not source.is_file():
                raise ValueError(f"Phase 1 trial-0 condition is unavailable: {source}")
            source_frame = pd.read_csv(source)
            expected = source_frame[
                source_frame["Comparison Phrase ID"].astype(str).isin(phrase_ids)
            ].copy()
            completed = expected["phrase_completed"].fillna(False).astype(bool)
            typed = expected["Typed Text"].fillna("").astype(str).str.rstrip()
            target = expected["Target Phrase"].fillna("").astype(str).str.rstrip()
            valid_subset = bool(
                len(expected) == len(phrase_ids)
                and set(expected["Comparison Phrase ID"].astype(str)) == phrase_ids
                and typed[completed].eq(target[completed]).all()
                and (expected["user_id"].astype(str) == str(user_id)).all()
                and (pd.to_numeric(expected["trial"]) == 0).all()
                and (
                    pd.to_numeric(expected["space_period_index"])
                    == int(combo["space_period_index"])
                ).all()
                and (
                    pd.to_numeric(expected["enter_period_index"])
                    == int(combo["enter_period_index"])
                ).all()
            )
            if not valid_subset:
                raise ValueError(
                    f"Phase 1 trial-0 condition lacks the requested subset: {source}"
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.is_file():
                destination_frame = pd.read_csv(destination)
                if _frame_checksum(expected) != _frame_checksum(destination_frame):
                    raise ValueError(
                        f"Saved Phase 2 trial 0 differs from Phase 1: {destination}"
                    )
            else:
                atomic_write_csv(expected, destination)
            if not condition_is_complete(
                destination,
                user_id,
                combo,
                0,
                phrase_ids,
            ):
                raise ValueError(
                    f"Reused Phase 2 trial-0 checkpoint is invalid: {destination}"
                )
            rows.append(
                {
                    "user_id": user_id,
                    "combo_id": combo["combo_id"],
                    "trial": 0,
                    "phrase_count": len(phrase_ids),
                    "source_condition": str(source.resolve()),
                    "destination_condition": str(destination.resolve()),
                    "source_subset_checksum": _frame_checksum(expected),
                    "reuse_validated": True,
                }
            )
    return pd.DataFrame(rows)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--users", default=",".join(DEFAULT_USERS))
    parser.add_argument(
        "--candidate-pairs",
        default=None,
        help="Optional shared SPACE_INDEX:ENTER_INDEX pairs for smoke tests",
    )
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--phrases", type=int, default=DEFAULT_PHRASES)
    parser.add_argument("--reliability-floor", type=float, default=DEFAULT_RELIABILITY_FLOOR)
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
    parser.add_argument("--resume-run-dir", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def run_phase2(args: argparse.Namespace) -> Path:
    if args.resume_run_dir is not None:
        run_dir = args.resume_run_dir.resolve()
        config_path = run_dir / "run_config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Resume directory lacks run_config.json: {run_dir}")
        saved = json.loads(config_path.read_text(encoding="utf-8"))
        users = [str(value) for value in saved["users"]]
        candidate_map = {
            user_id: build_combo_records(
                list(DEFAULT_PERIOD_INDICES),
                [tuple(pair) for pair in saved["candidate_pairs_by_user"][user_id]],
            )
            for user_id in users
        }
        trials = int(saved["trials"])
        phrase_count = int(saved["phrase_count"])
        reliability_floor = float(saved["reliability_floor"])
        max_word_attempts = int(saved["max_word_attempts"])
        max_enter_attempts = int(saved["max_enter_attempts"])
        max_clicks_per_word = int(saved["max_clicks_per_word"])
        seed = int(saved["seed"])
        cache_dir = Path(saved["oneclick_cache_dir"])
        phase1_run_dir = Path(saved["phase1_run_dir"])
        baseline_run_dir = Path(saved["baseline_run_dir"])
    else:
        users = parse_csv_values(args.users)
        candidate_map = build_candidate_map(
            users,
            parse_candidate_pairs(args.candidate_pairs),
        )
        trials = int(args.trials)
        phrase_count = int(args.phrases)
        reliability_floor = float(args.reliability_floor)
        max_word_attempts = int(args.max_word_attempts)
        max_enter_attempts = int(args.max_enter_attempts)
        max_clicks_per_word = int(args.max_clicks_per_word)
        seed = int(args.seed)
        cache_dir = args.oneclick_cache_dir.resolve()
        phase1_run_dir = (
            args.phase1_run_dir.resolve()
            if args.phase1_run_dir is not None
            else find_latest_phase1_run(args.output_dir.resolve(), users)
        )
        phase1_config = json.loads(
            (phase1_run_dir / "run_config.json").read_text(encoding="utf-8")
        )
        baseline_run_dir = Path(phase1_config["baseline_run_dir"]).resolve()
        run_dir = build_output_dir(args.output_dir.resolve())

    if not users or trials < 1 or phrase_count < 1:
        raise ValueError("users, trials, and phrases must be non-empty/positive")
    if not 0.0 <= reliability_floor <= 1.0:
        raise ValueError("reliability floor must be within [0, 1]")
    phrase_set = prepare_baseline_phrase_set(
        run_dir,
        phase1_run_dir,
        phrase_count,
    )
    phrase_ids = set(phrase_set["Comparison Phrase ID"].astype(str))
    phrase_sessions = set(
        pd.to_numeric(phrase_set["Session Num"], errors="raise").astype(int)
    )
    simulation_phrase_df = phrase_set[
        ["Session Num", "Phrase Num", "Phrase Text", "Comparison Phrase ID"]
    ].copy()

    candidate_pairs_by_user = {
        user_id: [
            [
                int(combo["space_period_index"]),
                int(combo["enter_period_index"]),
            ]
            for combo in combos
        ]
        for user_id, combos in candidate_map.items()
    }
    run_config = {
        "experiment": "oneclick_space_enter_phase2",
        "users": users,
        "candidate_pairs_by_user": candidate_pairs_by_user,
        "candidate_count_by_user": {
            user_id: len(combos) for user_id, combos in candidate_map.items()
        },
        "trials": trials,
        "reused_phase1_trials": [0],
        "new_trials": list(range(1, trials)),
        "phrase_count": phrase_count,
        "reused_phrase_attempts": sum(len(value) for value in candidate_map.values())
        * phrase_count,
        "new_phrase_attempts": sum(len(value) for value in candidate_map.values())
        * max(trials - 1, 0)
        * phrase_count,
        "total_analyzed_phrase_attempts": sum(
            len(value) for value in candidate_map.values()
        )
        * trials
        * phrase_count,
        "phrase_set_checksum": phrase_set_checksum(phrase_set),
        "phase1_run_dir": str(phase1_run_dir.resolve()),
        "baseline_run_dir": str(baseline_run_dir.resolve()),
        "reliability_floor": reliability_floor,
        "selection_policy": (
            "require pooled completion >= reliability floor, then maximize "
            "normalized completion-by-time AUC through 180 seconds"
        ),
        "max_word_attempts": max_word_attempts,
        "max_enter_attempts": max_enter_attempts,
        "max_clicks_per_word": max_clicks_per_word,
        "undo_mode": "protected",
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

    reuse_audit = reuse_phase1_trial_zero(
        run_dir,
        phase1_run_dir,
        candidate_map,
        phrase_ids,
    )
    atomic_write_csv(reuse_audit, run_dir / "phase1_trial0_reuse_audit.csv")

    manifest = build_manifest(
        run_dir,
        candidate_map,
        trials,
        phrase_ids,
    )
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
            for combo in candidate_map[user_id]:
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
                    candidate_map,
                    trials,
                    phrase_ids,
                )
                atomic_write_csv(manifest, run_dir / "condition_manifest.csv")

    atomic_write_csv(
        pd.DataFrame(schedule_rows),
        run_dir / "paired_schedule_checksums.csv",
    )
    profile_source = phase1_run_dir / "user_bootstrap_profiles.csv"
    if profile_source.is_file():
        profiles = pd.read_csv(profile_source)
        profiles = profiles[profiles["user_id"].astype(str).isin(users)]
        atomic_write_csv(profiles, run_dir / "user_bootstrap_profiles.csv")

    manifest = build_manifest(
        run_dir,
        candidate_map,
        trials,
        phrase_ids,
    )
    atomic_write_csv(manifest, run_dir / "condition_manifest.csv")
    if not (manifest["status"] == "completed").all():
        raise RuntimeError("Phase 2 contains incomplete conditions")
    phrase_results = pd.concat(
        [
            pd.read_csv(run_dir / relative_path)
            for relative_path in manifest["condition_file"]
        ],
        ignore_index=True,
    )
    summary, trial_summary = enrich_candidate_summary(
        phrase_results,
        reliability_floor,
    )
    curves = build_curve_points(phrase_results)
    failures = build_failure_distribution(phrase_results)
    selections = build_selection_summary(summary, reliability_floor)
    paired = build_paired_comparisons(phrase_results, summary, selections)
    validate_final_outputs(
        phrase_results,
        summary,
        trial_summary,
        curves,
        manifest,
        candidate_map,
        trials,
        phrase_count,
    )

    atomic_write_csv(phrase_results, run_dir / "phase2_phrase_results.csv")
    atomic_write_csv(trial_summary, run_dir / "phase2_trial_summary.csv")
    atomic_write_csv(summary, run_dir / "phase2_candidate_summary.csv")
    atomic_write_csv(curves, run_dir / "phase2_curve_points.csv")
    atomic_write_csv(failures, run_dir / "phase2_failure_distribution.csv")
    atomic_write_csv(selections, run_dir / "phase2_selection_summary.csv")
    atomic_write_csv(paired, run_dir / "phase2_paired_comparisons.csv")
    plot_outputs = create_plots(
        run_dir,
        phrase_results,
        summary,
        trial_summary,
        curves,
        selections,
    )
    print(f"Saved OneClick Space/Enter Phase 2 to: {run_dir}")
    print(selections.to_string(index=False))
    for output in plot_outputs:
        print(output)
    return run_dir


if __name__ == "__main__":
    run_phase2(parse_args())
