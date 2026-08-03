"""Held-out OG Nomon versus selected global OneClick comparison.

The comparison uses regime-aware historical timing for OG Nomon and a fixed
6.0-second Space / 6.0-second Enter-and-Undo period for protected OneClick.
Phrase sets, click schedules, and condition results are frozen and resumable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter

from User_Simulation.evaluation.evaluation_baseline import (
    DEFAULT_LM_CACHE_DIR,
    load_text_click_data,
    load_text_phrase_data,
    normalize_phrase_order,
)
from User_Simulation.evaluation.evaluation_nomon_oneclick_bootstrap_comparison import (
    bootstrap_profile_row,
    build_shared_bootstrap_click_df,
    clean_click_rows,
    effective_bootstrap_source_sessions,
    run_old_nomon_bootstrap,
)
from User_Simulation.evaluation.evaluation_nomon_oneclick_comparison import (
    DEFAULT_USERS,
    normalize_system_results,
    parse_csv_values,
    run_oneclick,
)
from User_Simulation.evaluation.evaluation_oneclick_clock_speed_tradeoff import (
    atomic_write_csv,
    atomic_write_json,
    phrase_set_checksum,
    select_common_prediction_reachable_phrases,
)
from User_Simulation.evaluation.synthetic_profiles import (
    CLICK_OFFSET_COL,
    DEAD_TIME_COL,
    estimate_profile,
    select_sessions,
    split_sessions,
)


DEFAULT_TRIALS = 5
DEFAULT_PHRASES = 20
DEFAULT_SEED = 54321
DEFAULT_SPACE_PERIOD_S = 6.0
DEFAULT_ENTER_PERIOD_S = 6.0
TIME_HORIZONS_S = (60.0, 120.0, 180.0)
SYSTEM_ORDER = ("og_nomon", "oneclick")
SYSTEM_LABELS = {"og_nomon": "OG Nomon", "oneclick": "OneClick"}
SYSTEM_COLORS = {"og_nomon": "#4C78A8", "oneclick": "#E45756"}
CONDITION_ATTEMPTS = 6
CONDITION_RETRY_DELAY_S = 20.0
REQUIRED_CONDITION_COLUMNS = {
    "user_id",
    "trial",
    "system",
    "Comparison Phrase ID",
    "Target Phrase",
    "Typed Text",
    "phrase_completed",
    "simulated_attempt_time_s",
    "simulated_completion_time_s",
    "paired_click_schedule_id",
    "paired_offset_checksum",
}


def build_output_dir(base_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    path = base_dir / f"nomon_oneclick_selected_global_comparison_{timestamp}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def find_latest_selection_run(output_dir: Path) -> Path:
    required = {
        "common_phrase_set.csv",
        "common_phrase_reachability_audit.csv",
        "common_phrase_word_reachability_audit.csv",
        "global_selection_summary.csv",
    }
    for candidate in sorted(
        output_dir.glob("oneclick_global_space_enter_sweep_*"),
        reverse=True,
    ):
        if all((candidate / filename).is_file() for filename in required):
            return candidate.resolve()
    raise FileNotFoundError(
        "No completed global Space/Enter sweep was found; pass "
        "--selection-run-dir explicitly"
    )


def validate_selected_periods(selection_run_dir: Path) -> None:
    selected = pd.read_csv(selection_run_dir / "global_selection_summary.csv")
    if len(selected) != 1:
        raise ValueError("global selection summary must contain exactly one selected cell")
    row = selected.iloc[0]
    if not math.isclose(
        float(row["space_period_s"]),
        DEFAULT_SPACE_PERIOD_S,
        rel_tol=0.0,
        abs_tol=1e-9,
    ) or not math.isclose(
        float(row["enter_period_s"]),
        DEFAULT_ENTER_PERIOD_S,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "the supplied global selection run did not select the configured "
            "6.0-second Space / 6.0-second Enter cell"
        )


def frame_checksum(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(index=False, float_format="%.17g")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def condition_path(
    run_dir: Path,
    user_id: str,
    system: str,
    trial: int,
) -> Path:
    return (
        run_dir
        / "conditions"
        / f"user_{user_id}"
        / system
        / f"trial_{trial:02d}.csv"
    )


def schedule_path(run_dir: Path, user_id: str, trial: int) -> Path:
    return run_dir / "paired_click_schedules" / f"user_{user_id}_trial_{trial:02d}.csv"


def select_heldout_phrases(
    reachability_audit: pd.DataFrame,
    tuning_phrase_set: pd.DataFrame,
    phrase_count: int,
    seed: int,
) -> pd.DataFrame:
    used_ids = set(tuning_phrase_set["phrase_id"].astype(str))
    candidates = reachability_audit[
        ~reachability_audit["phrase_id"].astype(str).isin(used_ids)
    ].copy()
    selected = select_common_prediction_reachable_phrases(
        candidates,
        phrase_count=phrase_count,
        seed=seed,
    )
    selected["Comparison Phrase ID"] = [
        f"heldout_phrase_{index:02d}" for index in range(1, len(selected) + 1)
    ]
    if set(selected["phrase_id"].astype(str)) & used_ids:
        raise ValueError("held-out phrase selection overlaps the tuning phrase set")
    if not selected["all_words_prediction_reachable"].fillna(False).astype(bool).all():
        raise ValueError("held-out phrase set contains a prediction-unreachable phrase")
    return selected


def prepare_heldout_phrase_set(
    run_dir: Path,
    selection_run_dir: Path,
    phrase_count: int,
    seed: int,
) -> pd.DataFrame:
    destination = run_dir / "heldout_phrase_set.csv"
    audit_destination = run_dir / "heldout_phrase_reachability_audit.csv"
    exclusion_destination = run_dir / "tuning_phrase_exclusions.csv"
    word_audit_destination = run_dir / "heldout_phrase_word_reachability_audit.csv"

    tuning = pd.read_csv(selection_run_dir / "common_phrase_set.csv")
    audit = pd.read_csv(selection_run_dir / "common_phrase_reachability_audit.csv")
    word_audit = pd.read_csv(
        selection_run_dir / "common_phrase_word_reachability_audit.csv"
    )
    expected = select_heldout_phrases(audit, tuning, phrase_count, seed)

    if destination.is_file():
        saved = pd.read_csv(destination)
        if frame_checksum(saved) != frame_checksum(expected):
            raise ValueError("saved held-out phrase set differs from deterministic selection")
        phrase_set = saved
    else:
        phrase_set = expected
        atomic_write_csv(phrase_set, destination)

    selected_ids = set(phrase_set["phrase_id"].astype(str))
    tuning_ids = set(tuning["phrase_id"].astype(str))
    heldout_audit = audit.copy()
    heldout_audit["excluded_as_tuning_phrase"] = heldout_audit[
        "phrase_id"
    ].astype(str).isin(tuning_ids)
    heldout_audit["eligible_heldout_candidate"] = (
        ~heldout_audit["excluded_as_tuning_phrase"]
        & heldout_audit["all_words_prediction_reachable"].fillna(False).astype(bool)
    )
    heldout_audit["selected_for_comparison"] = heldout_audit[
        "phrase_id"
    ].astype(str).isin(selected_ids)
    heldout_word_audit = word_audit[
        word_audit["phrase_id"].astype(str).isin(selected_ids)
    ].copy()
    exclusions = tuning.copy()
    if audit_destination.is_file():
        if frame_checksum(pd.read_csv(audit_destination)) != frame_checksum(
            heldout_audit
        ):
            raise ValueError("saved held-out reachability audit differs")
    else:
        atomic_write_csv(heldout_audit, audit_destination)
    if word_audit_destination.is_file():
        if frame_checksum(pd.read_csv(word_audit_destination)) != frame_checksum(
            heldout_word_audit
        ):
            raise ValueError("saved held-out word audit differs")
    else:
        atomic_write_csv(heldout_word_audit, word_audit_destination)
    if exclusion_destination.is_file():
        if frame_checksum(pd.read_csv(exclusion_destination)) != frame_checksum(
            exclusions
        ):
            raise ValueError("saved tuning exclusions differ")
    else:
        atomic_write_csv(exclusions, exclusion_destination)

    if len(phrase_set) != phrase_count:
        raise ValueError("held-out phrase count does not match configuration")
    if phrase_set["Comparison Phrase ID"].astype(str).duplicated().any():
        raise ValueError("held-out comparison phrase identifiers must be unique")
    if selected_ids & set(tuning["phrase_id"].astype(str)):
        raise ValueError("held-out phrases overlap the tuning phrases")
    return phrase_set


def schedule_seed(base_seed: int, user_id: str, trial: int) -> int:
    return int(base_seed + 100_000 + trial + sum(user_id.encode("utf-8")) * 1000)


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
) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = schedule_path(run_dir, user_id, trial)
    expected = build_shared_bootstrap_click_df(
        profile=profile,
        phrase_df=phrase_df,
        train_click_df=train_click_df,
        trial=trial,
        seed=seed,
        clicks_per_phrase=clicks_per_phrase,
        calibration_clicks=calibration_clicks,
    )
    active = expected["Session Num"].notna()
    expected.loc[active, DEAD_TIME_COL] = 0.0
    if path.is_file():
        saved = pd.read_csv(path)
        if frame_checksum(saved) != frame_checksum(expected):
            raise ValueError(
                f"saved paired schedule differs for user {user_id}, trial {trial}"
            )
        schedule = saved
    else:
        atomic_write_csv(expected, path)
        schedule = expected

    offsets = pd.to_numeric(schedule[CLICK_OFFSET_COL], errors="raise")
    if not np.isfinite(offsets).all():
        raise ValueError("paired schedule contains invalid click offsets")
    active = schedule["Session Num"].notna()
    if (pd.to_numeric(schedule.loc[active, DEAD_TIME_COL], errors="raise") != 0).any():
        raise ValueError("active dead time must be zero")
    offset_checksum = frame_checksum(schedule[["Session Num", CLICK_OFFSET_COL]].copy())
    metadata = {
        "user_id": user_id,
        "trial": int(trial),
        "schedule_id": f"{user_id}_trial_{trial:02d}",
        "schedule_file": str(path.relative_to(run_dir)),
        "schedule_checksum": frame_checksum(schedule),
        "offset_checksum": offset_checksum,
        "row_count": int(len(schedule)),
        "active_row_count": int(active.sum()),
        "active_dead_time_s": 0.0,
    }
    return schedule, metadata


def validate_condition(
    frame: pd.DataFrame,
    user_id: str,
    system: str,
    trial: int,
    phrase_ids: set[str],
) -> None:
    if not REQUIRED_CONDITION_COLUMNS.issubset(frame.columns):
        missing = sorted(REQUIRED_CONDITION_COLUMNS - set(frame.columns))
        raise ValueError(f"condition lacks required columns: {missing}")
    if len(frame) != len(phrase_ids):
        raise ValueError("condition phrase count is incorrect")
    if set(frame["Comparison Phrase ID"].astype(str)) != phrase_ids:
        raise ValueError("condition phrase identifiers are incorrect")
    if not (frame["user_id"].astype(str) == str(user_id)).all():
        raise ValueError("condition user identifier is incorrect")
    if not (frame["system"].astype(str) == system).all():
        raise ValueError("condition system identifier is incorrect")
    if not (pd.to_numeric(frame["trial"], errors="raise") == trial).all():
        raise ValueError("condition trial identifier is incorrect")

    attempts = pd.to_numeric(frame["simulated_attempt_time_s"], errors="coerce")
    completions = pd.to_numeric(
        frame["simulated_completion_time_s"],
        errors="coerce",
    )
    if attempts.isna().any() or (~np.isfinite(attempts)).any() or (attempts < 0).any():
        raise ValueError("condition contains invalid simulated attempt times")
    completed = frame["phrase_completed"].fillna(False).astype(bool)
    if completions[completed].isna().any() or completions[~completed].notna().any():
        raise ValueError("completion time must exist exactly for completed phrases")
    if (completions[completed] < 0).any() or (
        completions[completed] - attempts[completed] > 1e-8
    ).any():
        raise ValueError("condition contains invalid completion times")
    typed = frame["Typed Text"].fillna("").astype(str).str.rstrip()
    target = frame["Target Phrase"].fillna("").astype(str).str.rstrip()
    if not typed[completed].eq(target[completed]).all():
        raise ValueError("completed phrase text does not exactly match the target")
    if typed[~completed].eq(target[~completed]).any():
        raise ValueError("an exact phrase match was not marked completed")

    if system == "oneclick":
        space = pd.to_numeric(frame["space_clock_period_s"], errors="raise")
        enter = pd.to_numeric(frame["enter_clock_period_s"], errors="raise")
        if not np.allclose(space, DEFAULT_SPACE_PERIOD_S):
            raise ValueError("OneClick Space period is not fixed at 6.0 seconds")
        if not np.allclose(enter, DEFAULT_ENTER_PERIOD_S):
            raise ValueError("OneClick Enter period is not fixed at 6.0 seconds")
        stage_total = (
            pd.to_numeric(frame["letter_clock_time_s"], errors="coerce")
            + pd.to_numeric(frame["target_enter_clock_time_s"], errors="coerce")
            + pd.to_numeric(frame["undo_clock_time_s"], errors="coerce")
        )
        if not np.allclose(attempts, stage_total, rtol=0.0, atol=1e-8):
            raise ValueError("OneClick stage time totals do not equal attempt time")


def condition_is_complete(
    path: Path,
    user_id: str,
    system: str,
    trial: int,
    phrase_ids: set[str],
) -> bool:
    if not path.is_file():
        return False
    try:
        validate_condition(
            pd.read_csv(path),
            user_id=user_id,
            system=system,
            trial=trial,
            phrase_ids=phrase_ids,
        )
    except Exception:
        return False
    return True


def build_manifest(
    run_dir: Path,
    users: list[str],
    trials: int,
    phrase_ids: set[str],
) -> pd.DataFrame:
    rows = []
    for user_id in users:
        for system in SYSTEM_ORDER:
            for trial in range(trials):
                path = condition_path(run_dir, user_id, system, trial)
                rows.append(
                    {
                        "user_id": user_id,
                        "system": system,
                        "system_label": SYSTEM_LABELS[system],
                        "trial": trial,
                        "status": (
                            "completed"
                            if condition_is_complete(
                                path,
                                user_id,
                                system,
                                trial,
                                phrase_ids,
                            )
                            else "pending"
                        ),
                        "condition_file": str(path.relative_to(run_dir)),
                    }
                )
    return pd.DataFrame(rows)


def normalize_condition_results(
    raw_results: pd.DataFrame,
    user_id: str,
    system: str,
    trial: int,
    schedule_metadata: dict[str, Any],
) -> pd.DataFrame:
    result = normalize_system_results(raw_results, system, user_id, trial)
    result["system_label"] = SYSTEM_LABELS[system]
    result["paired_click_schedule_id"] = schedule_metadata["schedule_id"]
    result["paired_offset_checksum"] = schedule_metadata["offset_checksum"]
    result["space_clock_period_s"] = (
        DEFAULT_SPACE_PERIOD_S if system == "oneclick" else np.nan
    )
    result["enter_clock_period_s"] = (
        DEFAULT_ENTER_PERIOD_S if system == "oneclick" else np.nan
    )
    result["clock_period_policy"] = (
        "fixed_global_6.0s_space_6.0s_enter"
        if system == "oneclick"
        else "historical_regime_aware_selection_bootstrap"
    )
    completed = result["phrase_completed"].fillna(False).astype(bool)
    result.loc[completed, "phrase_failure_reason"] = ""
    missing_failure = (~completed) & result["phrase_failure_reason"].fillna("").eq("")
    result.loc[
        missing_failure,
        "phrase_failure_reason",
    ] = "click_stream_exhausted_or_incomplete"
    result.loc[~completed, "simulated_completion_time_s"] = np.nan
    return result


def completion_times(group: pd.DataFrame) -> np.ndarray:
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


def summarize_system_results(phrase_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_columns = ["user_id", "system", "system_label"]
    for key, group in phrase_results.groupby(group_columns, sort=False):
        user_id, system, system_label = key
        completed = group["phrase_completed"].fillna(False).astype(bool)
        times = completion_times(group)
        attempts = len(group)
        completed_group = group.loc[completed]
        row = {
            "user_id": user_id,
            "system": system,
            "system_label": system_label,
            "phrase_attempts": int(attempts),
            "completed_phrases": int(completed.sum()),
            "phrase_completion_rate": float(completed.mean()),
            "mean_correction_rate_percent": float(
                pd.to_numeric(group["correction_rate_percent"], errors="coerce").mean()
            ),
            "mean_attempt_clicks": float(
                pd.to_numeric(group["num_clicks"], errors="coerce").mean()
            ),
            "mean_attempt_time_s": float(
                pd.to_numeric(group["simulated_attempt_time_s"], errors="coerce").mean()
            ),
            "successful_mean_clicks": float(
                pd.to_numeric(completed_group["num_clicks"], errors="coerce").mean()
            ),
            "successful_mean_completion_time_s": float(
                pd.to_numeric(
                    completed_group["simulated_completion_time_s"],
                    errors="coerce",
                ).mean()
            ),
        }
        for horizon in TIME_HORIZONS_S:
            row[f"completion_by_{int(horizon)}s"] = float(
                (times <= horizon).sum() / attempts
            )
        rows.append(row)
    return pd.DataFrame(rows)


def build_macro_summary(per_user_summary: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "phrase_completion_rate",
        "completion_by_60s",
        "completion_by_120s",
        "completion_by_180s",
        "mean_correction_rate_percent",
        "mean_attempt_clicks",
        "mean_attempt_time_s",
        "successful_mean_clicks",
        "successful_mean_completion_time_s",
    ]
    rows = []
    for (system, system_label), group in per_user_summary.groupby(
        ["system", "system_label"],
        sort=False,
    ):
        row = {
            "system": system,
            "system_label": system_label,
            "users": int(group["user_id"].nunique()),
            "phrase_attempts": int(group["phrase_attempts"].sum()),
            "completed_phrases": int(group["completed_phrases"].sum()),
        }
        for column in metric_columns:
            row[f"macro_{column}"] = float(
                pd.to_numeric(group[column], errors="coerce").mean()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def build_curve_points(phrase_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scopes: list[tuple[str, pd.DataFrame]] = [("ALL", phrase_results)]
    scopes.extend(
        (str(user_id), group.copy())
        for user_id, group in phrase_results.groupby("user_id", sort=False)
    )
    for scope_user_id, scope in scopes:
        all_successes = completion_times(scope)
        endpoint = (
            max(180.0, math.ceil(float(all_successes.max()) / 30.0) * 30.0)
            if len(all_successes)
            else 180.0
        )
        for (system, system_label), group in scope.groupby(
            ["system", "system_label"],
            sort=False,
        ):
            attempts = len(group)
            times = completion_times(group)
            rows.append(
                {
                    "scope_user_id": scope_user_id,
                    "system": system,
                    "system_label": system_label,
                    "simulated_time_s": 0.0,
                    "cumulative_completed": 0,
                    "cumulative_completion_rate": 0.0,
                    "phrase_attempts": attempts,
                    "event_type": "start",
                    "plot_endpoint_s": endpoint,
                }
            )
            for rank, completion_time in enumerate(times, start=1):
                rows.append(
                    {
                        "scope_user_id": scope_user_id,
                        "system": system,
                        "system_label": system_label,
                        "simulated_time_s": float(completion_time),
                        "cumulative_completed": rank,
                        "cumulative_completion_rate": rank / attempts,
                        "phrase_attempts": attempts,
                        "event_type": "completion",
                        "plot_endpoint_s": endpoint,
                    }
                )
            rows.append(
                {
                    "scope_user_id": scope_user_id,
                    "system": system,
                    "system_label": system_label,
                    "simulated_time_s": endpoint,
                    "cumulative_completed": len(times),
                    "cumulative_completion_rate": len(times) / attempts,
                    "phrase_attempts": attempts,
                    "event_type": "endpoint",
                    "plot_endpoint_s": endpoint,
                }
            )
    return pd.DataFrame(rows)


def build_paired_results(phrase_results: pd.DataFrame) -> pd.DataFrame:
    keys = ["user_id", "trial", "Comparison Phrase ID"]
    columns = keys + [
        "Target Phrase",
        "paired_click_schedule_id",
        "paired_offset_checksum",
        "phrase_completed",
        "num_clicks",
        "num_corrections",
        "correction_rate_percent",
        "simulated_attempt_time_s",
        "simulated_completion_time_s",
        "phrase_failure_reason",
    ]
    og = phrase_results[phrase_results["system"] == "og_nomon"][columns].copy()
    oneclick = phrase_results[phrase_results["system"] == "oneclick"][columns].copy()
    paired = og.merge(
        oneclick,
        on=keys,
        how="outer",
        validate="one_to_one",
        suffixes=("_og", "_oneclick"),
    )
    if paired[["Target Phrase_og", "Target Phrase_oneclick"]].isna().any().any():
        raise ValueError("paired comparison contains an unpaired phrase result")
    if not paired["Target Phrase_og"].eq(paired["Target Phrase_oneclick"]).all():
        raise ValueError("paired systems used different target phrases")
    if not paired["paired_offset_checksum_og"].eq(
        paired["paired_offset_checksum_oneclick"]
    ).all():
        raise ValueError("paired systems used different offset schedules")

    og_completed = paired["phrase_completed_og"].fillna(False).astype(bool)
    oneclick_completed = paired["phrase_completed_oneclick"].fillna(False).astype(bool)
    paired["paired_outcome"] = np.select(
        [
            og_completed & oneclick_completed,
            og_completed & ~oneclick_completed,
            ~og_completed & oneclick_completed,
        ],
        ["both_completed", "og_only", "oneclick_only"],
        default="both_failed",
    )
    mutual = og_completed & oneclick_completed
    paired["mutually_completed"] = mutual
    paired["paired_click_reduction"] = np.where(
        mutual,
        paired["num_clicks_og"] - paired["num_clicks_oneclick"],
        np.nan,
    )
    paired["paired_click_reduction_percent"] = np.where(
        mutual & (paired["num_clicks_og"] > 0),
        paired["paired_click_reduction"] / paired["num_clicks_og"] * 100.0,
        np.nan,
    )
    paired["paired_time_reduction_s"] = np.where(
        mutual,
        paired["simulated_completion_time_s_og"]
        - paired["simulated_completion_time_s_oneclick"],
        np.nan,
    )
    paired["paired_time_reduction_percent"] = np.where(
        mutual & (paired["simulated_completion_time_s_og"] > 0),
        paired["paired_time_reduction_s"]
        / paired["simulated_completion_time_s_og"]
        * 100.0,
        np.nan,
    )
    return paired


def build_paired_summary(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scopes: list[tuple[str, pd.DataFrame]] = [("ALL", paired)]
    scopes.extend(
        (str(user_id), group.copy())
        for user_id, group in paired.groupby("user_id", sort=False)
    )
    for user_id, group in scopes:
        mutual = group["mutually_completed"].fillna(False).astype(bool)
        outcome_counts = group["paired_outcome"].value_counts()
        rows.append(
            {
                "user_id": user_id,
                "paired_phrase_trials": int(len(group)),
                "both_completed": int(outcome_counts.get("both_completed", 0)),
                "og_only": int(outcome_counts.get("og_only", 0)),
                "oneclick_only": int(outcome_counts.get("oneclick_only", 0)),
                "both_failed": int(outcome_counts.get("both_failed", 0)),
                "mutually_completed_phrase_trials": int(mutual.sum()),
                "mutually_completed_og_mean_clicks": float(
                    pd.to_numeric(
                        group.loc[mutual, "num_clicks_og"],
                        errors="coerce",
                    ).mean()
                ),
                "mutually_completed_oneclick_mean_clicks": float(
                    pd.to_numeric(
                        group.loc[mutual, "num_clicks_oneclick"],
                        errors="coerce",
                    ).mean()
                ),
                "mutually_completed_mean_click_reduction_percent": float(
                    pd.to_numeric(
                        group.loc[mutual, "paired_click_reduction_percent"],
                        errors="coerce",
                    ).mean()
                ),
                "mutually_completed_og_mean_time_s": float(
                    pd.to_numeric(
                        group.loc[mutual, "simulated_completion_time_s_og"],
                        errors="coerce",
                    ).mean()
                ),
                "mutually_completed_oneclick_mean_time_s": float(
                    pd.to_numeric(
                        group.loc[mutual, "simulated_completion_time_s_oneclick"],
                        errors="coerce",
                    ).mean()
                ),
                "mutually_completed_mean_time_reduction_percent": float(
                    pd.to_numeric(
                        group.loc[mutual, "paired_time_reduction_percent"],
                        errors="coerce",
                    ).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def build_trial_summary(phrase_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, group in phrase_results.groupby(
        ["user_id", "trial", "system", "system_label"],
        sort=False,
    ):
        user_id, trial, system, system_label = key
        completed = group["phrase_completed"].fillna(False).astype(bool)
        rows.append(
            {
                "user_id": user_id,
                "trial": int(trial),
                "system": system,
                "system_label": system_label,
                "phrase_attempts": int(len(group)),
                "completed_phrases": int(completed.sum()),
                "phrase_completion_rate": float(completed.mean()),
                "mean_clicks": float(
                    pd.to_numeric(group["num_clicks"], errors="coerce").mean()
                ),
                "mean_attempt_time_s": float(
                    pd.to_numeric(
                        group["simulated_attempt_time_s"],
                        errors="coerce",
                    ).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def build_failure_distribution(phrase_results: pd.DataFrame) -> pd.DataFrame:
    failed = phrase_results[
        ~phrase_results["phrase_completed"].fillna(False).astype(bool)
    ].copy()
    if failed.empty:
        return pd.DataFrame(
            columns=[
                "user_id",
                "system",
                "system_label",
                "phrase_failure_reason",
                "failure_count",
                "failure_share",
            ]
        )
    failed["phrase_failure_reason"] = (
        failed["phrase_failure_reason"]
        .fillna("")
        .replace("", "unknown_failure")
        .astype(str)
    )
    counts = (
        failed.groupby(
            ["user_id", "system", "system_label", "phrase_failure_reason"],
            dropna=False,
        )
        .size()
        .rename("failure_count")
        .reset_index()
    )
    counts["failure_share"] = counts["failure_count"] / counts.groupby(
        ["user_id", "system"]
    )["failure_count"].transform("sum")
    return counts


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def plot_step_curve(axis, curves: pd.DataFrame, system: str) -> None:
    curve = curves[curves["system"] == system].sort_values(
        ["simulated_time_s", "cumulative_completed"],
        kind="stable",
    )
    axis.step(
        curve["simulated_time_s"],
        curve["cumulative_completion_rate"],
        where="post",
        color=SYSTEM_COLORS[system],
        linewidth=2.3,
        label=SYSTEM_LABELS[system],
    )


def save_figure(figure: plt.Figure, png_path: Path) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_path, dpi=220, bbox_inches="tight")
    figure.savefig(png_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def create_plots(
    per_user_summary: pd.DataFrame,
    macro_summary: pd.DataFrame,
    curves: pd.DataFrame,
    paired_summary: pd.DataFrame,
    failures: pd.DataFrame,
    profile_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    configure_plot_style()
    plot_dir = output_dir / "plots"
    users = list(dict.fromkeys(per_user_summary["user_id"].astype(str)))
    profile_labels = {}
    absolute_spreads = (
        pd.to_numeric(profile_summary["stable_clock_period_s"], errors="raise")
        * pd.to_numeric(
            profile_summary["normalized_click_offset_sd"],
            errors="raise",
        )
    )
    maximum_spread = float(absolute_spreads.max())
    for profile in profile_summary.to_dict("records"):
        period_s = float(profile["stable_clock_period_s"])
        normalized_mean = float(profile["normalized_click_offset_mean"])
        spread_s = period_s * float(profile["normalized_click_offset_sd"])
        if str(profile["click_offset_bias"]) == "centered":
            bias_description = "centered"
        elif normalized_mean >= 0.05:
            bias_description = "late"
        elif normalized_mean > 0:
            bias_description = "slightly late"
        elif normalized_mean <= -0.05:
            bias_description = "early"
        else:
            bias_description = "slightly early"

        if spread_s <= 0.01:
            spread_description = "no measured spread"
        elif math.isclose(spread_s, maximum_spread, rel_tol=0.0, abs_tol=1e-9):
            spread_description = f"widest spread {spread_s:.2f}s"
        elif spread_s >= 0.25:
            spread_description = f"wider spread {spread_s:.2f}s"
        else:
            spread_description = f"spread {spread_s:.2f}s"
        profile_labels[str(profile["user_id"])] = (
            f"{str(profile['clock_pace']).capitalize()} pace · {bias_description}\n"
            f"{period_s:.2f}s period · {spread_description}"
        )

    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    positions = np.arange(len(users))
    width = 0.36
    for offset, system in zip((-width / 2, width / 2), SYSTEM_ORDER):
        values = (
            per_user_summary[per_user_summary["system"] == system]
            .set_index("user_id")
            .reindex(users)["phrase_completion_rate"]
            .to_numpy(float)
        )
        axes[0, 0].bar(
            positions + offset,
            values,
            width,
            label=SYSTEM_LABELS[system],
            color=SYSTEM_COLORS[system],
        )
    axes[0, 0].set_xticks(positions, users)
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0, 0].set_title("Exact phrase completion")
    axes[0, 0].set_ylabel("Completed phrase trials")
    axes[0, 0].legend(frameon=False)

    overall_curves = curves[curves["scope_user_id"] == "ALL"]
    for system in SYSTEM_ORDER:
        plot_step_curve(axes[0, 1], overall_curves, system)
    endpoint = float(overall_curves["plot_endpoint_s"].max())
    axes[0, 1].set_xlim(0, endpoint)
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].xaxis.set_major_locator(MultipleLocator(30))
    axes[0, 1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0, 1].grid(alpha=0.18)
    axes[0, 1].set_title("Phrases completed by simulated time")
    axes[0, 1].set_xlabel("Simulated clock-interaction time (s)")
    axes[0, 1].set_ylabel("Completed phrase trials")
    axes[0, 1].legend(frameon=False)

    metric_columns = [
        "macro_completion_by_60s",
        "macro_completion_by_120s",
        "macro_completion_by_180s",
        "macro_phrase_completion_rate",
    ]
    metric_labels = ["60 s", "120 s", "180 s", "Final"]
    metric_positions = np.arange(len(metric_labels))
    for offset, system in zip((-width / 2, width / 2), SYSTEM_ORDER):
        row = macro_summary[macro_summary["system"] == system].iloc[0]
        axes[1, 0].bar(
            metric_positions + offset,
            [row[column] for column in metric_columns],
            width,
            label=SYSTEM_LABELS[system],
            color=SYSTEM_COLORS[system],
        )
    axes[1, 0].set_xticks(metric_positions, metric_labels)
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1, 0].set_title("Equal-user completion summary")
    axes[1, 0].set_ylabel("Completed phrase trials")

    paired_users = paired_summary[paired_summary["user_id"] != "ALL"].set_index(
        "user_id"
    ).reindex(users)
    click_values = paired_users[
        "mutually_completed_mean_click_reduction_percent"
    ].to_numpy(float)
    axes[1, 1].bar(positions, click_values, color="#72B7B2")
    axes[1, 1].axhline(0, color="#444444", linewidth=0.8)
    axes[1, 1].set_xticks(positions, users)
    axes[1, 1].set_title("Paired click reduction, both systems successful")
    axes[1, 1].set_ylabel("OneClick reduction relative to OG Nomon (%)")
    figure.tight_layout()
    save_figure(figure, plot_dir / "comparison_dashboard.png")

    figure, axes = plt.subplots(2, 3, figsize=(15, 9.5), sharey=True)
    for axis, user_id in zip(axes.flat, users):
        user_curves = curves[curves["scope_user_id"] == user_id]
        for system in SYSTEM_ORDER:
            plot_step_curve(axis, user_curves, system)
        endpoint = float(user_curves["plot_endpoint_s"].max())
        axis.set_xlim(0, endpoint)
        axis.set_ylim(0, 1.05)
        axis.xaxis.set_major_locator(MultipleLocator(60))
        axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        axis.grid(alpha=0.18)
        description = profile_labels.get(str(user_id), "")
        title = f"User {user_id}"
        if description:
            title += f"\n{description}"
        axis.set_title(title, fontsize=11.5, linespacing=1.25)
        axis.set_xlabel("Simulated time (s)")
    axes[0, 0].set_ylabel("Completed phrase trials")
    axes[1, 0].set_ylabel("Completed phrase trials")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    figure.tight_layout(rect=(0, 0.055, 1, 1), h_pad=2.8)
    save_figure(figure, plot_dir / "per_user_completion_by_time.png")

    if not failures.empty:
        failure_pivot = (
            failures.groupby(["system_label", "phrase_failure_reason"])[
                "failure_count"
            ]
            .sum()
            .unstack(fill_value=0)
        )
        figure, axis = plt.subplots(figsize=(11, 6))
        labels = failure_pivot.index.astype(str).tolist()
        left = np.zeros(len(failure_pivot), dtype=float)
        colors = plt.get_cmap("Set2")(
            np.linspace(0.0, 1.0, max(len(failure_pivot.columns), 1))
        )
        for color, reason in zip(colors, failure_pivot.columns):
            values = failure_pivot[reason].to_numpy(float)
            axis.barh(
                labels,
                values,
                left=left,
                color=color,
                label=str(reason),
            )
            left += values
        axis.set_xlabel("Failed phrase trials")
        axis.set_ylabel("")
        axis.set_title("Failure reasons")
        axis.legend(
            title="Failure reason",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            frameon=False,
        )
        figure.tight_layout()
        save_figure(figure, plot_dir / "failure_reasons.png")


def write_report(
    output_dir: Path,
    macro_summary: pd.DataFrame,
    paired_summary: pd.DataFrame,
) -> None:
    rows = macro_summary.set_index("system")
    paired = paired_summary[paired_summary["user_id"] == "ALL"].iloc[0]
    lines = [
        "# OG Nomon vs OneClick Held-Out Comparison",
        "",
        "OneClick uses the globally selected 6.0-second Space and 6.0-second "
        "Enter/Undo periods with Protected Undo. OG Nomon uses each user’s "
        "historical regime-aware timing profile.",
        "",
        "## Completion",
        "",
        "| System | Final | By 60 s | By 120 s | By 180 s |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for system in SYSTEM_ORDER:
        row = rows.loc[system]
        lines.append(
            f"| {SYSTEM_LABELS[system]} | "
            f"{row['macro_phrase_completion_rate'] * 100:.1f}% | "
            f"{row['macro_completion_by_60s'] * 100:.1f}% | "
            f"{row['macro_completion_by_120s'] * 100:.1f}% | "
            f"{row['macro_completion_by_180s'] * 100:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Paired efficiency",
            "",
            f"- Mutually completed phrase trials: "
            f"{int(paired['mutually_completed_phrase_trials'])}.",
            f"- Mean paired OneClick click reduction: "
            f"{paired['mutually_completed_mean_click_reduction_percent']:.1f}%.",
            f"- Mean paired OneClick simulated-time reduction: "
            f"{paired['mutually_completed_mean_time_reduction_percent']:.1f}%.",
            "",
            "Efficiency reductions are conditioned on both systems completing the "
            "same phrase trial. Failed phrases remain in all completion denominators "
            "and never receive an artificial completion time.",
        ]
    )
    (output_dir / "comparison_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def expected_config(args: argparse.Namespace, selection_run_dir: Path) -> dict[str, Any]:
    return {
        "experiment": "og_nomon_vs_selected_global_oneclick",
        "users": parse_csv_values(args.users),
        "trials": int(args.trials),
        "phrase_count": int(args.phrase_count),
        "seed": int(args.seed),
        "selection_run_dir": str(selection_run_dir.resolve()),
        "space_clock_period_s": DEFAULT_SPACE_PERIOD_S,
        "enter_clock_period_s": DEFAULT_ENTER_PERIOD_S,
        "undo_mode": "protected",
        "validation_fraction": float(args.validation_fraction),
        "clicks_per_phrase": int(args.clicks_per_phrase),
        "calibration_clicks": int(args.calibration_clicks),
        "max_word_attempts": int(args.max_word_attempts),
        "max_enter_attempts": int(args.max_enter_attempts),
        "max_clicks_per_word": int(args.max_clicks_per_word),
        "dead_time_mode": "zero_active_dead_time",
        "phrase_time_ceiling_s": None,
        "og_clock_policy": "historical_regime_aware_selection_bootstrap",
        "oneclick_clock_policy": "fixed_6.0s_space_6.0s_enter",
        "offset_transfer": "paired_absolute_seconds",
        "lm_backend": str(args.lm_backend),
        "lm_size": str(args.lm_size),
        "lm_cache_dir": str(args.lm_cache_dir.resolve()),
        "oneclick_cache_dir": str(args.oneclick_cache_dir.resolve()),
        "strict_lm_errors": True,
        "worker_count": 1,
        "time_horizons_s": list(TIME_HORIZONS_S),
    }


def load_or_initialize_config(
    run_dir: Path,
    config: dict[str, Any],
) -> None:
    path = run_dir / "run_config.json"
    if path.is_file():
        saved = json.loads(path.read_text(encoding="utf-8"))
        comparable_saved = {
            key: saved.get(key)
            for key in config
        }
        if comparable_saved != config:
            raise ValueError("resume configuration differs from the saved run")
    else:
        atomic_write_json(config, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--users", default=",".join(DEFAULT_USERS))
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--phrase-count", type=int, default=DEFAULT_PHRASES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--clicks-per-phrase", type=int, default=500)
    parser.add_argument("--calibration-clicks", type=int, default=200)
    parser.add_argument("--max-word-attempts", type=int, default=5)
    parser.add_argument("--max-enter-attempts", type=int, default=5)
    parser.add_argument("--max-clicks-per-word", type=int, default=30)
    parser.add_argument(
        "--lm-backend",
        choices=["kenlm", "imagineville"],
        default="imagineville",
    )
    parser.add_argument("--lm-size", choices=["tiny", "medium"], default="tiny")
    parser.add_argument("--lm-cache-dir", type=Path, default=DEFAULT_LM_CACHE_DIR)
    parser.add_argument(
        "--oneclick-cache-dir",
        type=Path,
        default=Path(".cache") / "oneclick_phrase_audit",
    )
    parser.add_argument("--selection-run-dir", type=Path, default=None)
    parser.add_argument("--resume-run-dir", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
    )
    return parser.parse_args()


def run_comparison(args: argparse.Namespace) -> Path:
    users = parse_csv_values(args.users)
    if not users:
        raise ValueError("at least one user is required")
    if args.trials < 1 or args.phrase_count < 1:
        raise ValueError("trials and phrase count must be positive")
    if args.clicks_per_phrase < 1:
        raise ValueError("clicks per phrase must be positive")

    selection_run_dir = (
        args.selection_run_dir.resolve()
        if args.selection_run_dir is not None
        else find_latest_selection_run(args.output_dir)
    )
    validate_selected_periods(selection_run_dir)
    run_dir = (
        args.resume_run_dir.resolve()
        if args.resume_run_dir is not None
        else build_output_dir(args.output_dir)
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    config = expected_config(args, selection_run_dir)
    load_or_initialize_config(run_dir, config)

    phrase_df = prepare_heldout_phrase_set(
        run_dir,
        selection_run_dir,
        phrase_count=args.phrase_count,
        seed=args.seed,
    )
    phrase_ids = set(phrase_df["Comparison Phrase ID"].astype(str))
    config_path = run_dir / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update(
        {
            "heldout_phrase_set_checksum": phrase_set_checksum(phrase_df),
            "tuning_phrase_set_checksum": file_checksum(
                selection_run_dir / "common_phrase_set.csv"
            ),
            "heldout_reachability_audit_checksum": file_checksum(
                run_dir / "heldout_phrase_reachability_audit.csv"
            ),
            "tuning_phrase_exclusion_checksum": file_checksum(
                run_dir / "tuning_phrase_exclusions.csv"
            ),
            "analyzed_phrase_attempts": int(
                len(users) * args.trials * len(phrase_df) * len(SYSTEM_ORDER)
            ),
            "condition_count": int(len(users) * args.trials * len(SYSTEM_ORDER)),
            "condition_attempts": CONDITION_ATTEMPTS,
            "condition_retry_delay_s": CONDITION_RETRY_DELAY_S,
        }
    )
    atomic_write_json(config, config_path)

    profiles: dict[str, dict[str, Any]] = {}
    training_clicks: dict[str, pd.DataFrame] = {}
    profile_rows = []
    user_config_rows = []
    for user_id in users:
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
        profile = estimate_profile(
            user_id,
            train_click_df,
            train_phrase_df,
            validation_click_df,
            normalize_phrase_order(validation_phrase_df),
            train_sessions,
            validation_sessions,
        )
        profiles[user_id] = profile
        training_clicks[user_id] = train_click_df
        profile_rows.append(
            bootstrap_profile_row(user_id, profile, train_sessions, train_click_df)
        )
        user_config_rows.append(
            {
                "user_id": user_id,
                "train_sessions": ",".join(map(str, train_sessions)),
                "validation_sessions": ",".join(map(str, validation_sessions)),
                "bootstrap_source_sessions": ",".join(
                    map(
                        str,
                        effective_bootstrap_source_sessions(profile, train_sessions),
                    )
                ),
            }
        )
    atomic_write_csv(pd.DataFrame(profile_rows), run_dir / "user_bootstrap_profiles.csv")
    atomic_write_csv(pd.DataFrame(user_config_rows), run_dir / "user_data_splits.csv")

    schedule_metadata_rows = []
    schedules: dict[tuple[str, int], pd.DataFrame] = {}
    schedule_metadata: dict[tuple[str, int], dict[str, Any]] = {}
    for user_id in users:
        for trial in range(args.trials):
            schedule, metadata = load_or_build_schedule(
                run_dir=run_dir,
                user_id=user_id,
                trial=trial,
                profile=profiles[user_id],
                phrase_df=phrase_df,
                train_click_df=training_clicks[user_id],
                seed=schedule_seed(args.seed, user_id, trial),
                clicks_per_phrase=args.clicks_per_phrase,
                calibration_clicks=args.calibration_clicks,
            )
            schedules[(user_id, trial)] = schedule
            schedule_metadata[(user_id, trial)] = metadata
            schedule_metadata_rows.append(metadata)
    atomic_write_csv(
        pd.DataFrame(schedule_metadata_rows),
        run_dir / "paired_schedule_checksums.csv",
    )

    manifest_path = run_dir / "manifest.csv"
    manifest = build_manifest(run_dir, users, args.trials, phrase_ids)
    atomic_write_csv(manifest, manifest_path)
    for user_id in users:
        for trial in range(args.trials):
            schedule = schedules[(user_id, trial)]
            metadata = schedule_metadata[(user_id, trial)]
            for system in SYSTEM_ORDER:
                path = condition_path(run_dir, user_id, system, trial)
                if condition_is_complete(
                    path,
                    user_id,
                    system,
                    trial,
                    phrase_ids,
                ):
                    print(
                        f"Skipping completed condition: user {user_id}, "
                        f"{SYSTEM_LABELS[system]}, trial {trial + 1}/{args.trials}"
                    )
                    continue
                print(
                    f"Running user {user_id}, {SYSTEM_LABELS[system]}, "
                    f"trial {trial + 1}/{args.trials}"
                )
                for condition_attempt in range(1, CONDITION_ATTEMPTS + 1):
                    try:
                        if system == "og_nomon":
                            raw = run_old_nomon_bootstrap(
                                click_df=schedule.copy(),
                                phrase_df=phrase_df,
                                lm_backend=args.lm_backend,
                                lm_size=args.lm_size,
                                lm_cache_dir=args.lm_cache_dir,
                                selection_bootstrap_seed=schedule_seed(
                                    args.seed,
                                    user_id,
                                    trial,
                                ),
                                verbose=args.verbose,
                            )
                        else:
                            raw = run_oneclick(
                                click_df=schedule.copy(),
                                phrase_df=phrase_df,
                                max_word_attempts=args.max_word_attempts,
                                max_enter_attempts=args.max_enter_attempts,
                                max_clicks_per_word=args.max_clicks_per_word,
                                undo_mode="protected",
                                oneclick_cache_dir=args.oneclick_cache_dir,
                                perfect_letter_observations=False,
                                verbose=args.verbose,
                                fixed_space_clock_period_s=DEFAULT_SPACE_PERIOD_S,
                                fixed_enter_clock_period_s=DEFAULT_ENTER_PERIOD_S,
                                strict_lm_errors=True,
                            )
                        break
                    except RuntimeError:
                        if condition_attempt >= CONDITION_ATTEMPTS:
                            raise
                        print(
                            f"Strict LM condition attempt {condition_attempt}/"
                            f"{CONDITION_ATTEMPTS} failed; retrying in "
                            f"{CONDITION_RETRY_DELAY_S:.0f} seconds"
                        )
                        time.sleep(CONDITION_RETRY_DELAY_S)
                normalized = normalize_condition_results(
                    raw,
                    user_id=user_id,
                    system=system,
                    trial=trial,
                    schedule_metadata=metadata,
                )
                validate_condition(
                    normalized,
                    user_id=user_id,
                    system=system,
                    trial=trial,
                    phrase_ids=phrase_ids,
                )
                atomic_write_csv(normalized, path)
                if not condition_is_complete(
                    path,
                    user_id,
                    system,
                    trial,
                    phrase_ids,
                ):
                    raise ValueError(f"saved condition failed validation: {path}")
                manifest = build_manifest(run_dir, users, args.trials, phrase_ids)
                atomic_write_csv(manifest, manifest_path)

    manifest = build_manifest(run_dir, users, args.trials, phrase_ids)
    atomic_write_csv(manifest, manifest_path)
    if not manifest["status"].eq("completed").all():
        raise RuntimeError("comparison ended with pending conditions")

    phrase_results = pd.concat(
        [
            pd.read_csv(run_dir / relative_path)
            for relative_path in manifest["condition_file"]
        ],
        ignore_index=True,
    )
    expected_rows = len(users) * args.trials * len(phrase_df) * len(SYSTEM_ORDER)
    if len(phrase_results) != expected_rows:
        raise ValueError(
            f"expected {expected_rows} phrase rows, found {len(phrase_results)}"
        )
    per_user_summary = summarize_system_results(phrase_results)
    macro_summary = build_macro_summary(per_user_summary)
    curves = build_curve_points(phrase_results)
    paired = build_paired_results(phrase_results)
    paired_summary = build_paired_summary(paired)
    trial_summary = build_trial_summary(phrase_results)
    failures = build_failure_distribution(phrase_results)

    atomic_write_csv(phrase_results, run_dir / "system_phrase_results.csv")
    atomic_write_csv(per_user_summary, run_dir / "per_user_summary.csv")
    atomic_write_csv(macro_summary, run_dir / "macro_summary.csv")
    atomic_write_csv(curves, run_dir / "completion_by_time_curve_points.csv")
    atomic_write_csv(paired, run_dir / "paired_phrase_results.csv")
    atomic_write_csv(paired_summary, run_dir / "paired_summary.csv")
    atomic_write_csv(trial_summary, run_dir / "trial_summary.csv")
    atomic_write_csv(failures, run_dir / "failure_distribution.csv")
    create_plots(
        per_user_summary,
        macro_summary,
        curves,
        paired_summary,
        failures,
        pd.DataFrame(profile_rows),
        run_dir,
    )
    write_report(run_dir, macro_summary, paired_summary)
    print(f"Saved held-out OG Nomon vs OneClick comparison to: {run_dir}")
    print(
        per_user_summary[
            [
                "user_id",
                "system_label",
                "phrase_completion_rate",
                "completion_by_120s",
                "mean_attempt_clicks",
            ]
        ].to_string(index=False)
    )
    return run_dir


def main() -> None:
    run_comparison(parse_args())


if __name__ == "__main__":
    main()
