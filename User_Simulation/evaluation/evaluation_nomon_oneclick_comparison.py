"""Compare OG Nomon and OneClick on shared Gaussian synthetic click schedules.

Run from the repository root:

    python -m User_Simulation.evaluation.evaluation_nomon_oneclick_comparison
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from OneClick_Simulation.simulated_user import SimulatedUser as OneClickSimulatedUser
from User_Simulation.evaluation.evaluation_baseline import (
    DEFAULT_LM_CACHE_DIR,
    REPO_ROOT,
    load_text_click_data,
    load_text_phrase_data,
    lm_parameters,
    normalize_phrase_order,
    write_json,
)
from User_Simulation.evaluation.evaluation_click_offset_std_sweep import (
    CLICK_OFFSET_COL,
    CLOCK_PERIOD_COL,
    assign_clock_regimes,
    build_clock_regimes,
    build_regime_statistics,
)
from User_Simulation.evaluation.evaluation_oneclick_phrase_audit import (
    CachedOneClickWordClient,
    PHRASE_STATUS_ERROR,
    audit_phrase,
)
from User_Simulation.evaluation.synthetic_profiles import split_sessions
from User_Simulation.simulated_user_text import SimulatedUser as OldNomonSimulatedUser


DEFAULT_USERS = ("A", "B", "C", "D", "F", "G")
CLICK_OFFSET_ALIAS_COL = "Click Time Rlative (s)"
DEAD_TIME_COL = "Dead Time (s)"
DEFAULT_STD_SWEEP_GLOB = "click_offset_std_sweep_*"


def build_output_dir(base_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    output_dir = base_dir / f"nomon_oneclick_comparison_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def parse_csv_values(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def find_latest_std_sweep(outputs_dir: Path) -> Path:
    for output_dir in sorted(outputs_dir.glob(DEFAULT_STD_SWEEP_GLOB), reverse=True):
        if (output_dir / "selected_std_multipliers.csv").is_file():
            return output_dir
    raise FileNotFoundError(f"No complete click-offset std sweep found under {outputs_dir}")


def load_selected_std_multipliers(std_sweep_dir: Path, user_ids: list[str]) -> pd.DataFrame:
    selected_df = pd.read_csv(std_sweep_dir / "selected_std_multipliers.csv")
    missing = sorted(set(user_ids) - set(selected_df["user_id"].astype(str)))
    if missing:
        raise ValueError(f"Selected std multiplier file is missing users: {missing}")
    return selected_df[selected_df["user_id"].isin(user_ids)].copy()


def clean_click_rows(click_df: pd.DataFrame) -> pd.DataFrame:
    result = click_df.copy()
    result[CLOCK_PERIOD_COL] = pd.to_numeric(result[CLOCK_PERIOD_COL], errors="coerce")
    result[CLICK_OFFSET_COL] = pd.to_numeric(result[CLICK_OFFSET_COL], errors="coerce")
    invalid = result[[CLOCK_PERIOD_COL, CLICK_OFFSET_COL]].isna().any(axis=1)
    if invalid.any():
        raise ValueError(f"Found {int(invalid.sum())} click rows without numeric timing values")
    return result


def selected_multiplier_label(multiplier: float) -> str:
    if multiplier < 1.0:
        return "tighter"
    if multiplier > 1.0:
        return "noisier"
    return "original-scale"


def timing_profile_row(
    user_id: str,
    train_click_df: pd.DataFrame,
    regime_stats_df: pd.DataFrame,
    selected_row: pd.Series,
) -> dict[str, Any]:
    periods = sorted(float(value) for value in regime_stats_df["period_center_s"])
    if len(periods) > 1:
        clock_pace = "mixed"
    elif periods[0] < 2.0:
        clock_pace = "fast"
    elif periods[0] <= 4.0:
        clock_pace = "moderate"
    else:
        clock_pace = "slow"

    normalized_offsets = (
        train_click_df[CLICK_OFFSET_COL].astype(float)
        / train_click_df[CLOCK_PERIOD_COL].astype(float)
    )
    normalized_sd = float(normalized_offsets.std(ddof=0))
    if normalized_sd < 0.05:
        timing_variability = "low"
    elif normalized_sd <= 0.10:
        timing_variability = "moderate"
    else:
        timing_variability = "high"

    normalized_mean = float(normalized_offsets.mean())
    if abs(normalized_mean) <= 0.02:
        click_offset_bias = "centered"
    elif normalized_mean > 0:
        click_offset_bias = "late/positive"
    else:
        click_offset_bias = "early/negative"

    multiplier = float(selected_row["selected_std_multiplier"])
    confidence = str(selected_row.get("confidence", "standard"))
    description = (
        f"{clock_pace.capitalize()}-clock Gaussian profile with {timing_variability} "
        f"timing variability, {click_offset_bias} click-offset bias, and "
        f"{selected_multiplier_label(multiplier)} selected SD."
    )
    if confidence == "low_sample":
        description += " Low-sample timing estimate."

    return {
        "user_id": user_id,
        "clock_pace": clock_pace,
        "timing_variability": timing_variability,
        "click_offset_bias": click_offset_bias,
        "selected_std_multiplier": multiplier,
        "selected_std_interpretation": selected_multiplier_label(multiplier),
        "confidence": confidence,
        "low_sample": bool(selected_row.get("low_sample", False)),
        "training_click_rows": int(selected_row["train_click_rows"]),
        "validation_click_rows": int(selected_row["validation_click_rows"]),
        "observed_clock_regimes_s": ",".join(f"{period:g}" for period in periods),
        "normalized_click_offset_mean": normalized_mean,
        "normalized_click_offset_sd": normalized_sd,
        "qualitative_description": description,
    }


def phrase_period_lookup(validation_click_df: pd.DataFrame) -> dict[tuple[int, int], float]:
    lookup: dict[tuple[int, int], float] = {}
    required = {"Session Num", "Phrase Num", CLOCK_PERIOD_COL}
    if not required.issubset(validation_click_df.columns):
        return lookup
    for key, group_df in validation_click_df.groupby(["Session Num", "Phrase Num"], dropna=False):
        periods = pd.to_numeric(group_df[CLOCK_PERIOD_COL], errors="coerce").dropna()
        if not periods.empty:
            session_num, phrase_num = key
            lookup[(int(session_num), int(phrase_num))] = float(periods.median())
    return lookup


def build_comparison_phrase_df(
    user_id: str,
    validation_phrase_df: pd.DataFrame,
    validation_click_df: pd.DataFrame,
    regimes_df: pd.DataFrame,
    phrase_policy: str,
    oneclick_cache_dir: Path,
    max_phrases: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    phrase_df = normalize_phrase_order(validation_phrase_df).copy()
    phrase_df["Original Session Num"] = phrase_df["Session Num"].astype(int)
    phrase_df["Original Phrase Num"] = phrase_df["Phrase Num"].astype(int)
    phrase_df["Comparison Phrase ID"] = [
        f"{user_id}_{int(row['Session Num'])}_{int(row['Phrase Num'])}"
        for _, row in phrase_df.iterrows()
    ]

    audit_df = pd.DataFrame()
    if phrase_policy == "reachable":
        client = CachedOneClickWordClient(oneclick_cache_dir)
        audit_rows = []
        for row in phrase_df.to_dict("records"):
            audit_record = {
                "Phrase ID": row["Comparison Phrase ID"],
                "Phrase Text": row["Phrase Text"],
                "Source User ID": user_id,
            }
            phrase_audit, _word_rows = audit_phrase(audit_record, client)
            audit_rows.append(phrase_audit)
        audit_df = pd.DataFrame(audit_rows)
        error_rows = audit_df[audit_df["phrase_status"] == PHRASE_STATUS_ERROR]
        if not error_rows.empty:
            examples = "; ".join(error_rows["phrase_id"].astype(str).head(3))
            raise RuntimeError(f"OneClick phrase audit failed for {len(error_rows)} phrases: {examples}")
        reachable_ids = set(
            audit_df.loc[audit_df["all_words_overall_reachable"], "phrase_id"].astype(str)
        )
        phrase_df = phrase_df[phrase_df["Comparison Phrase ID"].isin(reachable_ids)].copy()

    if max_phrases is not None:
        phrase_df = phrase_df.head(max_phrases).copy()
    if phrase_df.empty:
        raise ValueError(f"User {user_id} has no phrases available after phrase policy filtering")

    period_lookup = phrase_period_lookup(validation_click_df)
    fallback_period = float(regimes_df["period_center_s"].iloc[0])
    phrase_df["Original Clock Period (s)"] = [
        period_lookup.get(
            (int(row["Original Session Num"]), int(row["Original Phrase Num"])),
            fallback_period,
        )
        for _, row in phrase_df.iterrows()
    ]
    regime_assignment_df = phrase_df.drop(
        columns=[CLOCK_PERIOD_COL],
        errors="ignore",
    ).copy()
    regime_assignment_df[CLOCK_PERIOD_COL] = phrase_df["Original Clock Period (s)"]
    assigned = assign_clock_regimes(regime_assignment_df, regimes_df)
    phrase_df["Clock Regime"] = assigned["Clock Regime"].values
    phrase_df["Synthetic Session Num"] = np.arange(1, len(phrase_df) + 1)
    phrase_df["Synthetic Phrase Num"] = 1
    phrase_df["Session Num"] = phrase_df["Synthetic Session Num"]
    phrase_df["Phrase Num"] = phrase_df["Synthetic Phrase Num"]
    return phrase_df.reset_index(drop=True), audit_df


def _sample_regime_offsets(
    regime_stats_df: pd.DataFrame,
    regime_id: str,
    std_multiplier: float,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    stats = regime_stats_df.set_index("regime_id").loc[regime_id]
    return rng.normal(
        loc=float(stats["offset_mean_s"]),
        scale=float(stats["offset_std_s"]) * std_multiplier,
        size=count,
    )


def build_shared_synthetic_click_df(
    phrase_df: pd.DataFrame,
    regime_stats_df: pd.DataFrame,
    selected_std_multiplier: float,
    clicks_per_phrase: int,
    calibration_clicks: int,
    rng: np.random.Generator,
    perfect_clicks: bool = False,
) -> pd.DataFrame:
    stats_by_regime = regime_stats_df.set_index("regime_id")
    rows: list[dict[str, Any]] = []

    eligible_stats = regime_stats_df[regime_stats_df["eligible"]].copy()
    weights = eligible_stats["train_click_rows"].astype(float).to_numpy()
    weights = weights / weights.sum()
    calibration_regimes = rng.choice(
        eligible_stats["regime_id"].to_numpy(),
        size=calibration_clicks,
        p=weights,
    )
    for index, regime_id in enumerate(calibration_regimes, start=1):
        stats = stats_by_regime.loc[regime_id]
        offset = 0.0 if perfect_clicks else _sample_regime_offsets(
            regime_stats_df, str(regime_id), selected_std_multiplier, 1, rng
        )[0]
        rows.append(
            {
                "Session Num": np.nan,
                "Phrase Num": np.nan,
                "Click Num": index,
                CLOCK_PERIOD_COL: float(stats["period_center_s"]),
                CLICK_OFFSET_COL: float(offset),
                CLICK_OFFSET_ALIAS_COL: float(offset),
                DEAD_TIME_COL: np.nan,
                "Synthetic Source": "gaussian_calibration",
            }
        )

    for phrase_row in phrase_df.to_dict("records"):
        regime_id = str(phrase_row["Clock Regime"])
        stats = stats_by_regime.loc[regime_id]
        offsets = (
            np.zeros(clicks_per_phrase)
            if perfect_clicks
            else _sample_regime_offsets(
                regime_stats_df,
                regime_id,
                selected_std_multiplier,
                clicks_per_phrase,
                rng,
            )
        )
        for click_num, offset in enumerate(offsets, start=1):
            rows.append(
                {
                    "Session Num": int(phrase_row["Session Num"]),
                    "Phrase Num": int(phrase_row["Phrase Num"]),
                    "Click Num": click_num,
                    CLOCK_PERIOD_COL: float(stats["period_center_s"]),
                    CLICK_OFFSET_COL: float(offset),
                    CLICK_OFFSET_ALIAS_COL: float(offset),
                    DEAD_TIME_COL: 0.0,
                    "Comparison Phrase ID": phrase_row["Comparison Phrase ID"],
                    "Synthetic Source": "gaussian_active",
                }
            )
    return pd.DataFrame(rows)


def run_old_nomon(
    click_df: pd.DataFrame,
    phrase_df: pd.DataFrame,
    lm_backend: str,
    lm_size: str,
    lm_cache_dir: Path,
    force_target_clock_selection: bool,
    verbose: bool,
) -> pd.DataFrame:
    sim = OldNomonSimulatedUser()
    params = {
        "click_df": click_df,
        "phrase_df": phrase_df,
        "record_attempted_phrases": True,
        "append_terminal_periods": False,
        "force_target_clock_selection": force_target_clock_selection,
    }
    params.update(lm_parameters(lm_backend, lm_size, lm_cache_dir, None))

    original_cwd = Path.cwd()
    try:
        os.chdir(REPO_ROOT / "User_Simulation")
        sim.parameter_metrics(params, trials=1, verbose=verbose)
    finally:
        os.chdir(original_cwd)
    return sim.result_df.copy()


def run_oneclick(
    click_df: pd.DataFrame,
    phrase_df: pd.DataFrame,
    max_word_attempts: int,
    max_enter_attempts: int,
    max_clicks_per_word: int,
    undo_mode: str,
    oneclick_cache_dir: Path,
    perfect_letter_observations: bool,
    verbose: bool,
    fixed_clock_period_s: float | None = None,
    fixed_space_clock_period_s: float | None = None,
    fixed_enter_clock_period_s: float | None = None,
    strict_lm_errors: bool = False,
) -> pd.DataFrame:
    sim = OneClickSimulatedUser()
    params = {
        "click_df": click_df,
        "phrase_df": phrase_df,
        "record_attempted_phrases": True,
        "max_word_attempts": max_word_attempts,
        "max_enter_attempts": max_enter_attempts,
        "max_clicks_per_word": max_clicks_per_word,
        "undo_mode": undo_mode,
        "stop_phrase_on_failed_word": True,
        "perfect_letter_observations": perfect_letter_observations,
        "oneclick_lm_config": {
            "cache_dir": str(oneclick_cache_dir),
            "strict_errors": bool(strict_lm_errors),
        },
    }
    if fixed_clock_period_s is not None:
        params["fixed_clock_period_s"] = float(fixed_clock_period_s)
    if fixed_space_clock_period_s is not None:
        params["fixed_space_clock_period_s"] = float(fixed_space_clock_period_s)
    if fixed_enter_clock_period_s is not None:
        params["fixed_enter_clock_period_s"] = float(fixed_enter_clock_period_s)
    sim.parameter_metrics(params, trials=1, verbose=verbose)
    return sim.result_df.copy()


def normalize_system_results(
    result_df: pd.DataFrame,
    system: str,
    user_id: str,
    trial: int,
) -> pd.DataFrame:
    if result_df.empty:
        return pd.DataFrame()
    result = result_df.copy()
    result["system"] = system
    result["user_id"] = user_id
    result["trial"] = trial
    result["num_clicks"] = pd.to_numeric(result.get("Num Clicks"), errors="coerce")
    result["num_corrections"] = pd.to_numeric(result.get("Num Corrections"), errors="coerce")
    result["correction_rate_percent"] = pd.to_numeric(
        result.get("Correction Rate (%)"), errors="coerce"
    )
    result["error_rate_percent"] = pd.to_numeric(result.get("Error Rate (%)"), errors="coerce")
    result["phrase_completed"] = result.get("Phrase Completed", False).astype(bool)
    result["target_word_count"] = pd.to_numeric(result.get("Target Word Count"), errors="coerce")
    result["completed_word_count"] = pd.to_numeric(
        result.get("Completed Word Count"), errors="coerce"
    )
    result["failed_word_count"] = pd.to_numeric(result.get("Failed Word Count"), errors="coerce")
    result["word_attempt_count"] = pd.to_numeric(result.get("Word Attempt Count"), errors="coerce")
    result["word_completion_rate_percent"] = pd.to_numeric(
        result.get("Word Completion Rate (%)"), errors="coerce"
    )
    recovery_columns = {
        "enter_retry_count": "Enter Retry Count",
        "wrong_word_commit_count": "Wrong Word Commit Count",
        "undo_attempt_count": "Undo Attempt Count",
        "undo_failure_count": "Undo Failure Count",
        "restored_state_count": "Restored State Count",
        "letter_press_count": "Letter Press Count",
        "target_enter_attempt_count": "Target Enter Attempt Count",
        "failed_word_position": "Failed Word Position",
        "failed_word_attempt": "Failed Word Attempt",
        "failure_word_click_count": "Failure Word Click Count",
        "failure_letter_press_count": "Failure Letter Press Count",
        "failure_target_enter_attempt_count": "Failure Target Enter Attempt Count",
        "failure_undo_attempt_count": "Failure Undo Attempt Count",
        "simulated_attempt_time_s": "Simulated Attempt Time (s)",
        "simulated_completion_time_s": "Simulated Completion Time (s)",
        "letter_clock_time_s": "Letter Clock Time (s)",
        "target_enter_clock_time_s": "Target Enter Clock Time (s)",
        "undo_clock_time_s": "Undo Clock Time (s)",
        "simulated_time_accounting_error_s": "Simulated Time Accounting Error (s)",
    }
    for normalized, source in recovery_columns.items():
        if source in result:
            result[normalized] = pd.to_numeric(result[source], errors="coerce")
        else:
            result[normalized] = np.nan
    text_diagnostic_columns = {
        "phrase_failure_reason": "Phrase Failure Reason",
        "phrase_failure_stage": "Phrase Failure Stage",
        "phrase_failure_limit": "Phrase Failure Limit",
        "phrase_failure_guard": "Phrase Failure Guard",
        "failed_target_word": "Failed Target Word",
    }
    for normalized, source in text_diagnostic_columns.items():
        result[normalized] = result[source].fillna("").astype(str) if source in result else ""
    if "Failed Target Was Displayed" in result:
        result["failed_target_was_displayed"] = result[
            "Failed Target Was Displayed"
        ].fillna(False).astype(bool)
    else:
        result["failed_target_was_displayed"] = False
    return result[
        [
            "user_id",
            "trial",
            "system",
            "Comparison Phrase ID",
            "Target Phrase",
            "Typed Text",
            "num_clicks",
            "num_corrections",
            "correction_rate_percent",
            "error_rate_percent",
            "phrase_completed",
            "Completion Fraction",
            "target_word_count",
            "completed_word_count",
            "failed_word_count",
            "word_attempt_count",
            "word_completion_rate_percent",
            "enter_retry_count",
            "wrong_word_commit_count",
            "undo_attempt_count",
            "undo_failure_count",
            "restored_state_count",
            "letter_press_count",
            "target_enter_attempt_count",
            "phrase_failure_reason",
            "phrase_failure_stage",
            "phrase_failure_limit",
            "phrase_failure_guard",
            "failed_target_word",
            "failed_word_position",
            "failed_word_attempt",
            "failed_target_was_displayed",
            "failure_word_click_count",
            "failure_letter_press_count",
            "failure_target_enter_attempt_count",
            "failure_undo_attempt_count",
            "simulated_attempt_time_s",
            "simulated_completion_time_s",
            "letter_clock_time_s",
            "target_enter_clock_time_s",
            "undo_clock_time_s",
            "simulated_time_accounting_error_s",
        ]
    ]


def build_phrase_trial_comparison(system_results_df: pd.DataFrame) -> pd.DataFrame:
    key_columns = ["user_id", "trial", "Comparison Phrase ID"]
    og = system_results_df[system_results_df["system"] == "og_nomon"].copy()
    oneclick = system_results_df[system_results_df["system"] == "oneclick"].copy()
    merged = og.merge(
        oneclick,
        on=key_columns,
        how="outer",
        suffixes=("_og", "_oneclick"),
    )
    merged["click_reduction"] = merged["num_clicks_og"] - merged["num_clicks_oneclick"]
    merged["click_reduction_percent"] = np.where(
        merged["num_clicks_og"] > 0,
        merged["click_reduction"] / merged["num_clicks_og"] * 100,
        np.nan,
    )
    merged["correction_rate_difference"] = (
        merged["correction_rate_percent_oneclick"] - merged["correction_rate_percent_og"]
    )
    undo_only = system_results_df[
        system_results_df["system"] == "oneclick_undo_only"
    ].copy()
    if not undo_only.empty:
        undo_only = undo_only.rename(
            columns={
                column: f"{column}_oneclick_undo_only"
                for column in undo_only.columns
                if column not in key_columns
            }
        )
        merged = merged.merge(undo_only, on=key_columns, how="outer")
        merged["click_reduction_undo_only"] = (
            merged["num_clicks_og"] - merged["num_clicks_oneclick_undo_only"]
        )
        merged["click_reduction_percent_undo_only"] = np.where(
            merged["num_clicks_og"] > 0,
            merged["click_reduction_undo_only"] / merged["num_clicks_og"] * 100,
            np.nan,
        )
        merged["correction_rate_difference_undo_only"] = (
            merged["correction_rate_percent_oneclick_undo_only"]
            - merged["correction_rate_percent_og"]
        )
    return merged


def build_trial_summary(system_results_df: pd.DataFrame) -> pd.DataFrame:
    def mean_or_nan(df: pd.DataFrame, column: str) -> float:
        if column not in df:
            return float("nan")
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        return float(values.mean()) if not values.empty else float("nan")

    def sum_or_nan(df: pd.DataFrame, column: str) -> float:
        if column not in df:
            return float("nan")
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        return float(values.sum()) if not values.empty else float("nan")

    rows = []
    for keys, group in system_results_df.groupby(["user_id", "trial", "system"], dropna=False):
        user_id, trial, system = keys
        rows.append(
            {
                "user_id": user_id,
                "trial": int(trial),
                "system": system,
                "phrase_count": int(len(group)),
                "mean_clicks_per_phrase": float(group["num_clicks"].mean()),
                "median_clicks_per_phrase": float(group["num_clicks"].median()),
                "mean_correction_rate_percent": float(group["correction_rate_percent"].mean()),
                "phrase_completion_rate": float(group["phrase_completed"].mean()),
                "mean_word_completion_rate_percent": mean_or_nan(
                    group, "word_completion_rate_percent"
                ),
                "failed_word_count": sum_or_nan(group, "failed_word_count"),
                "word_attempt_count": sum_or_nan(group, "word_attempt_count"),
                "enter_retry_count": sum_or_nan(group, "enter_retry_count"),
                "wrong_word_commit_count": sum_or_nan(group, "wrong_word_commit_count"),
                "undo_attempt_count": sum_or_nan(group, "undo_attempt_count"),
                "undo_failure_count": sum_or_nan(group, "undo_failure_count"),
                "restored_state_count": sum_or_nan(group, "restored_state_count"),
            }
        )
    return pd.DataFrame(rows)


def build_comparison_summary(phrase_trial_df: pd.DataFrame) -> pd.DataFrame:
    def mean_or_nan(df: pd.DataFrame, column: str) -> float:
        if column not in df:
            return float("nan")
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        return float(values.mean()) if not values.empty else float("nan")

    def sum_or_nan(df: pd.DataFrame, column: str) -> float:
        if column not in df:
            return float("nan")
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        return float(values.sum()) if not values.empty else float("nan")

    def mutually_completed_metrics(df: pd.DataFrame, suffix: str) -> dict[str, float]:
        phrase_column = f"phrase_completed_{suffix}"
        click_column = f"num_clicks_{suffix}"
        if phrase_column not in df or click_column not in df:
            return {
                "count": 0,
                "og_mean_clicks": float("nan"),
                "oneclick_mean_clicks": float("nan"),
                "click_reduction_percent": float("nan"),
            }
        mask = (
            df["phrase_completed_og"].fillna(False).astype(bool)
            & df[phrase_column].fillna(False).astype(bool)
        )
        paired = df.loc[mask]
        if paired.empty:
            return {
                "count": 0,
                "og_mean_clicks": float("nan"),
                "oneclick_mean_clicks": float("nan"),
                "click_reduction_percent": float("nan"),
            }
        og_clicks = float(paired["num_clicks_og"].mean())
        oneclick_clicks = float(paired[click_column].mean())
        reduction = (
            (og_clicks - oneclick_clicks) / og_clicks * 100
            if og_clicks > 0
            else float("nan")
        )
        return {
            "count": int(len(paired)),
            "og_mean_clicks": og_clicks,
            "oneclick_mean_clicks": oneclick_clicks,
            "click_reduction_percent": reduction,
        }

    rows = []
    for user_id, group in phrase_trial_df.groupby("user_id", sort=True):
        protected_paired = mutually_completed_metrics(group, "oneclick")
        undo_only_paired = mutually_completed_metrics(group, "oneclick_undo_only")
        og_mean_clicks = float(group["num_clicks_og"].mean())
        oneclick_mean_clicks = float(group["num_clicks_oneclick"].mean())
        mean_click_reduction = og_mean_clicks - oneclick_mean_clicks
        mean_click_reduction_percent = (
            mean_click_reduction / og_mean_clicks * 100 if og_mean_clicks > 0 else float("nan")
        )
        rows.append(
            {
                "user_id": user_id,
                "usable_phrase_trials": int(len(group)),
                "og_mean_clicks_per_phrase": og_mean_clicks,
                "oneclick_mean_clicks_per_phrase": oneclick_mean_clicks,
                "og_median_clicks_per_phrase": float(group["num_clicks_og"].median()),
                "oneclick_median_clicks_per_phrase": float(group["num_clicks_oneclick"].median()),
                "mean_click_reduction": mean_click_reduction,
                "median_click_reduction": float(group["click_reduction"].median()),
                "mean_click_reduction_percent": mean_click_reduction_percent,
                "mean_phrase_click_reduction_percent": float(
                    group["click_reduction_percent"].mean()
                ),
                "og_mean_correction_rate_percent": float(group["correction_rate_percent_og"].mean()),
                "oneclick_mean_correction_rate_percent": float(
                    group["correction_rate_percent_oneclick"].mean()
                ),
                "mean_correction_rate_difference": float(
                    group["correction_rate_difference"].mean()
                ),
                "og_phrase_completion_rate": float(group["phrase_completed_og"].mean()),
                "oneclick_phrase_completion_rate": float(group["phrase_completed_oneclick"].mean()),
                "oneclick_mean_word_completion_rate_percent": mean_or_nan(
                    group, "word_completion_rate_percent_oneclick"
                ),
                "oneclick_failed_word_count": sum_or_nan(group, "failed_word_count_oneclick"),
                "oneclick_word_attempt_count": sum_or_nan(group, "word_attempt_count_oneclick"),
                "oneclick_enter_retry_count": sum_or_nan(group, "enter_retry_count_oneclick"),
                "oneclick_wrong_word_commit_count": sum_or_nan(
                    group, "wrong_word_commit_count_oneclick"
                ),
                "oneclick_undo_attempt_count": sum_or_nan(group, "undo_attempt_count_oneclick"),
                "oneclick_undo_failure_count": sum_or_nan(group, "undo_failure_count_oneclick"),
                "oneclick_restored_state_count": sum_or_nan(
                    group, "restored_state_count_oneclick"
                ),
                "mutually_completed_phrase_trials": protected_paired["count"],
                "mutually_completed_og_mean_clicks": protected_paired["og_mean_clicks"],
                "mutually_completed_oneclick_mean_clicks": protected_paired[
                    "oneclick_mean_clicks"
                ],
                "mutually_completed_click_reduction_percent": protected_paired[
                    "click_reduction_percent"
                ],
                "oneclick_undo_only_mean_clicks_per_phrase": mean_or_nan(
                    group, "num_clicks_oneclick_undo_only"
                ),
                "oneclick_undo_only_phrase_completion_rate": mean_or_nan(
                    group, "phrase_completed_oneclick_undo_only"
                ),
                "oneclick_undo_only_mean_correction_rate_percent": mean_or_nan(
                    group, "correction_rate_percent_oneclick_undo_only"
                ),
                "undo_only_mean_correction_rate_difference": (
                    mean_or_nan(group, "correction_rate_percent_oneclick_undo_only")
                    - float(group["correction_rate_percent_og"].mean())
                ),
                "oneclick_undo_only_mean_word_completion_rate_percent": mean_or_nan(
                    group, "word_completion_rate_percent_oneclick_undo_only"
                ),
                "oneclick_undo_only_failed_word_count": sum_or_nan(
                    group, "failed_word_count_oneclick_undo_only"
                ),
                "oneclick_undo_only_enter_retry_count": sum_or_nan(
                    group, "enter_retry_count_oneclick_undo_only"
                ),
                "oneclick_undo_only_wrong_word_commit_count": sum_or_nan(
                    group, "wrong_word_commit_count_oneclick_undo_only"
                ),
                "oneclick_undo_only_undo_attempt_count": sum_or_nan(
                    group, "undo_attempt_count_oneclick_undo_only"
                ),
                "oneclick_undo_only_undo_failure_count": sum_or_nan(
                    group, "undo_failure_count_oneclick_undo_only"
                ),
                "oneclick_undo_only_restored_state_count": sum_or_nan(
                    group, "restored_state_count_oneclick_undo_only"
                ),
                "undo_only_mutually_completed_phrase_trials": undo_only_paired["count"],
                "undo_only_mutually_completed_og_mean_clicks": undo_only_paired[
                    "og_mean_clicks"
                ],
                "undo_only_mutually_completed_oneclick_mean_clicks": undo_only_paired[
                    "oneclick_mean_clicks"
                ],
                "undo_only_mutually_completed_click_reduction_percent": undo_only_paired[
                    "click_reduction_percent"
                ],
            }
        )
    return pd.DataFrame(rows)


def plot_summary(summary_df: pd.DataFrame, output_dir: Path) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    users = summary_df["user_id"].astype(str).to_numpy()
    x = np.arange(len(users))

    def save_bar_plot(filename: str, title: str, ylabel: str, values: list[tuple[str, np.ndarray]]) -> None:
        figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
        width = 0.36 if len(values) > 1 else 0.6
        offsets = (
            np.linspace(-width / 2, width / 2, len(values))
            if len(values) > 1
            else np.array([0.0])
        )
        for offset, (label, series) in zip(offsets, values):
            axis.bar(x + offset, series, width=width, label=label)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xticks(x)
        axis.set_xticklabels(users)
        axis.grid(axis="y", alpha=0.2)
        if len(values) > 1:
            axis.legend()
        figure.savefig(plot_dir / filename, dpi=160, facecolor="white")
        plt.close(figure)

    has_undo_only = summary_df["oneclick_undo_only_phrase_completion_rate"].notna().any()
    completion_series = [
        ("OG Nomon", summary_df["og_phrase_completion_rate"].to_numpy()),
        ("OneClick", summary_df["oneclick_phrase_completion_rate"].to_numpy()),
    ]
    correction_series = [
        ("OG Nomon", summary_df["og_mean_correction_rate_percent"].to_numpy()),
        ("OneClick", summary_df["oneclick_mean_correction_rate_percent"].to_numpy()),
    ]
    reduction_series = [
        (
            "OneClick",
            summary_df["mutually_completed_click_reduction_percent"].to_numpy(),
        )
    ]
    if has_undo_only:
        completion_series.append(
            (
                "Undo-only",
                summary_df["oneclick_undo_only_phrase_completion_rate"].to_numpy(),
            )
        )
        correction_series.append(
            (
                "Undo-only",
                summary_df[
                    "oneclick_undo_only_mean_correction_rate_percent"
                ].to_numpy(),
            )
        )
        reduction_series.append(
            (
                "Undo-only",
                summary_df[
                    "undo_only_mutually_completed_click_reduction_percent"
                ].to_numpy(),
            )
        )

    save_bar_plot(
        "phrase_completion_by_user.png",
        "Phrase completion rate",
        "Completion rate",
        completion_series,
    )
    save_bar_plot(
        "correction_rate_by_user.png",
        "Mean correction rate",
        "Correction rate (%)",
        correction_series,
    )
    save_bar_plot(
        "click_reduction_by_user.png",
        "Mutually completed phrase click reduction vs OG Nomon",
        "Click reduction (%)",
        reduction_series,
    )


def write_markdown_report(
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

    lines = ["# OG Nomon vs OneClick Gaussian Comparison", ""]
    lines.append("## Headline Summary")
    lines.append("")
    headline_df = summary_df[
        [
            "user_id",
            "mutually_completed_phrase_trials",
            "mutually_completed_og_mean_clicks",
            "mutually_completed_oneclick_mean_clicks",
            "mutually_completed_click_reduction_percent",
            "oneclick_phrase_completion_rate",
            "undo_only_mutually_completed_phrase_trials",
            "undo_only_mutually_completed_og_mean_clicks",
            "undo_only_mutually_completed_oneclick_mean_clicks",
            "undo_only_mutually_completed_click_reduction_percent",
            "oneclick_undo_only_phrase_completion_rate",
        ]
    ].rename(
        columns={
            "mutually_completed_phrase_trials": "protected_paired_phrases",
            "mutually_completed_og_mean_clicks": "protected_og_clicks",
            "mutually_completed_oneclick_mean_clicks": "protected_oneclick_clicks",
            "mutually_completed_click_reduction_percent": "protected_click_reduction_percent",
            "oneclick_phrase_completion_rate": "protected_phrase_completion_rate",
            "undo_only_mutually_completed_phrase_trials": "undo_only_paired_phrases",
            "undo_only_mutually_completed_og_mean_clicks": "undo_only_og_clicks",
            "undo_only_mutually_completed_oneclick_mean_clicks": "undo_only_oneclick_clicks",
            "undo_only_mutually_completed_click_reduction_percent": "undo_only_click_reduction_percent",
            "oneclick_undo_only_phrase_completion_rate": "undo_only_phrase_completion_rate",
        }
    )
    lines.append(
        markdown_table(headline_df)
    )
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    protected_reduction = summary_df["mutually_completed_click_reduction_percent"].mean()
    undo_only_reduction = summary_df[
        "undo_only_mutually_completed_click_reduction_percent"
    ].mean()
    protected_label = (
        "n/a" if pd.isna(protected_reduction) else f"{protected_reduction:.1f}%"
    )
    undo_only_label = (
        "n/a" if pd.isna(undo_only_reduction) else f"{undo_only_reduction:.1f}%"
    )
    lines.append(
        "Average mutually-completed click reduction: "
        f"protected {protected_label}, undo-only {undo_only_label}."
    )
    lines.append(
        "Average correction-rate change: protected "
        f"{summary_df['mean_correction_rate_difference'].mean():+.1f} points, "
        "undo-only "
        f"{summary_df['undo_only_mean_correction_rate_difference'].mean():+.1f} points."
    )
    lines.append("")
    lines.append(
        "Headline click reductions use only phrase trials completed by both OG Nomon "
        "and the corresponding OneClick variant."
    )
    lines.append(
        "Protected Undo keeps prediction clocks active during correction but suppresses "
        "non-Undo commits; Undo-only temporarily makes Undo the sole active word clock."
    )
    lines.append("")
    lines.append("## User Timing Profiles")
    lines.append("")
    for row in profile_df.sort_values("user_id").to_dict("records"):
        lines.append(f"- User {row['user_id']}: {row['qualitative_description']}")
    lines.append("")
    (output_dir / "comparison_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--users", default=",".join(DEFAULT_USERS))
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--max-phrases", type=int, default=None)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--clicks-per-phrase", type=int, default=500)
    parser.add_argument("--calibration-clicks", type=int, default=200)
    parser.add_argument("--oneclick-max-word-attempts", type=int, default=5)
    parser.add_argument("--oneclick-max-enter-attempts", type=int, default=5)
    parser.add_argument("--oneclick-max-clicks-per-word", type=int, default=30)
    parser.add_argument(
        "--perfect-clicks",
        action="store_true",
        help="Use zero click offsets instead of Gaussian sampled offsets.",
    )
    parser.add_argument(
        "--oneclick-perfect-letter-observations",
        action="store_true",
        help="Replace OneClick letter observations with near-certain intended letters.",
    )
    parser.add_argument(
        "--old-nomon-perfect-selections",
        action="store_true",
        help="Diagnostic mode: force OG Nomon to select the intended target clock.",
    )
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--phrase-policy", choices=["reachable", "all"], default="reachable")
    parser.add_argument("--lm-backend", choices=["kenlm", "imagineville"], default="imagineville")
    parser.add_argument("--lm-size", choices=["tiny", "medium"], default="tiny")
    parser.add_argument("--lm-cache-dir", type=Path, default=DEFAULT_LM_CACHE_DIR)
    parser.add_argument(
        "--oneclick-cache-dir",
        type=Path,
        default=REPO_ROOT / ".cache" / "oneclick_phrase_audit",
    )
    parser.add_argument("--std-sweep-dir", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
    )
    return parser.parse_args()


def run_comparison(args: argparse.Namespace) -> Path:
    user_ids = parse_csv_values(args.users)
    if args.trials < 1:
        raise ValueError("--trials must be at least 1")
    if args.clicks_per_phrase < 1:
        raise ValueError("--clicks-per-phrase must be at least 1")

    std_sweep_dir = args.std_sweep_dir or find_latest_std_sweep(args.output_dir)
    selected_std_df = load_selected_std_multipliers(std_sweep_dir, user_ids)
    selected_by_user = selected_std_df.set_index("user_id")
    output_dir = build_output_dir(args.output_dir)

    all_phrase_rows = []
    all_audit_rows = []
    all_profiles = []
    all_system_results = []

    for user_id in user_ids:
        real_click_df = clean_click_rows(load_text_click_data(user_id))
        real_phrase_df = normalize_phrase_order(load_text_phrase_data(user_id))
        train_sessions, validation_sessions = split_sessions(
            real_click_df,
            args.validation_fraction,
        )
        train_click_df = real_click_df[real_click_df["Session Num"].isin(train_sessions)].copy()
        validation_click_df = real_click_df[
            real_click_df["Session Num"].isin(validation_sessions)
        ].copy()
        validation_phrase_df = real_phrase_df[
            real_phrase_df["Session Num"].isin(validation_sessions)
        ].copy()

        regimes_df = build_clock_regimes(train_click_df)
        assigned_train_df = assign_clock_regimes(train_click_df, regimes_df)
        regime_stats_df = build_regime_statistics(assigned_train_df, regimes_df)
        selected_row = selected_by_user.loc[user_id]
        selected_multiplier = float(selected_row["selected_std_multiplier"])
        all_profiles.append(timing_profile_row(user_id, train_click_df, regime_stats_df, selected_row))

        phrase_df, audit_df = build_comparison_phrase_df(
            user_id=user_id,
            validation_phrase_df=validation_phrase_df,
            validation_click_df=validation_click_df,
            regimes_df=regimes_df,
            phrase_policy=args.phrase_policy,
            oneclick_cache_dir=args.oneclick_cache_dir,
            max_phrases=args.max_phrases,
        )
        phrase_df.insert(0, "user_id", user_id)
        all_phrase_rows.append(phrase_df)
        if not audit_df.empty:
            audit_df.insert(0, "user_id", user_id)
            all_audit_rows.append(audit_df)

        for trial in range(args.trials):
            print(f"User {user_id} trial {trial + 1}/{args.trials}: running OG Nomon")
            rng = np.random.default_rng(args.seed + trial + sum(user_id.encode("utf-8")))
            click_df = build_shared_synthetic_click_df(
                phrase_df=phrase_df,
                regime_stats_df=regime_stats_df,
                selected_std_multiplier=selected_multiplier,
                clicks_per_phrase=args.clicks_per_phrase,
                calibration_clicks=args.calibration_clicks,
                rng=rng,
                perfect_clicks=args.perfect_clicks,
            )
            old_results = run_old_nomon(
                click_df=click_df.copy(),
                phrase_df=phrase_df,
                lm_backend=args.lm_backend,
                lm_size=args.lm_size,
                lm_cache_dir=args.lm_cache_dir,
                force_target_clock_selection=args.old_nomon_perfect_selections,
                verbose=args.verbose,
            )
            all_system_results.append(
                normalize_system_results(old_results, "og_nomon", user_id, trial)
            )
            for undo_mode, system_name in [
                ("protected", "oneclick"),
                ("undo_only", "oneclick_undo_only"),
            ]:
                print(
                    f"User {user_id} trial {trial + 1}/{args.trials}: "
                    f"running OneClick ({undo_mode})"
                )
                oneclick_results = run_oneclick(
                    click_df=click_df.copy(),
                    phrase_df=phrase_df,
                    max_word_attempts=args.oneclick_max_word_attempts,
                    max_enter_attempts=args.oneclick_max_enter_attempts,
                    max_clicks_per_word=args.oneclick_max_clicks_per_word,
                    undo_mode=undo_mode,
                    oneclick_cache_dir=args.oneclick_cache_dir,
                    perfect_letter_observations=args.oneclick_perfect_letter_observations,
                    verbose=args.verbose,
                )
                all_system_results.append(
                    normalize_system_results(oneclick_results, system_name, user_id, trial)
                )

    phrase_set_df = pd.concat(all_phrase_rows, ignore_index=True)
    profile_df = pd.DataFrame(all_profiles)
    system_results_df = pd.concat(all_system_results, ignore_index=True)
    phrase_trial_df = build_phrase_trial_comparison(system_results_df)
    trial_summary_df = build_trial_summary(system_results_df)
    comparison_summary_df = build_comparison_summary(phrase_trial_df)
    comparison_summary_df = comparison_summary_df.merge(
        profile_df[
            [
                "user_id",
                "selected_std_multiplier",
                "confidence",
                "clock_pace",
                "timing_variability",
                "click_offset_bias",
                "qualitative_description",
            ]
        ],
        on="user_id",
        how="left",
    )

    phrase_set_df.to_csv(output_dir / "phrase_set.csv", index=False)
    selected_std_df.to_csv(output_dir / "selected_std_multipliers_used.csv", index=False)
    profile_df.to_csv(output_dir / "user_timing_profiles.csv", index=False)
    system_results_df.to_csv(output_dir / "system_phrase_results.csv", index=False)
    phrase_trial_df.to_csv(output_dir / "phrase_trial_results.csv", index=False)
    trial_summary_df.to_csv(output_dir / "trial_summary.csv", index=False)
    comparison_summary_df.to_csv(output_dir / "comparison_summary.csv", index=False)
    if all_audit_rows:
        pd.concat(all_audit_rows, ignore_index=True).to_csv(
            output_dir / "oneclick_phrase_reachability.csv",
            index=False,
        )
    plot_summary(comparison_summary_df, output_dir)
    write_markdown_report(output_dir, comparison_summary_df, profile_df)

    write_json(
        output_dir / "run_config.json",
        {
            "users": user_ids,
            "trials": args.trials,
            "max_phrases": args.max_phrases,
            "validation_fraction": args.validation_fraction,
            "clicks_per_phrase": args.clicks_per_phrase,
            "calibration_clicks": args.calibration_clicks,
            "oneclick_max_word_attempts": args.oneclick_max_word_attempts,
            "oneclick_max_enter_attempts": args.oneclick_max_enter_attempts,
            "oneclick_max_clicks_per_word": args.oneclick_max_clicks_per_word,
            "oneclick_stop_phrase_on_failed_word": True,
            "oneclick_undo_modes": ["protected", "undo_only"],
            "perfect_clicks": bool(args.perfect_clicks),
            "oneclick_perfect_letter_observations": bool(
                args.oneclick_perfect_letter_observations
            ),
            "old_nomon_perfect_selections": bool(args.old_nomon_perfect_selections),
            "seed": args.seed,
            "phrase_policy": args.phrase_policy,
            "std_sweep_dir": str(std_sweep_dir.resolve()),
            "lm_backend": args.lm_backend,
            "lm_size": args.lm_size,
            "lm_cache_dir": str(args.lm_cache_dir.resolve()),
            "oneclick_cache_dir": str(args.oneclick_cache_dir.resolve()),
            "dead_time_mode": "zero",
            "old_nomon_append_terminal_periods": False,
            "outputs": [
                "phrase_set.csv",
                "selected_std_multipliers_used.csv",
                "user_timing_profiles.csv",
                "system_phrase_results.csv",
                "phrase_trial_results.csv",
                "trial_summary.csv",
                "comparison_summary.csv",
                "comparison_report.md",
                "plots/",
            ],
        },
    )

    print(f"Saved OG Nomon vs OneClick comparison to: {output_dir}")
    print(
        comparison_summary_df[
            [
                "user_id",
                "mutually_completed_phrase_trials",
                "mutually_completed_click_reduction_percent",
                "oneclick_phrase_completion_rate",
                "undo_only_mutually_completed_phrase_trials",
                "undo_only_mutually_completed_click_reduction_percent",
                "oneclick_undo_only_phrase_completion_rate",
            ]
        ].to_string(index=False)
    )
    return output_dir


def main() -> None:
    run_comparison(parse_args())


if __name__ == "__main__":
    main()
