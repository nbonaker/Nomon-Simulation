"""Synthetic user profile estimation and click generation for Nomon text runs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from User_Simulation.evaluation.metrics import summarize_real_run


CLICK_OFFSET_COL = "Click Time Relative (s)"
CLICK_OFFSET_ALIAS_COL = "Click Time Rlative (s)"
DEAD_TIME_COL = "Dead Time (s)"
CLOCK_PERIOD_COL = "Clock Period (s)"


def _clean_numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[column], errors="coerce").dropna()


def _mean(values: pd.Series, default: float = 0.0) -> float:
    if values.empty:
        return default
    return float(values.mean())


def _std(values: pd.Series) -> float:
    if len(values) <= 1:
        return 0.0
    value = float(values.std(ddof=0))
    if np.isnan(value):
        return 0.0
    return value


def _quantile(values: pd.Series, q: float, default: float = 0.0) -> float:
    if values.empty:
        return default
    return float(values.quantile(q))


def split_sessions(
    click_df: pd.DataFrame,
    validation_fraction: float = 0.2,
) -> tuple[list[int], list[int]]:
    """Return chronological train/validation session ids."""

    sessions = sorted(int(session) for session in click_df["Session Num"].dropna().unique())
    if len(sessions) < 2:
        raise ValueError("Synthetic validation requires at least two sessions")

    holdout_count = max(1, int(np.ceil(len(sessions) * validation_fraction)))
    holdout_count = min(holdout_count, len(sessions) - 1)
    return sessions[:-holdout_count], sessions[-holdout_count:]


def select_sessions(
    click_df: pd.DataFrame,
    phrase_df: pd.DataFrame,
    sessions: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    session_set = set(sessions)
    return (
        click_df[click_df["Session Num"].isin(session_set)].copy(),
        phrase_df[phrase_df["Session Num"].isin(session_set)].copy(),
    )


def estimate_profile(
    user_id: str,
    train_click_df: pd.DataFrame,
    train_phrase_df: pd.DataFrame,
    validation_click_df: pd.DataFrame,
    validation_phrase_df: pd.DataFrame,
    train_sessions: list[int],
    validation_sessions: list[int],
) -> dict[str, Any]:
    """Estimate a parametric synthetic user profile from training clicks."""

    click_offsets = _clean_numeric(train_click_df, CLICK_OFFSET_COL)
    dead_times = _clean_numeric(train_click_df, DEAD_TIME_COL)
    clock_periods = _clean_numeric(train_click_df, CLOCK_PERIOD_COL)

    profile: dict[str, Any] = {
        "profile_version": 1,
        "profile_strategy": "one_per_user",
        "source_user_id": user_id,
        "train_sessions": train_sessions,
        "validation_sessions": validation_sessions,
        "train_click_rows": int(len(train_click_df)),
        "train_phrase_rows": int(len(train_phrase_df)),
        "validation_click_rows": int(len(validation_click_df)),
        "validation_phrase_rows": int(len(validation_phrase_df)),
        "click_offset_mean_s": _mean(click_offsets),
        "click_offset_sd_s": _std(click_offsets),
        "click_offset_clip_min_s": _quantile(click_offsets, 0.01),
        "click_offset_clip_max_s": _quantile(click_offsets, 0.99),
        "dead_time_mean_s": _mean(dead_times),
        "dead_time_sd_s": _std(dead_times),
        "dead_time_clip_max_s": _quantile(dead_times, 0.99),
        "clock_period_mean_s": _mean(clock_periods, default=1.0),
        "clock_period_sd_s": _std(clock_periods),
        "clock_period_min_s": float(clock_periods.min()) if not clock_periods.empty else 1.0,
        "clock_period_max_s": float(clock_periods.max()) if not clock_periods.empty else 1.0,
        "train_real_summary": summarize_real_run(train_click_df, train_phrase_df),
        "validation_real_summary": summarize_real_run(validation_click_df, validation_phrase_df),
    }

    if profile["click_offset_clip_min_s"] > profile["click_offset_clip_max_s"]:
        profile["click_offset_clip_min_s"] = profile["click_offset_clip_max_s"]
    if profile["clock_period_min_s"] > profile["clock_period_max_s"]:
        profile["clock_period_min_s"] = profile["clock_period_max_s"]

    return profile


def _sample_normal(
    rng: np.random.Generator,
    mean: float,
    sd: float,
    size: int,
    clip_min: float | None = None,
    clip_max: float | None = None,
) -> np.ndarray:
    if sd <= 0 or np.isnan(sd):
        samples = np.full(size, mean)
    else:
        samples = rng.normal(mean, sd, size)

    if clip_min is not None or clip_max is not None:
        samples = np.clip(samples, clip_min, clip_max)
    return samples


def generate_synthetic_click_df(
    profile: dict[str, Any],
    validation_phrase_df: pd.DataFrame,
    trial: int,
    rng: np.random.Generator,
    clicks_per_phrase: int = 500,
    calibration_clicks: int = 200,
) -> pd.DataFrame:
    """Generate simulator-compatible synthetic click rows for validation phrases."""

    rows: list[dict[str, Any]] = []

    calibration_offsets = _sample_normal(
        rng,
        profile["click_offset_mean_s"],
        profile["click_offset_sd_s"],
        calibration_clicks,
        profile["click_offset_clip_min_s"],
        profile["click_offset_clip_max_s"],
    )
    for click_num, click_offset in enumerate(calibration_offsets, start=1):
        rows.append(
            {
                "Session Num": np.nan,
                "Phrase Num": np.nan,
                "Click Num": click_num,
                CLOCK_PERIOD_COL: profile["clock_period_mean_s"],
                CLICK_OFFSET_COL: click_offset,
                CLICK_OFFSET_ALIAS_COL: click_offset,
                DEAD_TIME_COL: np.nan,
                "Synthetic Trial": trial,
                "Synthetic Source": "calibration",
            }
        )

    sorted_phrases = validation_phrase_df.sort_values(["Session Num", "Phrase Num"])
    for _index, phrase_row in sorted_phrases.iterrows():
        session_num = int(phrase_row["Session Num"])
        phrase_num = int(phrase_row["Phrase Num"])

        click_offsets = _sample_normal(
            rng,
            profile["click_offset_mean_s"],
            profile["click_offset_sd_s"],
            clicks_per_phrase,
            profile["click_offset_clip_min_s"],
            profile["click_offset_clip_max_s"],
        )
        dead_times = _sample_normal(
            rng,
            profile["dead_time_mean_s"],
            profile["dead_time_sd_s"],
            clicks_per_phrase,
            0.0,
            profile["dead_time_clip_max_s"],
        )
        clock_periods = _sample_normal(
            rng,
            profile["clock_period_mean_s"],
            profile["clock_period_sd_s"],
            clicks_per_phrase,
            profile["clock_period_min_s"],
            profile["clock_period_max_s"],
        )

        for click_num in range(clicks_per_phrase):
            click_offset = float(click_offsets[click_num])
            rows.append(
                {
                    "Session Num": session_num,
                    "Phrase Num": phrase_num,
                    "Click Num": click_num + 1,
                    CLOCK_PERIOD_COL: float(clock_periods[click_num]),
                    CLICK_OFFSET_COL: click_offset,
                    CLICK_OFFSET_ALIAS_COL: click_offset,
                    DEAD_TIME_COL: float(dead_times[click_num]),
                    "Synthetic Trial": trial,
                    "Synthetic Source": "validation",
                }
            )

    return pd.DataFrame(rows)
