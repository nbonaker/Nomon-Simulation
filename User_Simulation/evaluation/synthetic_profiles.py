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
SELECTION_COL = "Selection"
SELECTION_GROUP_COLS = ["Session Num", "Phrase Num", "Selection Num"]
CORRECTION_SELECTIONS = {"@", "#", "$", "Undo", "UNDO", "Undo+", "BACKSPACE", "CLEAR"}
SYNTHETIC_GROUP_ID_COL = "Synthetic Group ID"
SYNTHETIC_GROUP_CLICK_NUM_COL = "Synthetic Group Click Num"
SYNTHETIC_GROUP_SIZE_COL = "Synthetic Group Size"
SYNTHETIC_SELECTION_TYPE_COL = "Synthetic Selection Type"
SYNTHETIC_DEAD_TIME_CLIPPED_COL = "Synthetic Dead Time Clipped"
SELECTION_TYPES = ("character", "word_prediction", "correction")


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


def infer_stable_regime(train_click_df: pd.DataFrame) -> tuple[float, float, list[int]]:
    """Identify the latest contiguous clock-period regime using training data only."""

    session_periods = (
        train_click_df.dropna(subset=["Session Num", CLOCK_PERIOD_COL])
        .groupby("Session Num")[CLOCK_PERIOD_COL]
        .agg(["median", "min", "max"])
        .sort_index()
    )
    if session_periods.empty:
        raise ValueError("Cannot infer a stable regime without training clock periods")

    target_period = float(session_periods.iloc[-1]["median"])
    tolerance = max(0.02, abs(target_period) * 0.02)
    stable_sessions: list[int] = []
    for session_num, period_row in session_periods.iloc[::-1].iterrows():
        if (
            abs(float(period_row["median"]) - target_period) > tolerance
            or abs(float(period_row["min"]) - target_period) > tolerance
            or abs(float(period_row["max"]) - target_period) > tolerance
        ):
            break
        stable_sessions.append(int(session_num))

    stable_sessions.reverse()
    return target_period, tolerance, stable_sessions


def select_bootstrap_source_clicks(
    train_click_df: pd.DataFrame,
    profile: dict[str, Any],
) -> pd.DataFrame:
    source_sessions = profile.get("bootstrap_source_sessions", [])
    if not source_sessions:
        return train_click_df.copy()
    return train_click_df[train_click_df["Session Num"].isin(source_sessions)].copy()


def classify_selection(selection: Any) -> str:
    if selection is None or pd.isna(selection):
        return "character"
    selection_text = str(selection)
    if selection_text in CORRECTION_SELECTIONS:
        return "correction"
    if len(selection_text) > 1 and selection_text.endswith("_"):
        return "word_prediction"
    return "character"


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

    stable_period, stable_tolerance, stable_sessions = infer_stable_regime(train_click_df)
    bootstrap_click_df = train_click_df[
        train_click_df["Session Num"].isin(stable_sessions)
    ].copy()
    click_offsets = _clean_numeric(bootstrap_click_df, CLICK_OFFSET_COL)
    dead_times = _clean_numeric(bootstrap_click_df, DEAD_TIME_COL)
    clock_periods = _clean_numeric(bootstrap_click_df, CLOCK_PERIOD_COL)

    first_group_rows = (
        bootstrap_click_df.sort_values(SELECTION_GROUP_COLS + ["Click Num"])
        .groupby(SELECTION_GROUP_COLS, sort=False, dropna=False)
        .head(1)
    )
    first_dead_times = _clean_numeric(first_group_rows, DEAD_TIME_COL)
    dead_time_cap = _quantile(first_dead_times, 0.99)

    selection_summary = summarize_selection_groups(bootstrap_click_df)

    profile: dict[str, Any] = {
        "profile_version": 3,
        "profile_strategy": "one_per_user",
        "click_clock_sampling_mode": "paired_bootstrap",
        "active_click_sampling_mode": "regime_aware_selection_group_bootstrap",
        "source_user_id": user_id,
        "train_sessions": train_sessions,
        "validation_sessions": validation_sessions,
        "stable_clock_period_s": stable_period,
        "stable_clock_period_tolerance_s": stable_tolerance,
        "bootstrap_source_sessions": stable_sessions,
        "bootstrap_source_click_rows": int(len(bootstrap_click_df)),
        "bootstrap_source_phrase_rows": int(
            train_phrase_df[train_phrase_df["Session Num"].isin(stable_sessions)].shape[0]
        ),
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
        "dead_time_clip_max_s": dead_time_cap,
        "clock_period_mean_s": _mean(clock_periods, default=1.0),
        "clock_period_sd_s": _std(clock_periods),
        "clock_period_min_s": float(clock_periods.min()) if not clock_periods.empty else 1.0,
        "clock_period_max_s": float(clock_periods.max()) if not clock_periods.empty else 1.0,
        "click_clock_bootstrap_rows": int(
            bootstrap_click_df[[CLICK_OFFSET_COL, CLOCK_PERIOD_COL]].dropna().shape[0]
        ),
        **selection_summary,
        "train_real_summary": summarize_real_run(train_click_df, train_phrase_df),
        "bootstrap_real_summary": summarize_real_run(
            bootstrap_click_df,
            train_phrase_df[train_phrase_df["Session Num"].isin(stable_sessions)],
        ),
        "validation_real_summary": summarize_real_run(validation_click_df, validation_phrase_df),
    }

    if profile["click_offset_clip_min_s"] > profile["click_offset_clip_max_s"]:
        profile["click_offset_clip_min_s"] = profile["click_offset_clip_max_s"]
    if profile["clock_period_min_s"] > profile["clock_period_max_s"]:
        profile["clock_period_min_s"] = profile["clock_period_max_s"]

    return profile


def summarize_selection_groups(click_df: pd.DataFrame) -> dict[str, Any]:
    if click_df.empty or not set(SELECTION_GROUP_COLS).issubset(click_df.columns):
        return {
            "train_selection_groups": 0,
            "train_correction_selection_groups": 0,
            "train_correction_selection_rate": 0.0,
            "train_mean_clicks_per_selection_group": 0.0,
            "train_character_selection_groups": 0,
            "train_word_prediction_selection_groups": 0,
        }

    group_rows = []
    for _group_key, group_df in click_df.groupby(SELECTION_GROUP_COLS, dropna=False):
        final_selection = None
        if SELECTION_COL in group_df:
            final_values = group_df[SELECTION_COL].dropna().astype(str)
            if not final_values.empty:
                final_selection = final_values.iloc[-1]
        selection_type = classify_selection(final_selection)
        group_rows.append(
            {
                "num_clicks": int(len(group_df)),
                "selection_type": selection_type,
            }
        )

    if not group_rows:
        return {
            "train_selection_groups": 0,
            "train_correction_selection_groups": 0,
            "train_correction_selection_rate": 0.0,
            "train_mean_clicks_per_selection_group": 0.0,
            "train_character_selection_groups": 0,
            "train_word_prediction_selection_groups": 0,
        }

    group_df = pd.DataFrame(group_rows)
    type_counts = group_df["selection_type"].value_counts()
    correction_groups = int(type_counts.get("correction", 0))
    total_groups = int(len(group_df))
    return {
        "train_selection_groups": total_groups,
        "train_correction_selection_groups": correction_groups,
        "train_correction_selection_rate": float(correction_groups / total_groups),
        "train_mean_clicks_per_selection_group": float(group_df["num_clicks"].mean()),
        "train_character_selection_groups": int(type_counts.get("character", 0)),
        "train_word_prediction_selection_groups": int(type_counts.get("word_prediction", 0)),
    }


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


def _sample_click_clock_pairs(
    rng: np.random.Generator,
    train_click_df: pd.DataFrame | None,
    size: int,
    profile: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    if train_click_df is None:
        click_offsets = _sample_normal(
            rng,
            profile["click_offset_mean_s"],
            profile["click_offset_sd_s"],
            size,
            profile["click_offset_clip_min_s"],
            profile["click_offset_clip_max_s"],
        )
        clock_periods = _sample_normal(
            rng,
            profile["clock_period_mean_s"],
            profile["clock_period_sd_s"],
            size,
            profile["clock_period_min_s"],
            profile["clock_period_max_s"],
        )
        return click_offsets, clock_periods

    valid_pairs = train_click_df[[CLICK_OFFSET_COL, CLOCK_PERIOD_COL]].dropna()
    if valid_pairs.empty:
        return _sample_click_clock_pairs(rng, None, size, profile)

    sampled_indices = rng.integers(0, len(valid_pairs), size=size)
    sampled_pairs = valid_pairs.iloc[sampled_indices]
    return (
        sampled_pairs[CLICK_OFFSET_COL].to_numpy(dtype=float),
        sampled_pairs[CLOCK_PERIOD_COL].to_numpy(dtype=float),
    )


def _build_selection_group_pools(train_click_df: pd.DataFrame | None) -> dict[str, list[pd.DataFrame]]:
    if train_click_df is None or train_click_df.empty:
        return {selection_type: [] for selection_type in SELECTION_TYPES}
    if not set(SELECTION_GROUP_COLS).issubset(train_click_df.columns):
        return {selection_type: [] for selection_type in SELECTION_TYPES}

    required_cols = [
        *SELECTION_GROUP_COLS,
        "Click Num",
        CLICK_OFFSET_COL,
        CLOCK_PERIOD_COL,
        DEAD_TIME_COL,
    ]
    complete_click_df = train_click_df.dropna(subset=required_cols)

    pools = {selection_type: [] for selection_type in SELECTION_TYPES}
    for _group_key, group_df in complete_click_df.groupby(SELECTION_GROUP_COLS, sort=False, dropna=False):
        if group_df.empty:
            continue

        final_selection = None
        if SELECTION_COL in group_df:
            final_values = group_df[SELECTION_COL].dropna().astype(str)
            if not final_values.empty:
                final_selection = final_values.iloc[-1]

        clean_group_df = group_df.sort_values("Click Num")[required_cols].copy()
        pools[classify_selection(final_selection)].append(clean_group_df)

    return pools


def _append_synthetic_group(
    rows: list[dict[str, Any]],
    sampled_group: pd.DataFrame,
    group_id: int,
    selection_type: str,
    session_num: int,
    phrase_num: int,
    trial: int,
    dead_time_cap: float,
) -> int:
    group_size = int(len(sampled_group))
    source_key = sampled_group.iloc[0]
    for group_click_num, (_, source_row) in enumerate(sampled_group.iterrows(), start=1):
        click_offset = float(source_row[CLICK_OFFSET_COL])
        dead_time = float(source_row[DEAD_TIME_COL])
        clipped = group_click_num == 1 and dead_time_cap > 0 and dead_time > dead_time_cap
        if clipped:
            dead_time = dead_time_cap
        rows.append(
            {
                "Session Num": session_num,
                "Phrase Num": phrase_num,
                "Click Num": len(rows) + 1,
                CLOCK_PERIOD_COL: float(source_row[CLOCK_PERIOD_COL]),
                CLICK_OFFSET_COL: click_offset,
                CLICK_OFFSET_ALIAS_COL: click_offset,
                DEAD_TIME_COL: dead_time,
                "Synthetic Trial": trial,
                "Synthetic Source": "validation",
                SYNTHETIC_GROUP_ID_COL: group_id,
                SYNTHETIC_GROUP_CLICK_NUM_COL: group_click_num,
                SYNTHETIC_GROUP_SIZE_COL: group_size,
                SYNTHETIC_SELECTION_TYPE_COL: selection_type,
                SYNTHETIC_DEAD_TIME_CLIPPED_COL: bool(clipped),
                "Synthetic Source Session Num": int(source_key["Session Num"]),
                "Synthetic Source Phrase Num": int(source_key["Phrase Num"]),
                "Synthetic Source Selection Num": int(source_key["Selection Num"]),
            }
        )
    return group_size


def generate_synthetic_click_df(
    profile: dict[str, Any],
    validation_phrase_df: pd.DataFrame,
    trial: int,
    rng: np.random.Generator,
    clicks_per_phrase: int = 500,
    calibration_clicks: int = 200,
    train_click_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Generate simulator-compatible synthetic click rows for validation phrases."""

    rows: list[dict[str, Any]] = []
    bootstrap_click_df = (
        select_bootstrap_source_clicks(train_click_df, profile)
        if train_click_df is not None
        else None
    )

    calibration_offsets, calibration_clock_periods = _sample_click_clock_pairs(
        rng,
        bootstrap_click_df,
        calibration_clicks,
        profile,
    )
    for click_num, click_offset in enumerate(calibration_offsets, start=1):
        rows.append(
            {
                "Session Num": np.nan,
                "Phrase Num": np.nan,
                "Click Num": click_num,
                CLOCK_PERIOD_COL: float(calibration_clock_periods[click_num - 1]),
                CLICK_OFFSET_COL: click_offset,
                CLICK_OFFSET_ALIAS_COL: click_offset,
                DEAD_TIME_COL: np.nan,
                "Synthetic Trial": trial,
                "Synthetic Source": "calibration",
            }
        )

    group_pools = _build_selection_group_pools(bootstrap_click_df)
    available_types = [selection_type for selection_type, pool in group_pools.items() if pool]
    if not available_types:
        raise ValueError("Selection bootstrap requires at least one complete training selection group")

    pool_group_count = sum(len(pool) for pool in group_pools.values())
    pool_types = [
        selection_type
        for selection_type, pool in group_pools.items()
        for _ in range(len(pool))
    ]
    dead_time_cap = float(profile.get("dead_time_clip_max_s", 0.0))
    next_group_id = 1
    sorted_phrases = validation_phrase_df.sort_values(["Session Num", "Phrase Num"])
    for _index, phrase_row in sorted_phrases.iterrows():
        session_num = int(phrase_row["Session Num"])
        phrase_num = int(phrase_row["Phrase Num"])

        phrase_click_count = 0
        # Guarantee a usable pool for each source type, then fill to the row budget
        # according to the empirical source-group type distribution.
        pending_types = available_types.copy()
        while phrase_click_count < clicks_per_phrase or pending_types:
            if pending_types:
                selection_type = pending_types.pop(0)
            else:
                selection_type = pool_types[int(rng.integers(0, pool_group_count))]
            pool = group_pools[selection_type]
            sampled_group = pool[int(rng.integers(0, len(pool)))]
            phrase_click_count += _append_synthetic_group(
                rows,
                sampled_group,
                next_group_id,
                selection_type,
                session_num,
                phrase_num,
                trial,
                dead_time_cap,
            )
            next_group_id += 1

    return pd.DataFrame(rows)
