"""Summary metrics for real and simulated Nomon text-entry runs.

The first evaluator intentionally starts with coarse metrics. These are enough
to expose whether the current replay-style simulator behaves like the original
user at the phrase/session level before introducing richer user models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MetricComparison:
    metric: str
    real_value: float | None
    simulated_value: float | None

    @property
    def absolute_error(self) -> float | None:
        if self.real_value is None or self.simulated_value is None:
            return None
        return self.simulated_value - self.real_value

    @property
    def relative_error(self) -> float | None:
        if self.real_value in (None, 0) or self.simulated_value is None:
            return None
        return (self.simulated_value - self.real_value) / self.real_value


def _clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").dropna()


def _mean(df: pd.DataFrame, column: str) -> float | None:
    if column not in df:
        return None
    values = _clean_numeric(df[column])
    if values.empty:
        return None
    return float(values.mean())


def _selection_groups(click_df: pd.DataFrame) -> pd.core.groupby.DataFrameGroupBy:
    return click_df.groupby(["Session Num", "Phrase Num", "Selection Num"], dropna=False)


def _align_clicks_to_phrases(click_df: pd.DataFrame, phrase_df: pd.DataFrame) -> pd.DataFrame:
    key_columns = ["Session Num", "Phrase Num"]
    if click_df.empty or phrase_df.empty or not set(key_columns).issubset(phrase_df.columns):
        return click_df
    phrase_keys = phrase_df[key_columns].dropna().drop_duplicates()
    return click_df.merge(phrase_keys, on=key_columns, how="inner")


def _estimate_word_selection_rate(click_df: pd.DataFrame) -> float | None:
    """Estimate word-prediction selections from real click logs.

    The real click CSV does not include an explicit "was word prediction" flag.
    In the logs, selected word completions end with Nomon's space marker "_".
    Requiring more than one character excludes a direct space-key selection
    while retaining one-letter predictions such as ``a_`` and ``i_``.
    """

    if "Selection" not in click_df or click_df.empty:
        return None

    final_rows = _selection_groups(click_df).tail(1)
    selections = final_rows["Selection"].dropna().astype(str)
    if selections.empty:
        return None

    def is_word_selection(selection: str) -> bool:
        return len(selection) > 1 and selection.endswith("_")

    return float(selections.map(is_word_selection).mean() * 100)


def summarize_real_run(click_df: pd.DataFrame, phrase_df: pd.DataFrame) -> dict[str, float | None]:
    """Summarize observed user behavior from the OSF click and phrase logs."""

    aligned_click_df = _align_clicks_to_phrases(click_df, phrase_df)
    selection_sizes = (
        _selection_groups(aligned_click_df).size()
        if not aligned_click_df.empty
        else pd.Series(dtype=float)
    )

    summary: dict[str, float | None] = {
        "num_sessions": (
            float(aligned_click_df["Session Num"].nunique())
            if "Session Num" in aligned_click_df
            else None
        ),
        "num_phrases": float(len(phrase_df)),
        "num_clicks": float(len(aligned_click_df)),
        "num_selections": float(len(selection_sizes)),
        "clicks_per_selection": float(selection_sizes.mean()) if not selection_sizes.empty else None,
        "click_time_relative_mean_s": _mean(aligned_click_df, "Click Time Relative (s)"),
        "click_time_relative_sd_s": None,
        "dead_time_mean_s": _mean(aligned_click_df, "Dead Time (s)"),
        "clock_period_mean_s": _mean(aligned_click_df, "Clock Period (s)"),
        "click_load_clicks_per_character": _mean(phrase_df, "Click Load (clicks/character)"),
        "entry_rate_wpm": _mean(phrase_df, "Entry Rate (wpm)"),
        "correction_rate_percent": _mean(phrase_df, "Correction Rate (% of selections)"),
        "error_rate_percent": _mean(phrase_df, "Final Error Rate (%)"),
        "word_prediction_usage_percent_estimated": _estimate_word_selection_rate(aligned_click_df),
    }

    if "Click Time Relative (s)" in aligned_click_df:
        click_offsets = _clean_numeric(aligned_click_df["Click Time Relative (s)"])
    else:
        click_offsets = pd.Series(dtype=float)
    if not click_offsets.empty:
        summary["click_time_relative_sd_s"] = float(click_offsets.std(ddof=0))

    return summary


def summarize_simulated_run(result_df: pd.DataFrame, clicks_used: int | None = None) -> dict[str, float | None]:
    """Summarize the simulator's phrase-level output."""

    if {"Num Selections", "Num Word Prediction Selections"}.issubset(result_df.columns):
        selection_counts = _clean_numeric(result_df["Num Selections"])
        word_selection_counts = _clean_numeric(result_df["Num Word Prediction Selections"])
        num_selections = float(selection_counts.sum()) if not selection_counts.empty else None
        word_prediction_usage = (
            float(word_selection_counts.sum() / num_selections * 100)
            if num_selections
            else None
        )
    else:
        # Compatibility for result files generated before raw counts were saved.
        num_selections = None
        word_prediction_usage = _mean(result_df, "Word Prediction Usage (%)")

    summary: dict[str, float | None] = {
        "num_sessions": float(result_df["Session Num"].nunique()) if "Session Num" in result_df else None,
        "num_phrases": float(len(result_df)),
        "num_clicks": float(clicks_used) if clicks_used is not None else None,
        "num_selections": num_selections,
        "clicks_per_selection": _mean(result_df, "Click Load (clicks/selection)"),
        "click_time_relative_mean_s": None,
        "click_time_relative_sd_s": None,
        "dead_time_mean_s": None,
        "clock_period_mean_s": None,
        "click_load_clicks_per_character": _mean(result_df, "Click Load (clicks/character)"),
        "entry_rate_wpm": _mean(result_df, "Entry Rate (wpm)"),
        "correction_rate_percent": _mean(result_df, "Correction Rate (%)"),
        "error_rate_percent": _mean(result_df, "Error Rate (%)"),
        "word_prediction_usage_percent_estimated": word_prediction_usage,
    }

    return summary


def compare_summaries(
    real_summary: Mapping[str, Any],
    simulated_summary: Mapping[str, Any],
) -> pd.DataFrame:
    """Return a long-form comparison table for matching summary keys."""

    metric_names = sorted(set(real_summary) | set(simulated_summary))
    rows = []
    for metric in metric_names:
        comparison = MetricComparison(
            metric=metric,
            real_value=real_summary.get(metric),
            simulated_value=simulated_summary.get(metric),
        )
        rows.append(
            {
                "metric": comparison.metric,
                "real_value": comparison.real_value,
                "simulated_value": comparison.simulated_value,
                "absolute_error": comparison.absolute_error,
                "relative_error": comparison.relative_error,
            }
        )

    return pd.DataFrame(rows)
