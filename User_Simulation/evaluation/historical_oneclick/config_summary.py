"""Historical QuickClick data/analysis extracted from OneClick_Simulation.examples.text_simulation.run_config_sweep.

No simulator execution. Saved result semantics are preserved.
"""
from __future__ import annotations

from pathlib import Path

from typing import Mapping, Any, Sequence

import pandas as pd

SIGMA_MARGIN_SWEEP_VALUES = (1.5, 2.0, 2.5, 3.0, 3.5)

ALGORITHM_CONDITION_DESCRIPTIONS = {
    "baseline": "Frozen reference; no algorithm change",
    "enter_offset_compensation": "Enable learned Enter-offset compensation only",
    "separate_space_enter_models": "Use independent Space and Enter timing models only",
    "adaptive_word_clocks": (
        "Enable adaptive word-clock capacity with BEST-first alternating priority"
    ),
    "combined_offset_separate_models": (
        "Enable learned Enter-offset compensation with independent Space and "
        "Enter timing models"
    ),
    "dynamic_character_rephasing": (
        "Rebuild character-clock phases after every Space observation"
    ),
    **{
        f"adaptive_sigma_{margin:.1f}".replace(".", "_"): (
            "Adaptive BEST-first word clocks with "
            f"sigma margin k={margin:.1f}"
        )
        for margin in SIGMA_MARGIN_SWEEP_VALUES
    },
}

ALGORITHM_CONDITION_LABELS = {
    "baseline": "Baseline",
    "enter_offset_compensation": "Enter offset",
    "separate_space_enter_models": "Separate models",
    "adaptive_word_clocks": "Adaptive clocks",
    "combined_offset_separate_models": "Offset + separate models",
    "dynamic_character_rephasing": "Dynamic character rephasing",
    **{
        f"adaptive_sigma_{margin:.1f}".replace(".", "_"): (
            f"Adaptive k={margin:.1f}"
        )
        for margin in SIGMA_MARGIN_SWEEP_VALUES
    },
}

CONFIG_COLUMNS = [
    "config_id",
    "algorithm_condition",
    "character_clock_mode",
    "clock_period",
    "use_click_offset",
    "delay_learning_mode",
    "word_clock_mode",
    "prediction_priority_mode",
    "sigma_margin",
]

SUMMARY_METRIC_COLUMNS = [
    "Clicks per Character",
    "Active Typing Time per Phrase",
    "Correction Rate",
    "Enter Misselection Rate",
    "Completion Rate",
    "Completed Words",
    "Failed Words",
]

SUMMARY_COLUMNS = [*CONFIG_COLUMNS, "user_id", *SUMMARY_METRIC_COLUMNS]

def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else float("nan")

def summarize_by_user_config(
    phrase_results: pd.DataFrame,
    config_id: str,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Aggregate phrase rows into one deterministic row per config and user."""
    config_values = {"config_id": config_id, **dict(config)}
    rows = []
    for user_id, phrases in phrase_results.groupby("user_id", sort=True):
        attempted_clicks = phrases["Num Clicks"].sum()
        successful_characters = phrases["Successful Word Character Count"].sum()
        completed_words = phrases["Completed Word Count"].sum()
        failed_words = phrases["Failed Word Count"].sum()
        target_words = phrases["Target Word Count"].sum()
        corrections = phrases["Corrective Undo Action Count"].sum()
        enter_misselections = phrases["Enter Misselection Count"].sum()
        enter_presses = phrases["Enter Press Count"].sum()
        rows.append(
            {
                **config_values,
                "user_id": str(user_id),
                "Clicks per Character": _ratio(
                    attempted_clicks, successful_characters
                ),
                "Active Typing Time per Phrase": phrases[
                    "Active Typing Time (s)"
                ].mean(),
                "Correction Rate": _ratio(corrections, completed_words),
                "Enter Misselection Rate": _ratio(
                    enter_misselections, enter_presses
                ),
                "Completion Rate": _ratio(completed_words, target_words),
                "Completed Words": int(completed_words),
                "Failed Words": int(failed_words),
            }
        )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)

def summarize_by_config(user_summary: pd.DataFrame) -> pd.DataFrame:
    """Average real-user summaries and retain synthetic user P separately."""
    if user_summary.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    config_values = user_summary.iloc[0][CONFIG_COLUMNS].to_dict()
    rows = []
    real_users = user_summary[user_summary["user_id"] != "P"]
    if not real_users.empty:
        rows.append(
            {
                **config_values,
                "user_id": "MEAN_REAL_USERS",
                **{
                    column: real_users[column].mean()
                    for column in SUMMARY_METRIC_COLUMNS
                },
            }
        )
    synthetic = user_summary[user_summary["user_id"] == "P"]
    if not synthetic.empty:
        rows.append(synthetic.iloc[0][SUMMARY_COLUMNS].to_dict())
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)

def _report_value(value) -> str:
    if pd.isna(value):
        return "-"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)

def _markdown_table(columns: Sequence[str], rows: Sequence[Sequence]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        values = [
            _report_value(value).replace("|", "\\|").replace("\n", " ")
            for value in row
        ]
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *body])

_REPORT_CONFIG_COLUMNS = [
    "config_id",
    "algorithm_condition",
    "clock_period",
    "use_click_offset",
    "delay_learning_mode",
    "word_clock_mode",
    "prediction_priority_mode",
    "sigma_margin",
]

_REPORT_RESULT_COLUMNS = [
    "Completion Rate",
    "Clicks per Character",
    "Active Typing Time per Phrase",
    "Correction Rate",
    "Enter Misselection Rate",
    "Completed Words",
    "Failed Words",
]

def _completion_first(frame: pd.DataFrame) -> pd.DataFrame:
    """Rank transparently, prioritizing successful completion."""
    if frame.empty:
        return frame.copy()
    return frame.sort_values(
        [
            "Completion Rate",
            "Failed Words",
            "Clicks per Character",
            "Active Typing Time per Phrase",
            "Enter Misselection Rate",
            "Correction Rate",
            "config_id",
        ],
        ascending=[False, True, True, True, True, True, True],
        kind="mergesort",
        na_position="last",
    )

def _configuration_rows(frame: pd.DataFrame, limit: int) -> list[list]:
    rows = []
    for rank, (_, row) in enumerate(
        _completion_first(frame).head(limit).iterrows(), start=1
    ):
        rows.append(
            [
                rank,
                *[row[column] for column in _REPORT_CONFIG_COLUMNS],
                *[row[column] for column in _REPORT_RESULT_COLUMNS],
            ]
        )
    return rows

def write_sweep_report(output_root: Path, lm_metadata: dict) -> Path:
    """Generate a deterministic Markdown overview of the algorithm study."""
    manifest = pd.read_csv(output_root / "sweep_config.csv")
    phrase_results = pd.read_csv(output_root / "phrase_results.csv")
    user_summary = pd.read_csv(output_root / "summary_by_user_config.csv")
    config_summary = pd.read_csv(output_root / "summary_by_config.csv")
    corpus_path = output_root / "fixed_iv_phrase_corpus.csv"
    corpus = pd.read_csv(corpus_path) if corpus_path.exists() else pd.DataFrame()
    sufficiency_path = output_root / "click_stream_sufficiency.csv"
    sufficiency = (
        pd.read_csv(sufficiency_path)
        if sufficiency_path.exists()
        else pd.DataFrame()
    )

    real = config_summary[
        config_summary["user_id"].astype(str) == "MEAN_REAL_USERS"
    ].copy()
    synthetic = config_summary[
        config_summary["user_id"].astype(str) == "P"
    ].copy()
    users = sorted(user_summary["user_id"].astype(str).unique())
    real_user_count = sum(user_id != "P" for user_id in users)
    expected_rows = (
        len(manifest) * len(users) * len(corpus) if not corpus.empty else None
    )
    failed_words = (
        int(pd.to_numeric(phrase_results["Failed Word Count"], errors="coerce").sum())
        if "Failed Word Count" in phrase_results
        else 0
    )
    completed_phrases = (
        int(
            phrase_results["Phrase Completed"]
            .astype(str)
            .str.lower()
            .isin({"true", "1", "yes"})
            .sum()
        )
        if "Phrase Completed" in phrase_results
        else None
    )
    exhausted_streams = (
        int(
            sufficiency["Click Stream Exhausted"]
            .astype(str)
            .str.lower()
            .isin({"true", "1", "yes"})
            .sum()
        )
        if "Click Stream Exhausted" in sufficiency
        else 0
    )

    lines = [
        "# QuickClick Algorithm Experiment Report",
        "",
        "Generated deterministically from the paired algorithm-condition outputs.",
        "",
        "## Run completeness",
        "",
        _markdown_table(
            ["Item", "Value"],
            [
                ["Algorithm conditions planned", len(manifest)],
                [
                    "Algorithm conditions represented in summaries",
                    config_summary["config_id"].nunique(),
                ],
                ["Users", ", ".join(users)],
                ["Phrases in fixed corpus", len(corpus) if not corpus.empty else None],
                ["Phrase-result rows", len(phrase_results)],
                ["Expected phrase-result rows", expected_rows],
                ["Completed phrase rows", completed_phrases],
                ["Failed words", failed_words],
                ["Exhausted real-user click streams", exhausted_streams],
            ],
        ),
        "",
        "## Language model",
        "",
        _markdown_table(
            ["Setting", "Value"],
            [
                ["Backend", lm_metadata.get("backend")],
                ["Model source", lm_metadata.get("model_source")],
                ["Network access", lm_metadata.get("network_access")],
                ["Model path", lm_metadata.get("model_path")],
                ["Device", lm_metadata.get("resolved_device")],
                ["Precision", lm_metadata.get("precision")],
                ["TextSlinger version", lm_metadata.get("textslinger_version")],
                ["Recognizer n-best", lm_metadata.get("recognizer_nbest")],
                [
                    "Character cache hits / misses",
                    "{hits} / {misses}".format(
                        **lm_metadata.get("result_cache", {}).get(
                            "character", {"hits": 0, "misses": 0}
                        )
                    ),
                ],
                [
                    "Word cache hits / misses",
                    "{hits} / {misses}".format(
                        **lm_metadata.get("result_cache", {}).get(
                            "word", {"hits": 0, "misses": 0}
                        )
                    ),
                ],
            ],
        ),
        "",
        "## Algorithm conditions",
        "",
        "The clock period, language model, corpus, users, and click schedules "
        "remain fixed across all conditions.",
        "",
        _markdown_table(
            [
                "Config",
                "Algorithm condition",
                "Change from baseline",
                "Offset",
                "Delay mode",
                "Word-clock mode",
                "Prediction priority",
                "Sigma margin",
            ],
            [
                [
                    row["config_id"],
                    row["algorithm_condition"],
                    ALGORITHM_CONDITION_DESCRIPTIONS[row["algorithm_condition"]],
                    row["use_click_offset"],
                    row["delay_learning_mode"],
                    row["word_clock_mode"],
                    row["prediction_priority_mode"],
                    row["sigma_margin"],
                ]
                for _, row in manifest.iterrows()
            ],
        ),
    ]

    table_headers = [
        "Rank",
        *_REPORT_CONFIG_COLUMNS,
        *_REPORT_RESULT_COLUMNS,
    ]
    if not real.empty:
        aggregate_rows = [
            [
                ALGORITHM_CONDITION_LABELS.get(
                    row["algorithm_condition"], row["algorithm_condition"]
                ),
                f'{row["Completion Rate"]:.2%}',
                f'{row["Clicks per Character"]:.3f}',
                f'{row["Active Typing Time per Phrase"]:.2f} s',
                f'{row["Correction Rate"]:.3f}',
                f'{row["Enter Misselection Rate"]:.2%}',
            ]
            for _, row in real.sort_values("config_id", kind="mergesort").iterrows()
        ]
        leader_rows = []
        for label, metric, ascending in [
            ("Highest completion", "Completion Rate", False),
            ("Lowest clicks/character", "Clicks per Character", True),
            ("Lowest active time", "Active Typing Time per Phrase", True),
            ("Lowest correction rate", "Correction Rate", True),
            ("Lowest Enter misselection", "Enter Misselection Rate", True),
        ]:
            eligible = real.dropna(subset=[metric]).sort_values(
                [metric, "config_id"],
                ascending=[ascending, True],
                kind="mergesort",
            )
            if eligible.empty:
                continue
            row = eligible.iloc[0]
            leader_rows.append(
                [
                    label,
                    row["config_id"],
                    row["algorithm_condition"],
                    row[metric],
                    row["Completion Rate"],
                    row["clock_period"],
                    row["use_click_offset"],
                    row["delay_learning_mode"],
                    row["word_clock_mode"],
                    row["sigma_margin"],
                ]
            )

        lines.extend(
            [
                "",
                "## Cumulative real-user results",
                "",
                "Each value is the mean of the per-user aggregate for "
                f"{real_user_count} real user(s). Synthetic user P is excluded.",
                "",
                _markdown_table(
                    [
                        "Condition",
                        "Word completion",
                        "Clicks/character",
                        "Time/phrase",
                        "Correction rate",
                        "Enter misselection",
                    ],
                    aggregate_rows,
                ),
                "",
                "## Real-user metric leaders",
                "",
                "Each metric is evaluated independently; this is not a universal winner.",
                "",
                _markdown_table(
                    [
                        "Metric",
                        "Config",
                        "Algorithm condition",
                        "Metric value",
                        "Completion",
                        "Period",
                        "Offset",
                        "Delay mode",
                        "Word-clock mode",
                        "Sigma margin",
                    ],
                    leader_rows,
                ),
                "",
                "## Algorithm-condition results",
                "",
                "Ordering: highest completion, then fewest failed words, lowest "
                "clicks/character, lowest active time, lowest Enter misselection, "
                "and lowest correction rate.",
                "",
                _markdown_table(
                    table_headers,
                    _configuration_rows(real, 10),
                ),
            ]
        )

    if not synthetic.empty:
        lines.extend(
            [
                "",
                "## Synthetic user P (kept separate)",
                "",
                _markdown_table(
                    table_headers,
                    _configuration_rows(synthetic, 10),
                ),
            ]
        )

    lines.extend(
        [
            "",
            "## Analysis files",
            "",
            "- summary_by_config.csv: configuration-level real-user means and P.",
            "- summary_by_user_config.csv: one row per configuration and user.",
            "- phrase_results.csv: complete phrase-level diagnostics.",
            "- sweep_config.csv: exact named algorithm-condition manifest.",
            "- click_stream_sufficiency.csv: real-user click-stream audit.",
            "- lm_config.json: local TextSlinger reproducibility metadata.",
            "",
        ]
    )
    report_path = output_root / "sweep_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
