"""Run reproducible, one-change-at-a-time QuickClick algorithm experiments."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from OneClick_Text import kconfig


BASELINE_CLOCK_PERIOD = 3.6391839582758005
ADAPTIVE_SIGMA_MARGIN = 3.0
SIGMA_MARGIN_SWEEP_VALUES = (1.5, 2.0, 2.5, 3.0, 3.5)
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
BASELINE_NGRAM_MODEL_PATH = (
    REPOSITORY_ROOT / "Nomon_Text" / "resources" / "lm_char_medium.kenlm"
)
BASELINE_VOCABULARY_PATH = (
    REPOSITORY_ROOT / "Nomon_Text" / "resources" / "vocab_lower_100k.txt"
)
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


@dataclass(frozen=True)
class SweepConfig:
    """One named QuickClick algorithm condition."""

    algorithm_condition: str
    clock_period: Optional[float]
    use_click_offset: bool
    delay_learning_mode: str
    word_clock_mode: str
    prediction_priority_mode: str
    sigma_margin: Optional[float]
    character_clock_mode: str = "fixed"

    def simulation_parameters(self) -> dict:
        """Translate this sweep row into parameters accepted by SimulatedUser."""
        parameters = {
            "character_clock_mode": self.character_clock_mode,
            "use_click_offset": self.use_click_offset,
            "delay_learning_mode": self.delay_learning_mode,
            "word_clock_mode": self.word_clock_mode,
            "prediction_priority_mode": self.prediction_priority_mode,
        }
        if self.clock_period is not None:
            parameters["fixed_clock_period_s"] = self.clock_period
        if self.word_clock_mode == "adaptive":
            if self.sigma_margin is None:
                raise ValueError("adaptive word-clock mode requires sigma_margin")
            parameters["sigma_margin"] = self.sigma_margin
        return parameters


def baseline_config() -> SweepConfig:
    """Return the frozen reference condition for every algorithm comparison."""
    return SweepConfig(
        algorithm_condition="baseline",
        clock_period=BASELINE_CLOCK_PERIOD,
        use_click_offset=False,
        delay_learning_mode="enter_only",
        word_clock_mode="fixed",
        prediction_priority_mode="legacy",
        sigma_margin=None,
        character_clock_mode="fixed",
    )


def algorithm_experiment_configs() -> list[SweepConfig]:
    """Return the baseline, isolated changes, their combination, and adaptive clocks."""
    configs = [
        baseline_config(),
        SweepConfig(
            algorithm_condition="enter_offset_compensation",
            clock_period=BASELINE_CLOCK_PERIOD,
            use_click_offset=True,
            delay_learning_mode="enter_only",
            word_clock_mode="fixed",
            prediction_priority_mode="legacy",
            sigma_margin=None,
        ),
        SweepConfig(
            algorithm_condition="separate_space_enter_models",
            clock_period=BASELINE_CLOCK_PERIOD,
            use_click_offset=False,
            delay_learning_mode="separate_space_enter",
            word_clock_mode="fixed",
            prediction_priority_mode="legacy",
            sigma_margin=None,
        ),
        SweepConfig(
            algorithm_condition="adaptive_word_clocks",
            clock_period=BASELINE_CLOCK_PERIOD,
            use_click_offset=False,
            delay_learning_mode="enter_only",
            word_clock_mode="adaptive",
            prediction_priority_mode="alternating",
            sigma_margin=ADAPTIVE_SIGMA_MARGIN,
        ),
        SweepConfig(
            algorithm_condition="combined_offset_separate_models",
            clock_period=BASELINE_CLOCK_PERIOD,
            use_click_offset=True,
            delay_learning_mode="separate_space_enter",
            word_clock_mode="fixed",
            prediction_priority_mode="legacy",
            sigma_margin=None,
        ),
        SweepConfig(
            algorithm_condition="dynamic_character_rephasing",
            clock_period=BASELINE_CLOCK_PERIOD,
            use_click_offset=False,
            delay_learning_mode="enter_only",
            word_clock_mode="fixed",
            prediction_priority_mode="legacy",
            sigma_margin=None,
            character_clock_mode="dynamic",
        ),
    ]
    for config in configs:
        config.simulation_parameters()
    return configs


def dynamic_rephasing_experiment_configs() -> list[SweepConfig]:
    """Return the frozen baseline and isolated dynamic character rephasing."""
    dynamic_config = next(
        config
        for config in algorithm_experiment_configs()
        if config.algorithm_condition == "dynamic_character_rephasing"
    )
    return [baseline_config(), dynamic_config]


def sigma_margin_experiment_configs() -> list[SweepConfig]:
    """Return the fixed baseline and five otherwise-identical adaptive margins."""
    configs = [baseline_config()]
    configs.extend(
        SweepConfig(
            algorithm_condition=f"adaptive_sigma_{margin:.1f}".replace(".", "_"),
            clock_period=BASELINE_CLOCK_PERIOD,
            use_click_offset=False,
            delay_learning_mode="enter_only",
            word_clock_mode="adaptive",
            prediction_priority_mode="alternating",
            sigma_margin=margin,
        )
        for margin in SIGMA_MARGIN_SWEEP_VALUES
    )
    for config in configs:
        config.simulation_parameters()
    return configs


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else float("nan")


def summarize_by_user_config(
    phrase_results: pd.DataFrame,
    config_id: str,
    config: SweepConfig,
) -> pd.DataFrame:
    """Aggregate phrase rows into one deterministic row per config and user."""
    config_values = {"config_id": config_id, **asdict(config)}
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


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {seconds:02d}s"
    return f"{minutes:d}m {seconds:02d}s"


def _print_sweep_progress(completed: int, total: int, started_at: float) -> None:
    """Print a durable progress-bar line after each configuration."""
    fraction = completed / total if total else 1.0
    width = 30
    filled = int(width * fraction)
    bar = "#" * filled + "-" * (width - filled)
    elapsed = time.monotonic() - started_at
    eta = elapsed / completed * (total - completed) if completed else None
    eta_text = _format_duration(eta) if eta is not None else "calculating"
    print(
        f"Sweep progress [{bar}] {completed}/{total} "
        f"({fraction * 100:5.1f}%) | elapsed {_format_duration(elapsed)} "
        f"| ETA {eta_text}",
        flush=True,
    )


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


def run_config_sweep(
    configs: Optional[Sequence[SweepConfig]] = None,
    *,
    dry_run: bool = True,
    output_directory: Optional[str] = None,
    language_model=None,
    lm_backend: str = "ngram",
    lm_model_path: Optional[str] = None,
    lm_vocabulary_path: Optional[str] = None,
    lm_device: str = "cpu",
    lm_precision: str = "fp32",
    lm_recognizer_nbest: int = 1000,
    study_users: Optional[Sequence[str]] = None,
    phrase_limit: Optional[int] = None,
) -> list[SweepConfig]:
    """Print or execute each named algorithm condition as a paired study."""
    configs = list(
        algorithm_experiment_configs() if configs is None else configs
    )
    if not configs:
        raise ValueError("at least one algorithm condition is required")
    condition_names = [config.algorithm_condition for config in configs]
    if len(set(condition_names)) != len(condition_names):
        raise ValueError("algorithm condition names must be unique")
    for config in configs:
        config.simulation_parameters()
    if dry_run:
        print(f"Dry run: {len(configs)} configuration(s)")
        for index, config in enumerate(configs, start=1):
            print(f"{index:03d}: {asdict(config)}")
        return configs

    from OneClick_Simulation.examples.text_simulation.run_full_study import (
        prepare_study_inputs,
        run_full_study,
    )
    from OneClick_Text.language_model import (
        language_model_metadata,
        load_local_language_model,
    )

    # Validate and load before creating outputs. The same local adapter is
    # reused for every algorithm condition and user.
    if language_model is None:
        if lm_model_path is None:
            raise ValueError("--lm-model-path is required for an actual study run")
        language_model = load_local_language_model(
            lm_model_path,
            backend=lm_backend,
            vocabulary_path=lm_vocabulary_path,
            device=lm_device,
            precision=lm_precision,
            recognizer_nbest=lm_recognizer_nbest,
        )
    lm_metadata = language_model_metadata(language_model)
    print(
        "Language model: local TextSlinger "
        f"({lm_metadata.get('model_path')}) on "
        f"{lm_metadata.get('resolved_device') or lm_device}; "
        "network access disabled",
        flush=True,
    )

    if output_directory is None:
        timestamp = datetime.now().strftime("%m_%d_%Y-%H_%M_%S")
        output_root = (
            Path(__file__).resolve().parent
            / "results"
            / f"algorithm-study-{timestamp}"
        )
    else:
        output_root = Path(output_directory).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    phrase_results_path = output_root / "phrase_results.csv"
    user_summary_path = output_root / "summary_by_user_config.csv"
    config_summary_path = output_root / "summary_by_config.csv"
    existing_outputs = [
        path
        for path in (phrase_results_path, user_summary_path, config_summary_path)
        if path.exists()
    ]
    if existing_outputs:
        raise FileExistsError(
            "Refusing to append a new study to existing output(s): "
            + ", ".join(map(str, existing_outputs))
        )
    (output_root / "lm_config.json").write_text(
        json.dumps(lm_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifest_rows = [
        {"config_id": f"config_{index:03d}", **asdict(config)}
        for index, config in enumerate(configs, start=1)
    ]
    with (output_root / "sweep_config.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=manifest_rows[0].keys())
        writer.writeheader()
        writer.writerows(manifest_rows)
    study_input_options = {}
    if study_users is not None:
        study_input_options["real_users"] = tuple(study_users)
    if phrase_limit is not None:
        study_input_options["phrase_limit"] = phrase_limit
    study_inputs = prepare_study_inputs(output_root, **study_input_options)

    print(
        f"Running {len(configs)} algorithm condition(s) in {output_root}",
        flush=True,
    )
    sweep_started_at = time.monotonic()
    _print_sweep_progress(0, len(configs), sweep_started_at)
    for index, config in enumerate(configs, start=1):
        configuration_id = f"config_{index:03d}"
        simulation_parameters = config.simulation_parameters()

        print(
            f"\n===== {configuration_id}: {config.algorithm_condition} "
            f"({index}/{len(configs)}) =====",
            flush=True,
        )
        study_result = run_full_study(
            simulation_parameters=simulation_parameters,
            output_directory=output_root,
            configuration_id=configuration_id,
            language_model=language_model,
            study_inputs=study_inputs,
            write_outputs=False,
        )
        results = study_result["user_results"].copy()
        if results.empty:
            _print_sweep_progress(index, len(configs), sweep_started_at)
            continue
        user_summary = summarize_by_user_config(results, configuration_id, config)
        config_summary = summarize_by_config(user_summary)
        config_columns = {
            "config_id": configuration_id,
            **asdict(config),
        }
        for column, value in reversed(tuple(config_columns.items())):
            results.insert(0, column, value)
        results.to_csv(
            phrase_results_path,
            mode="a",
            header=not phrase_results_path.exists(),
            index=False,
        )
        user_summary.to_csv(
            user_summary_path,
            mode="a",
            header=not user_summary_path.exists(),
            index=False,
        )
        config_summary.to_csv(
            config_summary_path,
            mode="a",
            header=not config_summary_path.exists(),
            index=False,
        )
        _print_sweep_progress(index, len(configs), sweep_started_at)
    if not phrase_results_path.exists():
        pd.DataFrame(columns=manifest_rows[0].keys()).to_csv(
            phrase_results_path,
            index=False,
        )
    for summary_path in (user_summary_path, config_summary_path):
        if not summary_path.exists():
            pd.DataFrame(columns=SUMMARY_COLUMNS).to_csv(summary_path, index=False)
    lm_metadata = language_model_metadata(language_model)
    (output_root / "lm_config.json").write_text(
        json.dumps(lm_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path = write_sweep_report(output_root, lm_metadata)
    print(f"Algorithm report: {report_path}", flush=True)
    return configs


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configurations without running simulations",
    )
    mode.add_argument(
        "--run",
        action="store_true",
        help="Run all six algorithm conditions on the complete study corpus",
    )
    mode.add_argument(
        "--dynamic-rephasing",
        action="store_true",
        help=(
            "Run the frozen baseline and dynamic character-clock rephasing "
            "on the complete study corpus"
        ),
    )
    mode.add_argument(
        "--sigma-sweep",
        action="store_true",
        help=(
            "Run the fixed baseline plus five BEST-first adaptive sigma margins "
            "on the complete study corpus"
        ),
    )
    mode.add_argument(
        "--baseline",
        action="store_true",
        help=(
            "Run the single frozen n-gram baseline on all study users and "
            "the complete fixed phrase corpus"
        ),
    )
    mode.add_argument(
        "--smoke",
        action="store_true",
        help="Run all six algorithm conditions for user A on one phrase",
    )
    parser.add_argument(
        "--output-directory",
        help="Root directory for algorithm-study result tables",
    )
    parser.add_argument(
        "--lm-model-path",
        help=(
            "TextSlinger n-gram model file; defaults to lm_char_medium.kenlm"
        ),
    )
    parser.add_argument(
        "--lm-vocabulary-path",
        help="TextSlinger word-list file; defaults to vocab_lower_100k.txt",
    )
    parser.add_argument("--lm-recognizer-nbest", type=int, default=1000)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.baseline:
        configs = [baseline_config()]
        study_users = None
        phrase_limit = None
        output_label = "baseline"
    elif args.dynamic_rephasing:
        configs = dynamic_rephasing_experiment_configs()
        study_users = None
        phrase_limit = None
        output_label = "character-rephasing-study"
    elif args.sigma_sweep:
        configs = sigma_margin_experiment_configs()
        study_users = None
        phrase_limit = None
        output_label = "sigma-margin-study"
    elif args.smoke:
        configs = algorithm_experiment_configs()
        study_users = ("A",)
        phrase_limit = 1
        output_label = "algorithm-smoke"
    else:
        configs = algorithm_experiment_configs()
        study_users = None
        phrase_limit = None
        output_label = "algorithm-study"

    output_directory = args.output_directory
    should_run = (
        args.run
        or args.dynamic_rephasing
        or args.sigma_sweep
        or args.baseline
        or args.smoke
    )
    if should_run and output_directory is None:
        timestamp = datetime.now().strftime("%m_%d_%Y-%H_%M_%S")
        output_directory = str(
            Path(__file__).resolve().parent
            / "results"
            / f"{output_label}-{timestamp}"
        )
    lm_backend = "ngram"
    lm_model_path = args.lm_model_path or str(BASELINE_NGRAM_MODEL_PATH)
    lm_vocabulary_path = (
        args.lm_vocabulary_path or str(BASELINE_VOCABULARY_PATH)
    )

    configs = run_config_sweep(
        configs=configs,
        dry_run=not should_run,
        output_directory=output_directory,
        lm_backend=lm_backend,
        lm_model_path=lm_model_path,
        lm_vocabulary_path=lm_vocabulary_path,
        lm_device="cpu",
        lm_precision="fp32",
        lm_recognizer_nbest=args.lm_recognizer_nbest,
        study_users=study_users,
        phrase_limit=phrase_limit,
    )
    if args.baseline:
        if len(configs) != 1:
            raise RuntimeError("baseline mode must produce exactly one configuration")
        baseline_record = {
            "configuration_id": "config_001",
            **asdict(configs[0]),
            "lm_backend": lm_backend,
            "lm_model_path": str(Path(lm_model_path).resolve()),
            "lm_vocabulary_path": str(Path(lm_vocabulary_path).resolve()),
            "maximum_word_clocks": kconfig.fixed_max_word_clocks,
            "study_users": "A,B,C,D,F,G",
            "synthetic_user": "P",
            "phrase_count": 10,
        }
        baseline_path = Path(output_directory) / "baseline_config.json"
        baseline_path.write_text(
            json.dumps(baseline_record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"Baseline configuration: {baseline_path}", flush=True)


if __name__ == "__main__":
    main()
