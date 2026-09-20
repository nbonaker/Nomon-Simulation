"""Run reproducible QuickClick full-study configuration sweeps."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Optional, Sequence, Tuple

import pandas as pd


SIGMA_MARGIN_VALUES = (1.5, 2.0, 2.5, 3.0, 3.5)
CLOCK_PERIOD_VALUES = (
    0.9917933293295194,
    1.479581783649639,
    2.207276647028654,
    2.9795118227484574,
    4.4449093240903075,
    5.4290245082157575,
)

CONFIG_COLUMNS = [
    "config_id",
    "clock_period",
    "use_click_offset",
    "delay_learning_mode",
    "word_clock_mode",
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
    """One future QuickClick simulation configuration."""

    clock_period: Optional[float]
    use_click_offset: bool
    delay_learning_mode: str
    word_clock_mode: str
    sigma_margin: Optional[float]

    def simulation_parameters(self) -> dict:
        """Translate this sweep row into parameters accepted by SimulatedUser."""
        parameters = {
            "use_click_offset": self.use_click_offset,
            "delay_learning_mode": self.delay_learning_mode,
            "word_clock_mode": self.word_clock_mode,
        }
        if self.clock_period is not None:
            parameters["fixed_clock_period_s"] = self.clock_period
        if self.word_clock_mode == "adaptive":
            if self.sigma_margin is None:
                raise ValueError("adaptive word-clock mode requires sigma_margin")
            parameters["sigma_margin"] = self.sigma_margin
        return parameters


@dataclass(frozen=True)
class SweepValues:
    """Candidate values whose Cartesian product defines a sweep."""

    clock_period: Tuple[Optional[float], ...] = CLOCK_PERIOD_VALUES
    use_click_offset: Tuple[bool, ...] = (False, True)
    delay_learning_mode: Tuple[str, ...] = (
        "enter_only",
        "separate_space_enter",
    )
    word_clock_mode: Tuple[str, ...] = ("fixed", "adaptive")
    sigma_margin: Tuple[float, ...] = SIGMA_MARGIN_VALUES


def generate_config_combinations(values: SweepValues) -> list[SweepConfig]:
    """Return configurations in stable order without changing fixed-N mode."""
    configs = []
    base_values = product(
        values.clock_period,
        values.use_click_offset,
        values.delay_learning_mode,
        values.word_clock_mode,
    )
    for clock_period, use_click_offset, delay_mode, word_clock_mode in base_values:
        margins = values.sigma_margin if word_clock_mode == "adaptive" else (None,)
        for sigma_margin in margins:
            config = SweepConfig(
                clock_period,
                use_click_offset,
                delay_mode,
                word_clock_mode,
                sigma_margin,
            )
            config.simulation_parameters()
            configs.append(config)
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
        successful_clicks = phrases["Successful Word Click Count"].sum()
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
                    successful_clicks, successful_characters
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
    "clock_period",
    "use_click_offset",
    "delay_learning_mode",
    "word_clock_mode",
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
    """Generate a deterministic Markdown overview from the completed CSVs."""
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
        "# QuickClick Configuration Sweep Report",
        "",
        "Generated deterministically from the sweep CSV outputs.",
        "",
        "## Run completeness",
        "",
        _markdown_table(
            ["Item", "Value"],
            [
                ["Configurations planned", len(manifest)],
                [
                    "Configurations represented in summaries",
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
    ]

    table_headers = [
        "Rank",
        *_REPORT_CONFIG_COLUMNS,
        *_REPORT_RESULT_COLUMNS,
    ]
    if not real.empty:
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
                "## Real-user metric leaders",
                "",
                "Each metric is evaluated independently; this is not a universal winner.",
                "",
                _markdown_table(
                    [
                        "Metric",
                        "Config",
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
                "## Completion-first top configurations",
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

        mode_rows = []
        for mode in ("fixed", "adaptive"):
            subset = real[real["word_clock_mode"] == mode]
            if subset.empty:
                continue
            top = _completion_first(subset).iloc[0]
            mode_rows.append(
                [
                    mode,
                    *[top[column] for column in _REPORT_CONFIG_COLUMNS],
                    *[top[column] for column in _REPORT_RESULT_COLUMNS],
                ]
            )
        lines.extend(
            [
                "",
                "## Best configuration by word-clock mode",
                "",
                _markdown_table(
                    ["Mode", *_REPORT_CONFIG_COLUMNS, *_REPORT_RESULT_COLUMNS],
                    mode_rows,
                ),
            ]
        )

        factor_rows = []
        for factor in (
            "clock_period",
            "use_click_offset",
            "delay_learning_mode",
            "word_clock_mode",
            "sigma_margin",
        ):
            factor_data = real
            if factor == "sigma_margin":
                factor_data = factor_data[
                    factor_data["word_clock_mode"] == "adaptive"
                ].dropna(subset=[factor])
            for value, group in factor_data.groupby(
                factor, sort=True, dropna=False
            ):
                factor_rows.append(
                    [
                        factor,
                        value,
                        group["config_id"].nunique(),
                        group["Completion Rate"].mean(),
                        group["Clicks per Character"].mean(),
                        group["Active Typing Time per Phrase"].mean(),
                        group["Correction Rate"].mean(),
                        group["Enter Misselection Rate"].mean(),
                    ]
                )
        lines.extend(
            [
                "",
                "## Real-user factor averages",
                "",
                "Descriptive averages across all configurations containing each value.",
                "",
                _markdown_table(
                    [
                        "Factor",
                        "Value",
                        "Configs",
                        "Completion",
                        "Clicks/character",
                        "Active time/phrase",
                        "Correction rate",
                        "Enter misselection",
                    ],
                    factor_rows,
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
            "- sweep_config.csv: exact configuration manifest.",
            "- click_stream_sufficiency.csv: real-user click-stream audit.",
            "- lm_config.json: local TextSlinger reproducibility metadata.",
            "",
        ]
    )
    report_path = output_root / "sweep_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def run_config_sweep(
    values: SweepValues = SweepValues(),
    *,
    dry_run: bool = True,
    output_directory: Optional[str] = None,
    language_model=None,
    lm_model_path: Optional[str] = None,
    lm_device: str = "mps",
    lm_precision: str = "fp32",
    lm_recognizer_nbest: int = 1000,
    study_users: Optional[Sequence[str]] = None,
    phrase_limit: Optional[int] = None,
) -> list[SweepConfig]:
    """Print or execute every configuration as an independent full study."""
    configs = generate_config_combinations(values)
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

    # Validate and load before creating any sweep outputs. The same adapter and
    # underlying GPU model are reused for every configuration and user.
    if language_model is None:
        if lm_model_path is None:
            raise ValueError("--lm-model-path is required for an actual sweep run")
        language_model = load_local_language_model(
            lm_model_path,
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
        output_root = Path(__file__).resolve().parent / "results" / f"sweep-{timestamp}"
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
            "Refusing to append a new sweep to existing output(s): "
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

    print(f"Running {len(configs)} configuration(s) in {output_root}", flush=True)
    sweep_started_at = time.monotonic()
    _print_sweep_progress(0, len(configs), sweep_started_at)
    for index, config in enumerate(configs, start=1):
        configuration_id = f"config_{index:03d}"
        simulation_parameters = config.simulation_parameters()

        print(
            f"\n===== {configuration_id} ({index}/{len(configs)}) =====",
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
    print(f"Sweep report: {report_path}", flush=True)
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
        help="Execute every configuration as a full study",
    )
    mode.add_argument(
        "--smoke",
        action="store_true",
        help="Run four configurations for user A and synthetic P on one phrase",
    )
    mode.add_argument(
        "--pilot",
        action="store_true",
        help=(
            "Run 48 screening configurations for user A and synthetic P "
            "on three phrases"
        ),
    )
    parser.add_argument(
        "--output-directory",
        help="Root directory for sweep result tables",
    )
    parser.add_argument(
        "--lm-model-path",
        help="Local model directory (required with --run, --pilot, or --smoke)",
    )
    parser.add_argument(
        "--lm-device",
        choices=("mps", "cpu", "cuda"),
        default="mps",
    )
    parser.add_argument(
        "--lm-precision",
        choices=("fp32", "fp16", "bf16"),
        default="fp32",
    )
    parser.add_argument("--lm-recognizer-nbest", type=int, default=1000)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    smoke_values = SweepValues(
        clock_period=(2.207276647028654,),
        use_click_offset=(False,),
        delay_learning_mode=("enter_only", "separate_space_enter"),
        word_clock_mode=("fixed", "adaptive"),
        sigma_margin=(3.0,),
    )
    pilot_values = SweepValues(
        clock_period=(
            CLOCK_PERIOD_VALUES[0],
            CLOCK_PERIOD_VALUES[2],
            CLOCK_PERIOD_VALUES[-1],
        ),
        use_click_offset=(False, True),
        delay_learning_mode=("enter_only", "separate_space_enter"),
        word_clock_mode=("fixed", "adaptive"),
        sigma_margin=(1.5, 2.5, 3.5),
    )

    if args.smoke:
        values = smoke_values
        study_users = ("A",)
        phrase_limit = 1
        output_label = "sweep-smoke"
    elif args.pilot:
        values = pilot_values
        study_users = ("A",)
        phrase_limit = 3
        output_label = "sweep-pilot"
    else:
        values = SweepValues()
        study_users = None
        phrase_limit = None
        output_label = "sweep"

    output_directory = args.output_directory
    if (args.smoke or args.pilot) and output_directory is None:
        timestamp = datetime.now().strftime("%m_%d_%Y-%H_%M_%S")
        output_directory = str(
            Path(__file__).resolve().parent
            / "results"
            / f"{output_label}-{timestamp}"
        )
    run_config_sweep(
        values=values,
        dry_run=not (args.run or args.pilot or args.smoke),
        output_directory=output_directory,
        lm_model_path=args.lm_model_path,
        lm_device=args.lm_device,
        lm_precision=args.lm_precision,
        lm_recognizer_nbest=args.lm_recognizer_nbest,
        study_users=study_users,
        phrase_limit=phrase_limit,
    )


if __name__ == "__main__":
    main()
