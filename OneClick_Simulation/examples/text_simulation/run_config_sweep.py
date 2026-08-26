"""Run reproducible QuickClick full-study configuration sweeps."""

from __future__ import annotations

import argparse
import csv
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
        "all_confirmed_clicks",
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


def run_config_sweep(
    values: SweepValues = SweepValues(),
    *,
    dry_run: bool = True,
    output_directory: Optional[str] = None,
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
    study_inputs = prepare_study_inputs(output_root)

    print(f"Running {len(configs)} configuration(s) in {output_root}", flush=True)
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
            study_inputs=study_inputs,
            write_outputs=False,
        )
        results = study_result["user_results"].copy()
        if results.empty:
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
    if not phrase_results_path.exists():
        pd.DataFrame(columns=manifest_rows[0].keys()).to_csv(
            phrase_results_path,
            index=False,
        )
    for summary_path in (user_summary_path, config_summary_path):
        if not summary_path.exists():
            pd.DataFrame(columns=SUMMARY_COLUMNS).to_csv(summary_path, index=False)
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
    parser.add_argument(
        "--output-directory",
        help="Root directory for sweep result tables",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    run_config_sweep(
        dry_run=not args.run,
        output_directory=args.output_directory,
    )


if __name__ == "__main__":
    main()
