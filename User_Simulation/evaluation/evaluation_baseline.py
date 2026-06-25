"""Baseline evaluator for the current replay-style simulated user.

Run from the repository root:

    python -m User_Simulation.evaluation.evaluation_baseline

By default this evaluates user A on the first available session to keep the
feedback loop short. Use --all-sessions for a full run.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from User_Simulation.evaluation.metrics import (
    compare_summaries,
    summarize_real_run,
    summarize_simulated_run,
)
from User_Simulation.simulated_user_text import SimulatedUser


REPO_ROOT = Path(__file__).resolve().parents[2]
USER_DATA_ROOT = REPO_ROOT / "Nomon_User_Data" / "OSF Data"
TEXT_DATA_ROOT = USER_DATA_ROOT / "text_entry_task"
SYMBOL_DATA_ROOT = USER_DATA_ROOT / "picture_selection_task"
TEXT_RESOURCE_ROOT = REPO_ROOT / "Nomon_Text" / "resources"


def load_text_click_data(user_id: str) -> pd.DataFrame:
    path = TEXT_DATA_ROOT / f"user_{user_id}_text_click_data_clean.csv"
    return pd.read_csv(path)


def load_text_phrase_data(user_id: str) -> pd.DataFrame:
    path = TEXT_DATA_ROOT / f"user_{user_id}_text_phrase_data_clean.csv"
    return pd.read_csv(path)


def load_calibration_click_data(user_id: str) -> pd.DataFrame:
    path = SYMBOL_DATA_ROOT / f"user_{user_id}_click_data.csv"
    columns = ["Session Num", "Clock Period (s)", "Click Time Relative (s)", "Dead Time (s)"]

    if not path.exists():
        return pd.DataFrame(columns=columns)

    calibration_df = pd.read_csv(path, usecols=columns)
    calibration_df["Session Num"] = np.nan
    calibration_df["Dead Time (s)"] = np.nan
    return calibration_df


def select_sessions(
    click_df: pd.DataFrame,
    phrase_df: pd.DataFrame,
    max_sessions: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if max_sessions is None:
        return click_df, phrase_df

    sessions = sorted(click_df["Session Num"].dropna().unique())[:max_sessions]
    return (
        click_df[click_df["Session Num"].isin(sessions)].copy(),
        phrase_df[phrase_df["Session Num"].isin(sessions)].copy(),
    )


def normalize_phrase_order(phrase_df: pd.DataFrame) -> pd.DataFrame:
    return phrase_df.sort_values(["Session Num", "Phrase Num"]).reset_index(drop=True)


def lm_files(lm_size: str) -> list[str]:
    word_lm_path = TEXT_RESOURCE_ROOT / f"lm_word_{lm_size}.kenlm"
    char_lm_path = TEXT_RESOURCE_ROOT / f"lm_char_{lm_size}.kenlm"
    vocab_path = TEXT_RESOURCE_ROOT / "vocab_lower_100k.txt"
    char_path = TEXT_RESOURCE_ROOT / "char_set.txt"

    paths = [word_lm_path, char_lm_path, vocab_path, char_path]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing language model resources: " + ", ".join(missing))

    return [str(path) for path in paths]


def make_sim_click_df(user_id: str, text_click_df: pd.DataFrame) -> pd.DataFrame:
    calibration_df = load_calibration_click_data(user_id)
    full_click_df = pd.concat([calibration_df, text_click_df], ignore_index=True, sort=False)

    # SimulatedUser.parameter_metrics currently has a typo in this column name.
    # Supplying the alias keeps the evaluator non-invasive while preserving the
    # existing simulator as the baseline under test.
    full_click_df["Click Time Rlative (s)"] = full_click_df["Click Time Relative (s)"]
    return full_click_df


def run_baseline_simulator(
    user_id: str,
    click_df: pd.DataFrame,
    phrase_df: pd.DataFrame,
    lm_size: str,
    trials: int,
    verbose: bool,
) -> SimulatedUser:
    sim = SimulatedUser()
    params = {
        "click_df": make_sim_click_df(user_id, click_df),
        "phrase_df": phrase_df,
        "lm_files": lm_files(lm_size),
    }

    # Some legacy paths in the simulator are relative to User_Simulation.
    original_cwd = Path.cwd()
    try:
        os.chdir(REPO_ROOT / "User_Simulation")
        sim.parameter_metrics(params, trials=trials, verbose=verbose)
    finally:
        os.chdir(original_cwd)

    return sim


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def build_output_dir(base_dir: Path, user_id: str) -> Path:
    timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    output_dir = base_dir / f"baseline_user_{user_id}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def build_phrase_comparison(real_phrase_df: pd.DataFrame, simulated_phrase_df: pd.DataFrame) -> pd.DataFrame:
    real_columns = {
        "Phrase Num": "Original Phrase Num",
        "Phrase Text": "Real Target Phrase",
        "Typed Text": "Real Typed Text",
        "Entry Rate (wpm)": "Real Entry Rate (wpm)",
        "Click Load (clicks/character)": "Real Click Load (clicks/character)",
        "Correction Rate (% of selections)": "Real Correction Rate (%)",
        "Final Error Rate (%)": "Real Error Rate (%)",
    }
    sim_columns = {
        "Phrase Num": "Sim Phrase Num",
        "Target Phrase": "Sim Target Phrase",
        "Typed Text": "Sim Typed Text",
        "Entry Rate (wpm)": "Sim Entry Rate (wpm)",
        "Click Load (clicks/character)": "Sim Click Load (clicks/character)",
        "Correction Rate (%)": "Sim Correction Rate (%)",
        "Error Rate (%)": "Sim Error Rate (%)",
    }

    real_compare = real_phrase_df[["Session Num"] + list(real_columns)].rename(columns=real_columns)

    if simulated_phrase_df.empty:
        sim_compare = pd.DataFrame(columns=["Session Num", "Original Phrase Num"] + list(sim_columns.values()))
    else:
        sim_compare = simulated_phrase_df.copy()
        if "Original Phrase Num" not in sim_compare:
            sim_compare["Original Phrase Num"] = pd.NA
        sim_compare = sim_compare[["Session Num", "Original Phrase Num"] + list(sim_columns)].rename(columns=sim_columns)

    comparison_df = real_compare.merge(
        sim_compare,
        on=["Session Num", "Original Phrase Num"],
        how="outer",
        indicator=True,
    )
    comparison_df["Alignment Status"] = comparison_df["_merge"].map(
        {
            "both": "matched",
            "left_only": "missing_simulated",
            "right_only": "extra_simulated",
        }
    )
    comparison_df = comparison_df.drop(columns=["_merge"])
    comparison_df = comparison_df.sort_values(["Session Num", "Original Phrase Num"], na_position="last")
    return comparison_df.reset_index(drop=True)


def build_alignment_summary(phrase_comparison_df: pd.DataFrame, simulated_phrase_df: pd.DataFrame) -> dict:
    status_counts = phrase_comparison_df["Alignment Status"].value_counts()
    simulated_original_phrase_nums = (
        simulated_phrase_df["Original Phrase Num"].notna().sum()
        if "Original Phrase Num" in simulated_phrase_df
        else 0
    )

    return {
        "real_phrase_rows": int((phrase_comparison_df["Alignment Status"] != "extra_simulated").sum()),
        "simulated_phrase_rows": int(len(simulated_phrase_df)),
        "matched_rows": int(status_counts.get("matched", 0)),
        "missing_simulated_rows": int(status_counts.get("missing_simulated", 0)),
        "extra_simulated_rows": int(status_counts.get("extra_simulated", 0)),
        "simulated_rows_with_original_phrase_num": int(simulated_original_phrase_nums),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-id", default="A", help="OSF user id to evaluate, for example A or B.")
    parser.add_argument("--lm-size", default="tiny", choices=["tiny", "medium"], help="Bundled KenLM model size.")
    parser.add_argument("--trials", type=int, default=1, help="Number of simulator trials.")
    parser.add_argument("--max-sessions", type=int, default=1, help="Number of sessions to evaluate.")
    parser.add_argument("--all-sessions", action="store_true", help="Evaluate all sessions for the user.")
    parser.add_argument("--verbose", action="store_true", help="Print target/typed details from the simulator.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Directory where evaluation reports are written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_sessions = None if args.all_sessions else args.max_sessions

    real_click_df = load_text_click_data(args.user_id)
    real_phrase_df = load_text_phrase_data(args.user_id)
    real_click_df, real_phrase_df = select_sessions(real_click_df, real_phrase_df, max_sessions)
    real_phrase_df = normalize_phrase_order(real_phrase_df)

    real_summary = summarize_real_run(real_click_df, real_phrase_df)

    sim = run_baseline_simulator(
        user_id=args.user_id,
        click_df=real_click_df,
        phrase_df=real_phrase_df,
        lm_size=args.lm_size,
        trials=args.trials,
        verbose=args.verbose,
    )
    simulated_summary = summarize_simulated_run(
        sim.result_df,
        clicks_used=getattr(sim, "num_clicks_total", None),
    )

    comparison_df = compare_summaries(real_summary, simulated_summary)
    phrase_comparison_df = build_phrase_comparison(real_phrase_df, sim.result_df)
    alignment_summary = build_alignment_summary(phrase_comparison_df, sim.result_df)

    output_dir = build_output_dir(args.output_dir, args.user_id)
    comparison_df.to_csv(output_dir / "metric_comparison.csv", index=False)
    real_phrase_df.to_csv(output_dir / "real_phrase_results.csv", index=False)
    sim.result_df.to_csv(output_dir / "simulated_phrase_results.csv", index=False)
    phrase_comparison_df.to_csv(output_dir / "phrase_comparison.csv", index=False)
    write_json(output_dir / "real_summary.json", real_summary)
    write_json(output_dir / "simulated_summary.json", simulated_summary)
    write_json(output_dir / "alignment_summary.json", alignment_summary)
    write_json(
        output_dir / "run_config.json",
        {
            "user_id": args.user_id,
            "lm_size": args.lm_size,
            "trials": args.trials,
            "max_sessions": max_sessions,
            "real_click_rows": int(len(real_click_df)),
            "real_phrase_rows": int(len(real_phrase_df)),
            "alignment_summary": alignment_summary,
        },
    )

    print(f"Saved evaluation report to: {output_dir}")
    print("Alignment summary:")
    print(json.dumps(alignment_summary, indent=2, sort_keys=True))
    print(comparison_df.to_string(index=False))


if __name__ == "__main__":
    main()
