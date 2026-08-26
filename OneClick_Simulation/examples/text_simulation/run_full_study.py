"""
Full OneClick study run: real users A-G and synthetic user P all attempt the
same first 30 canonically labelled IV phrases from watch-iv.txt. Real-user
timing streams are extended to the full worst-case click budget with a
deterministic, within-session moving-block bootstrap.

Writes a combined phrase-level user_results.csv and a summary.csv containing
per-user metrics, an equal-user-weight real-user mean, and a separate
synthetic-perfect-user row. Run from Nomon-Simulation/:

    python3 -m OneClick_Simulation.examples.text_simulation.run_full_study
"""
import os
import sys
import inspect
import json
import re
from datetime import datetime

import numpy as np
import pandas as pd

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(os.path.dirname(currentdir)))  # Nomon-Simulation/
sys.path.insert(0, parentdir)
os.chdir(parentdir)

from OneClick_Core import config
from OneClick_Simulation.examples.text_simulation.analyze_results import (
    summarize_all_users,
    summarize_user,
)
from OneClick_Simulation.simulated_user import SimulatedUser
from OneClick_Simulation import sim_config

DATA_ROOT = os.path.join(parentdir, "Nomon_User_Data", "OSF Data")
OUT_DIR = os.path.join(currentdir, "results",
                       "sim-" + datetime.now().strftime("%m_%d_%Y-%H_%M"))
REAL_USERS = ["A", "B", "C", "D", "F", "G"]
IV_PHRASE_PATH = os.path.join(parentdir, "Nomon_Text", "resources", "watch-iv.txt")
FIXED_IV_PHRASE_COUNT = 10
CLICK_SAMPLING_MODE = "block_bootstrap"
BOOTSTRAP_BLOCK_SIZE = 20
BOOTSTRAP_BASE_SEED = 20260825

CONFIGURATION = "full_study_block_bootstrap"
WORD_CLOCK_MODE = "fixed"
SIGMA_MARGIN = None


def study_simulation_parameters():
    """Return word-clock parameters passed through to every study simulation."""
    parameters = {"word_clock_mode": WORD_CLOCK_MODE}
    if WORD_CLOCK_MODE == "adaptive":
        if SIGMA_MARGIN is None:
            raise ValueError("adaptive study mode requires SIGMA_MARGIN")
        parameters["sigma_margin"] = SIGMA_MARGIN
    return parameters


def load_fixed_iv_phrases():
    """Load the first 30 existing IV labels/texts in repository file order."""
    rows = []
    with open(IV_PHRASE_PATH, "r") as phrase_file:
        for line in phrase_file:
            if "\t" not in line:
                continue
            phrase_id, phrase_text = line.rstrip("\n").split("\t", 1)
            phrase_text = re.sub(r"[^a-z \']+", "", phrase_text.lower())
            phrase_text = re.sub(r"  +", " ", phrase_text).strip()
            rows.append((phrase_id, phrase_text))
            if len(rows) == FIXED_IV_PHRASE_COUNT:
                break
    if len(rows) != FIXED_IV_PHRASE_COUNT:
        raise ValueError(
            f"Expected {FIXED_IV_PHRASE_COUNT} IV phrases in {IV_PHRASE_PATH}, "
            f"found {len(rows)}"
        )
    return pd.DataFrame(
        {
            "Session Num": [1.0] * len(rows),
            "Phrase Num": np.arange(1, len(rows) + 1),
            "Comparison Phrase ID": [phrase_id for phrase_id, _ in rows],
            "Phrase Text": [phrase_text for _, phrase_text in rows],
            "Phrase Type": ["iv"] * len(rows),
        }
    )


def load_real_user_clicks(user_id):
    """Load text-entry clicks only; all users start from the config timing prior."""
    cols = ["Session Num", "Clock Period (s)", "Click Time Relative (s)", "Dead Time (s)"]
    txt = pd.read_csv(os.path.join(DATA_ROOT, "text_entry_task",
                                   f"user_{user_id}_text_click_data_clean.csv"), usecols=cols)
    txt["Original Session Num"] = txt["Session Num"]
    txt["Bootstrap Source Row"] = np.arange(len(txt), dtype=int)
    return txt


def corpus_maximum_click_budget(phrase_df):
    word_count = int(phrase_df["Phrase Text"].str.split().str.len().sum())
    return word_count * int(sim_config.max_clicks_per_word)


def bootstrap_seed_for_user(user_id):
    """Stable across processes/config runs; unlike hash(), this never randomizes."""
    return BOOTSTRAP_BASE_SEED + REAL_USERS.index(user_id)


def block_bootstrap_click_stream(
    click_df,
    required_clicks,
    block_size=BOOTSTRAP_BLOCK_SIZE,
    seed=BOOTSTRAP_BASE_SEED,
):
    """Create a fixed-length moving-block bootstrap without crossing sessions."""
    if required_clicks < 1:
        raise ValueError("required_clicks must be at least 1")
    if block_size < 1:
        raise ValueError("block_size must be at least 1")

    calibration = click_df[click_df["Session Num"].isna()].copy()
    source = click_df[click_df["Session Num"].notna()].copy()
    if source.empty:
        raise ValueError("cannot bootstrap an empty text-entry click stream")
    if "Original Session Num" not in source:
        source["Original Session Num"] = source["Session Num"]
    if "Bootstrap Source Row" not in source:
        source["Bootstrap Source Row"] = np.arange(len(source), dtype=int)

    # Each candidate is a consecutive source block wholly inside one original
    # session. Sampling candidates with replacement is the moving-block bootstrap.
    candidates = []
    for _session_num, session in source.groupby(
        "Original Session Num", sort=False, dropna=False
    ):
        session = session.reset_index(drop=True)
        current_block_size = min(int(block_size), len(session))
        for start in range(len(session) - current_block_size + 1):
            candidates.append(session.iloc[start : start + current_block_size].copy())
    if not candidates:
        raise ValueError("no within-session bootstrap blocks could be constructed")

    rng = np.random.default_rng(int(seed))
    sampled_blocks = []
    sampled_count = 0
    block_id = 0
    while sampled_count < required_clicks:
        block = candidates[int(rng.integers(0, len(candidates)))].copy()
        block["Bootstrap Block ID"] = block_id
        block["Bootstrap Position in Block"] = np.arange(len(block), dtype=int)
        sampled_blocks.append(block)
        sampled_count += len(block)
        block_id += 1

    schedule = pd.concat(sampled_blocks, ignore_index=True).iloc[:required_clicks].copy()
    schedule["Bootstrap Schedule Row"] = np.arange(len(schedule), dtype=int)
    # SimulatedUser sees a single fixed-corpus playthrough. Original session IDs
    # remain in their own column for audit and block-boundary verification.
    schedule["Session Num"] = 1.0
    return pd.concat([calibration, schedule], ignore_index=True, sort=False)


def preflight_click_stream(user_id, click_df, phrase_df):
    """Return deterministic lower/upper click-budget checks before simulation."""
    available = int(click_df["Session Num"].notna().sum())
    word_counts = phrase_df["Phrase Text"].str.split().str.len().astype(int)
    minimum_by_phrase = word_counts * 2  # at least one Space + one Enter per word
    maximum_by_phrase = word_counts * int(sim_config.max_clicks_per_word)
    minimum_required = int(minimum_by_phrase.sum())
    maximum_required = int(maximum_by_phrase.sum())
    first_not_guaranteed = ""
    cumulative_maximum = maximum_by_phrase.cumsum()
    if available < maximum_required:
        index = int(np.flatnonzero(cumulative_maximum.to_numpy() > available)[0])
        first_not_guaranteed = str(phrase_df.iloc[index]["Comparison Phrase ID"])
    return {
        "User": user_id,
        "Available Click Rows": available,
        "Minimum Clicks Required": minimum_required,
        "Maximum Click Budget": maximum_required,
        "Minimum Sufficiency Passed": bool(available >= minimum_required),
        "Full Corpus Guaranteed": bool(available >= maximum_required),
        "First Phrase Not Guaranteed": first_not_guaranteed,
        "Attempted All 30 Phrases": False,
        "Click Stream Exhausted": False,
        "Exhaustion Phrase ID": "",
        "Exhaustion Reason": "",
        "First Unattempted Phrase ID": "",
        "Clicks Consumed": 0,
    }


def audit_click_stream_result(report, result_df, phrase_df):
    """Add the exact post-run exhaustion location to a preflight report."""
    expected_ids = phrase_df["Comparison Phrase ID"].astype(str).tolist()
    actual_ids = result_df.get("Comparison Phrase ID", pd.Series(dtype=str)).astype(str).tolist()
    report = dict(report)
    report["Attempted All 30 Phrases"] = actual_ids == expected_ids
    report["Clicks Consumed"] = int(result_df["Num Clicks"].sum()) if not result_df.empty else 0
    if len(actual_ids) < len(expected_ids):
        report["First Unattempted Phrase ID"] = expected_ids[len(actual_ids)]

    for row in result_df.to_dict("records"):
        events = json.loads(row.get("Failure Events", "[]"))
        for event in events:
            if str(event.get("reason", "")).startswith("click_stream_exhausted"):
                report["Click Stream Exhausted"] = True
                report["Exhaustion Phrase ID"] = str(row.get("Comparison Phrase ID", ""))
                report["Exhaustion Reason"] = str(event["reason"])
                return report
    return report


def perfect_click_df(n=8000):
    T = config.period_li[config.default_rotate_ind]
    return pd.DataFrame({
        "Session Num": [1.0] * n,
        "Clock Period (s)": [T] * n,
        "Click Time Relative (s)": [0.0] * n,
        "Dead Time (s)": [0.0] * n,
    })


def run_one(user_id, params, simulation_parameters=None):
    # Output-only policy: retain every attempted phrase, including attempts that
    # terminate before reaching the old halfway-recording threshold.
    if simulation_parameters is None:
        simulation_parameters = study_simulation_parameters()
    params = {**simulation_parameters, **params}
    params["record_attempted_phrases"] = True
    sim = SimulatedUser()
    sim.parameter_metrics(params, trials=1, verbose=False)
    df = sim.result_df.copy()
    df["user_id"] = user_id
    return df


def summarize_result(user_id, df, configuration_id=CONFIGURATION):
    analysis_df = df.copy()
    analysis_df["Configuration"] = configuration_id
    return summarize_user(analysis_df, user_id)


def build_summary(summary_rows):
    """Keep user rows, add an equal-user-weight real-user mean, and keep P separate."""
    if not summary_rows:
        return pd.DataFrame()
    users = pd.DataFrame(summary_rows)
    real = users[users["User"].isin(REAL_USERS)]
    synthetic = users[~users["User"].isin(REAL_USERS)]
    parts = [real]
    if not real.empty:
        real_mean = summarize_all_users(real)
        real_mean["User"] = "MEAN_REAL_USERS"
        if "Attempted All 30 Phrases" in real:
            real_mean["Attempted All 30 Phrases"] = bool(
                real["Attempted All 30 Phrases"].all()
            )
        if "Click Stream Exhausted" in real:
            real_mean["Click Stream Exhausted"] = bool(
                real["Click Stream Exhausted"].any()
            )
        parts.append(real_mean)
    if not synthetic.empty:
        parts.append(synthetic)
    return pd.concat(parts, ignore_index=True, sort=False)


def write_summary(summary_rows, output_directory=OUT_DIR):
    summary = build_summary(summary_rows)
    summary.to_csv(os.path.join(output_directory, "summary.csv"), index=False)
    return summary


def prepare_study_inputs(output_directory):
    """Prepare shared corpus, deterministic click schedules, and preflight data."""
    output_directory = os.fspath(output_directory)
    os.makedirs(output_directory, exist_ok=True)
    phrase_df = load_fixed_iv_phrases()
    phrase_df.to_csv(
        os.path.join(output_directory, "fixed_iv_phrase_corpus.csv"),
        index=False,
    )

    original_real_clicks = {
        user_id: load_real_user_clicks(user_id) for user_id in REAL_USERS
    }
    maximum_click_budget = corpus_maximum_click_budget(phrase_df)
    real_clicks = {}
    bootstrap_metadata = {}
    for user_id in REAL_USERS:
        seed = bootstrap_seed_for_user(user_id)
        original = original_real_clicks[user_id]
        real_clicks[user_id] = block_bootstrap_click_stream(
            original,
            required_clicks=maximum_click_budget,
            block_size=BOOTSTRAP_BLOCK_SIZE,
            seed=seed,
        )
        bootstrap_metadata[user_id] = {
            "Original Click Rows": int(original["Session Num"].notna().sum()),
            "Click Sampling Mode": CLICK_SAMPLING_MODE,
            "Bootstrap Block Size": BOOTSTRAP_BLOCK_SIZE,
            "Bootstrap Seed": seed,
        }

    sufficiency_rows = [
        {
            **preflight_click_stream(user_id, real_clicks[user_id], phrase_df),
            **bootstrap_metadata[user_id],
        }
        for user_id in REAL_USERS
    ]
    pd.DataFrame(sufficiency_rows).to_csv(
        os.path.join(output_directory, "click_stream_sufficiency.csv"),
        index=False,
    )
    return {
        "phrase_df": phrase_df,
        "real_clicks": real_clicks,
        "bootstrap_metadata": bootstrap_metadata,
        "sufficiency_rows": sufficiency_rows,
        "configuration_sufficiency_rows": [],
        "sufficiency_output_path": os.path.join(
            output_directory, "click_stream_sufficiency.csv"
        ),
        "maximum_click_budget": maximum_click_budget,
    }


def run_full_study(
    simulation_parameters,
    output_directory,
    configuration_id,
    study_inputs=None,
    write_outputs=True,
):
    """Run the complete study once for a supplied simulation configuration."""
    simulation_parameters = dict(simulation_parameters)
    output_directory = os.fspath(output_directory)
    os.makedirs(output_directory, exist_ok=True)
    print("Writing results to", output_directory, flush=True)
    summary_rows = []
    user_result_frames = []
    if study_inputs is None:
        study_inputs = prepare_study_inputs(output_directory)
    phrase_df = study_inputs["phrase_df"]
    real_clicks = study_inputs["real_clicks"]
    bootstrap_metadata = study_inputs["bootstrap_metadata"]
    sufficiency_rows = [dict(row) for row in study_inputs["sufficiency_rows"]]
    maximum_click_budget = study_inputs["maximum_click_budget"]
    print("\n===== CLICK-STREAM PREFLIGHT =====", flush=True)
    print(pd.DataFrame(sufficiency_rows).to_string(index=False), flush=True)

    for report in sufficiency_rows:
        user_id = report["User"]
        print(f"\n===== USER {user_id} =====", flush=True)
        if not report["Minimum Sufficiency Passed"]:
            print(
                f"USER {user_id} SKIPPED: {report['Available Click Rows']} clicks "
                f"cannot meet the {report['Minimum Clicks Required']}-click minimum",
                flush=True,
            )
            continue
        try:
            df = run_one(
                user_id,
                {"click_df": real_clicks[user_id], "phrase_df": phrase_df},
                simulation_parameters,
            )
            df["config_click_sampling_mode"] = CLICK_SAMPLING_MODE
            df["config_bootstrap_block_size"] = BOOTSTRAP_BLOCK_SIZE
            df["config_bootstrap_seed"] = bootstrap_seed_for_user(user_id)
            df["config_original_click_rows"] = bootstrap_metadata[user_id][
                "Original Click Rows"
            ]
            df["config_scheduled_click_rows"] = maximum_click_budget
            df["Configuration"] = configuration_id
            user_result_frames.append(df)
            updated_report = audit_click_stream_result(report, df, phrase_df)
            sufficiency_rows[REAL_USERS.index(user_id)] = updated_report
            summary = summarize_result(user_id, df, configuration_id)
            summary["Attempted All 30 Phrases"] = updated_report[
                "Attempted All 30 Phrases"
            ]
            summary["Click Stream Exhausted"] = updated_report["Click Stream Exhausted"]
            summary_rows.append(summary)
            if write_outputs:
                write_summary(summary_rows, output_directory)
            if updated_report["Click Stream Exhausted"]:
                print(
                    f"USER {user_id} RAN OUT at "
                    f"{updated_report['Exhaustion Phrase ID']}: "
                    f"{updated_report['Exhaustion Reason']}; first unattempted="
                    f"{updated_report['First Unattempted Phrase ID'] or 'none'}",
                    flush=True,
                )
            elif not updated_report["Attempted All 30 Phrases"]:
                print(
                    f"USER {user_id} DID NOT ATTEMPT ALL PHRASES; first missing="
                    f"{updated_report['First Unattempted Phrase ID']}",
                    flush=True,
                )
            print(f"USER {user_id} done: {len(df)} phrases", flush=True)
        except Exception as e:
            print(f"USER {user_id} FAILED: {e}", flush=True)

    configuration_sufficiency = [
        {"Configuration": configuration_id, **row} for row in sufficiency_rows
    ]
    accumulated_sufficiency = study_inputs.setdefault(
        "configuration_sufficiency_rows", []
    )
    accumulated_sufficiency.extend(configuration_sufficiency)
    pd.DataFrame(accumulated_sufficiency).to_csv(
        study_inputs["sufficiency_output_path"],
        index=False,
    )

    # Synthetic perfect user P on the same fixed IV corpus, with no timing noise.
    print("\n===== USER P (perfect) =====", flush=True)
    try:
        perfect_clicks = perfect_click_df()
        df = run_one(
            "P",
            {"click_df": perfect_clicks, "phrase_df": phrase_df},
            simulation_parameters,
        )
        df["config_click_sampling_mode"] = "synthetic_perfect"
        df["config_bootstrap_block_size"] = np.nan
        df["config_bootstrap_seed"] = np.nan
        df["config_original_click_rows"] = len(perfect_clicks)
        df["config_scheduled_click_rows"] = len(perfect_clicks)
        df["Configuration"] = configuration_id
        user_result_frames.append(df)
        perfect_report = audit_click_stream_result(
            preflight_click_stream("P", perfect_clicks, phrase_df),
            df,
            phrase_df,
        )
        perfect_summary = summarize_result("P", df, configuration_id)
        perfect_summary["Attempted All 30 Phrases"] = perfect_report[
            "Attempted All 30 Phrases"
        ]
        perfect_summary["Click Stream Exhausted"] = perfect_report[
            "Click Stream Exhausted"
        ]
        summary_rows.append(perfect_summary)
        if write_outputs:
            write_summary(summary_rows, output_directory)
        print(f"USER P done: {len(df)} phrases", flush=True)
    except Exception as e:
        print(f"USER P FAILED: {e}", flush=True)

    user_results = (
        pd.concat(user_result_frames, ignore_index=True, sort=False)
        if user_result_frames
        else pd.DataFrame()
    )
    final_summary = build_summary(summary_rows)
    if write_outputs:
        user_results.to_csv(
            os.path.join(output_directory, "user_results.csv"),
            index=False,
        )
        final_summary.to_csv(
            os.path.join(output_directory, "summary.csv"),
            index=False,
        )
    print("\n========== SUMMARY ==========", flush=True)
    print(final_summary.to_string(index=False), flush=True)
    if write_outputs:
        print(
            "\nALL DONE ->",
            os.path.join(output_directory, "summary.csv"),
            flush=True,
        )
    return {"summary": final_summary, "user_results": user_results}


def main():
    run_full_study(
        simulation_parameters=study_simulation_parameters(),
        output_directory=OUT_DIR,
        configuration_id=CONFIGURATION,
    )


if __name__ == "__main__":
    main()
