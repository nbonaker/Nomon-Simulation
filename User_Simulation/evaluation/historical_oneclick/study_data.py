"""Historical QuickClick data/analysis extracted from OneClick_Simulation.examples.text_simulation.run_full_study.

No simulator execution. Saved result semantics are preserved.
"""
import os

import json

import re

from pathlib import Path

import numpy as np

import pandas as pd

parentdir = str(Path(__file__).resolve().parents[3])

from OneClick_Simulation.examples.text_simulation.analyze_results import summarize_all_users, summarize_user

DATA_ROOT = os.path.join(parentdir, "Nomon_User_Data", "OSF Data")

REAL_USERS = ["A", "B", "C", "D", "F", "G"]

IV_PHRASE_PATH = os.path.join(parentdir, "Nomon_Text", "resources", "watch-iv.txt")

FIXED_IV_PHRASE_COUNT = 10

BOOTSTRAP_BLOCK_SIZE = 20

BOOTSTRAP_BASE_SEED = 20260825

CONFIGURATION = "full_study_block_bootstrap"

def load_fixed_iv_phrases(phrase_count=FIXED_IV_PHRASE_COUNT):
    """Load a fixed number of existing IV labels/texts in repository order."""
    phrase_count = int(phrase_count)
    if phrase_count < 1:
        raise ValueError("phrase_count must be at least 1")
    rows = []
    with open(IV_PHRASE_PATH, "r") as phrase_file:
        for line in phrase_file:
            if "\t" not in line:
                continue
            phrase_id, phrase_text = line.rstrip("\n").split("\t", 1)
            phrase_text = re.sub(r"[^a-z \']+", "", phrase_text.lower())
            phrase_text = re.sub(r"  +", " ", phrase_text).strip()
            rows.append((phrase_id, phrase_text))
            if len(rows) == phrase_count:
                break
    if len(rows) != phrase_count:
        raise ValueError(
            f"Expected {phrase_count} IV phrases in {IV_PHRASE_PATH}, "
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

def corpus_maximum_click_budget(phrase_df, *, max_clicks_per_word):
    word_count = int(phrase_df["Phrase Text"].str.split().str.len().sum())
    return word_count * int(max_clicks_per_word)

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

def preflight_click_stream(user_id, click_df, phrase_df, *, max_clicks_per_word):
    """Return deterministic lower/upper click-budget checks before simulation."""
    available = int(click_df["Session Num"].notna().sum())
    word_counts = phrase_df["Phrase Text"].str.split().str.len().astype(int)
    minimum_by_phrase = word_counts * 2  # at least one Space + one Enter per word
    maximum_by_phrase = word_counts * int(max_clicks_per_word)
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
        "Attempted All Phrases": False,
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
    report["Attempted All Phrases"] = actual_ids == expected_ids
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

def perfect_click_df(n=8000, *, period_s):
    T = float(period_s)
    return pd.DataFrame({
        "Session Num": [1.0] * n,
        "Clock Period (s)": [T] * n,
        "Click Time Relative (s)": [0.0] * n,
        "Dead Time (s)": [0.0] * n,
    })

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
        if "Attempted All Phrases" in real:
            real_mean["Attempted All Phrases"] = bool(
                real["Attempted All Phrases"].all()
            )
        if "Click Stream Exhausted" in real:
            real_mean["Click Stream Exhausted"] = bool(
                real["Click Stream Exhausted"].any()
            )
        parts.append(real_mean)
    if not synthetic.empty:
        parts.append(synthetic)
    return pd.concat(parts, ignore_index=True, sort=False)

def write_summary(summary_rows, output_directory):
    summary = build_summary(summary_rows)
    summary.to_csv(os.path.join(output_directory, "summary.csv"), index=False)
    return summary
