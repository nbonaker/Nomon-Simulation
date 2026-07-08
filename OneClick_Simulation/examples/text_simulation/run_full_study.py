"""
Full OneClick study run (fixed code): real users A-G on their own recorded phrases,
plus the synthetic perfect user P on the watch-iv/oov phrase set.

Writes one CSV per user as it completes (incremental, crash-safe) and a combined
summary CSV of per-user means. Run from Nomon-Simulation/:

    python3 -m OneClick_Simulation.examples.text_simulation.run_full_study
"""
import os
import sys
import inspect
from datetime import datetime

import numpy as np
import pandas as pd

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(os.path.dirname(currentdir)))  # Nomon-Simulation/
sys.path.insert(0, parentdir)
os.chdir(parentdir)

from OneClick_Core import config
from OneClick_Simulation.simulated_user import SimulatedUser

DATA_ROOT = os.path.join(parentdir, "Nomon_User_Data", "OSF Data")
OUT_DIR = os.path.join(currentdir, "results",
                       "sim-" + datetime.now().strftime("%m_%d_%Y-%H_%M"))
REAL_USERS = ["A", "B", "C", "D", "F", "G"]

# metrics summarised per user (mean over that user's phrases)
METRIC_COLS = [
    "Click Load (clicks/selection)",
    "Entry Rate (wpm)",
    "Correction Rate (%)",
    "Error Rate (%)",
    "Word Prediction Usage (%)",
]


def load_real_user(user_id):
    """Calibration (picture task, Session Num NaN) + text-entry clicks + phrases."""
    pic = os.path.join(DATA_ROOT, "picture_selection_task", f"user_{user_id}_click_data.csv")
    cols = ["Session Num", "Clock Period (s)", "Click Time Relative (s)", "Dead Time (s)"]
    if os.path.exists(pic):
        sym = pd.read_csv(pic, usecols=cols)
        sym["Session Num"] = np.nan
        sym["Dead Time (s)"] = np.nan
    else:
        sym = pd.DataFrame(columns=cols)
    txt = pd.read_csv(os.path.join(DATA_ROOT, "text_entry_task",
                                   f"user_{user_id}_text_click_data_clean.csv"), usecols=cols)
    click_df = pd.concat([sym, txt], ignore_index=True)
    phrase_df = pd.read_csv(os.path.join(DATA_ROOT, "text_entry_task",
                                         f"user_{user_id}_text_phrase_data_clean.csv"),
                            usecols=["Session Num", "Phrase Text"])
    return click_df, phrase_df


def perfect_click_df(n=8000):
    T = config.period_li[config.default_rotate_ind]
    return pd.DataFrame({
        "Session Num": [1.0] * n,
        "Clock Period (s)": [T] * n,
        "Click Time Relative (s)": [0.0] * n,
        "Dead Time (s)": [0.0] * n,
    })


def run_one(user_id, params):
    sim = SimulatedUser()
    sim.parameter_metrics(params, trials=1, verbose=False)
    df = sim.result_df.copy()
    df["user_id"] = user_id
    return df


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Writing results to", OUT_DIR, flush=True)
    summary_rows = []

    for user_id in REAL_USERS:
        print(f"\n===== USER {user_id} =====", flush=True)
        try:
            click_df, phrase_df = load_real_user(user_id)
            df = run_one(user_id, {"click_df": click_df, "phrase_df": phrase_df})
            df.to_csv(os.path.join(OUT_DIR, f"user_{user_id}.csv"), index=False)
            row = {"User": user_id, "Phrases": len(df)}
            row.update({c: round(float(df[c].mean()), 2) for c in METRIC_COLS})
            summary_rows.append(row)
            pd.DataFrame(summary_rows).to_csv(os.path.join(OUT_DIR, "summary.csv"), index=False)
            print(f"USER {user_id} done: {len(df)} phrases", flush=True)
        except Exception as e:
            print(f"USER {user_id} FAILED: {e}", flush=True)

    # synthetic perfect user P (watch-iv/oov phrase set, default period, no noise)
    print("\n===== USER P (perfect) =====", flush=True)
    try:
        df = run_one("P", {"click_df": perfect_click_df()})
        df.to_csv(os.path.join(OUT_DIR, "user_P.csv"), index=False)
        row = {"User": "P", "Phrases": len(df)}
        row.update({c: round(float(df[c].mean()), 2) for c in METRIC_COLS})
        summary_rows.append(row)
        pd.DataFrame(summary_rows).to_csv(os.path.join(OUT_DIR, "summary.csv"), index=False)
        print(f"USER P done: {len(df)} phrases", flush=True)
    except Exception as e:
        print(f"USER P FAILED: {e}", flush=True)

    print("\n========== SUMMARY ==========", flush=True)
    print(pd.DataFrame(summary_rows).to_string(index=False), flush=True)
    print("\nALL DONE ->", os.path.join(OUT_DIR, "summary.csv"), flush=True)


if __name__ == "__main__":
    main()
