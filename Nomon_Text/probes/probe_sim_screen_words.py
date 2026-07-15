#!/usr/bin/python
"""Run the real Nomon sim (user A, session 1) and log every word that appears
on screen (i.e. every non-empty slot returned by get_words), for both the
TextSlinger ngram backend and the legacy KenLM backend. Then report what
fraction of on-screen words are real (in vocab_lower_100k.txt).
"""

import os, sys, json, inspect

SCRATCH = os.path.dirname(os.path.abspath(__file__))  # Nomon_Text/probes
REPO = os.path.dirname(os.path.dirname(SCRATCH))

# Mirror run_sim.py's environment: cwd = User_Simulation
os.chdir(os.path.join(REPO, "User_Simulation"))
sys.path.insert(0, REPO)

import numpy as np
import pandas as pd

from User_Simulation.simulated_user_text import SimulatedUser
from Nomon_Text.textslinger_lm import TextSlingerLM
from Nomon_Text.kenlm.kenlm_lm import LanguageModel as OldLM

BACKEND = sys.argv[1]  # "textslinger" or "kenlm"
LOG_PATH = os.path.join(SCRATCH, f"screen_words_{BACKEND}.jsonl")

RES = os.path.join(REPO, "Nomon_Text", "resources")
char_lm_path = os.path.join(RES, "lm_char_tiny.kenlm")
word_lm_path = os.path.join(RES, "lm_word_tiny.kenlm")
vocab_path = os.path.join(RES, "vocab_lower_100k.txt")
char_path = os.path.join(RES, "char_set.txt")

log_f = open(LOG_PATH, "w")

def make_logged(cls):
    orig = cls.get_words
    def logged(self, left_context, context, keys_li, **kw):
        out = orig(self, left_context, context, keys_li, **kw)
        word_preds, word_probs = out[0], out[1]
        shown, probs = [], []
        for row_w, row_p in zip(word_preds, word_probs):
            for w, p in zip(row_w, row_p):
                if w and str(w).strip():
                    shown.append(str(w).strip())
                    probs.append(float(p))
        log_f.write(json.dumps({
            "left_context": left_context[-40:],
            "prefix": context,
            "shown": shown,
            "probs": probs,
        }) + "\n")
        return out
    cls.get_words = logged

if BACKEND == "textslinger":
    make_logged(TextSlingerLM)
else:
    make_logged(OldLM)

# ---- load user A data, session 1 only ----
user_id = "A"
osf = os.path.join(REPO, "Nomon_User_Data", "OSF Data")

sym = pd.read_csv(os.path.join(osf, "picture_selection_task", f"user_{user_id}_click_data.csv"),
                  usecols=["Session Num", "Clock Period (s)", "Click Time Relative (s)", "Dead Time (s)"])
sym["Session Num"] = np.nan
sym["Dead Time (s)"] = np.nan

txt = pd.read_csv(os.path.join(osf, "text_entry_task", f"user_{user_id}_text_click_data_clean.csv"),
                  usecols=["Session Num", "Clock Period (s)", "Click Time Relative (s)", "Dead Time (s)"])
txt = txt[txt["Session Num"] == 1]

click_df = pd.concat([sym, txt])

phrase_df = pd.read_csv(os.path.join(osf, "text_entry_task", f"user_{user_id}_text_phrase_data_clean.csv"),
                        usecols=["Session Num", "Phrase Text"])
phrase_df = phrase_df[phrase_df["Session Num"] == 1]

params = {"click_df": click_df, "phrase_df": phrase_df, "phrase_shuffle_seed": 0}
if BACKEND == "textslinger":
    params["lm_config"] = {
        "backend": "ngram",
        "lm_path": char_lm_path,
        "character_set_path": char_path,
    }
else:
    params["lm_files"] = (word_lm_path, char_lm_path, vocab_path, char_path)

sim = SimulatedUser()
sim.parameter_metrics(params, trials=1, verbose=False)
log_f.close()
metric_columns = [
    "Click Load (clicks/character)",
    "Correction Rate (%)",
    "Word Prediction Usage (%)",
]
metric_means = sim.result_df[metric_columns].mean().to_dict()
print("Mean metrics:", json.dumps(metric_means, sort_keys=True))
print(f"\nDONE {BACKEND}. Log: {LOG_PATH}")
