"""
Sanity check: perfect simulated user.

A "perfect" user has click_offset=0 and dead_time=0 for every click.
This means every Space press lands exactly when the target letter clock is at noon,
and every Enter press lands exactly when the target word clock is at noon.

Run from Nomon-Simulation/:
    python3 -m OneClick_Simulation.sanity_checks
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

from OneClick_Core import config
from OneClick_Simulation.simulated_user import SimulatedUser


def make_perfect_clicks(n, session_num=1):
    """Return a click DataFrame where every click is exactly at noon."""
    T = config.period_li[config.default_rotate_ind]
    return pd.DataFrame({
        'Session Num': [float(session_num)] * n,
        'Clock Period (s)': [T] * n,
        'Click Time Relative (s)': [0.0] * n,
        'Dead Time (s)': [0.0] * n,
    })


def run_perfect_user(phrase, n_clicks=200, verbose=True):
    """
    Run the simulation with a perfect user on a single phrase.
    Returns the results dict and prints a comparison of expected vs actual.
    """
    words = phrase.strip().split()
    T = config.period_li[config.default_rotate_ind]

    click_df = make_perfect_clicks(n_clicks)
    phrase_df = pd.DataFrame({
        'Session Num': [1.0],
        'Phrase Text': [phrase],
    })

    sim = SimulatedUser()
    sim.parameter_metrics({'click_df': click_df, 'phrase_df': phrase_df},
                          trials=1, verbose=verbose)

    if sim.result_df.empty:
        print("No phrases completed — ran out of clicks or phrase too short.")
        return None

    row = sim.result_df.iloc[0]

    print("\n" + "="*60)
    print(f"Phrase: '{phrase}'")
    print(f"Words:  {words}")
    print("="*60)

    # Upper bound on click load per selection: one Space per letter + one Enter per
    # word (N+1), achieved only if every word requires all its letters before the
    # API suggests it. Early commits (API suggests the word after k < N letters)
    # bring the actual value below this bound.
    avg_word_len = sum(len(w) for w in words) / len(words)
    upper_bound_clicks_per_sel = avg_word_len + 1

    print(f"\nUpper bound clicks/selection (N+1 per word, avg N={avg_word_len:.1f}): "
          f"{upper_bound_clicks_per_sel:.2f}")
    print(f"Actual  clicks/selection:  {row['Click Load (clicks/selection)']:.2f}")

    actual_cps = row['Click Load (clicks/selection)']
    # With 0% correction rate, actual <= upper bound:
    #   - actual == upper bound: every word needed all N letters before commit
    #   - actual <  upper bound: some words committed early (API suggested them
    #     before all letters were typed) — this is the EXPECTED, GOOD outcome
    # With corrections (undo cycles), actual can exceed the upper bound.
    if abs(actual_cps - upper_bound_clicks_per_sel) < 0.01:
        print("  ✓ AT UPPER BOUND — zero corrections, no early commits "
              "(every word needed all N letters)")
    elif actual_cps < upper_bound_clicks_per_sel:
        saved = upper_bound_clicks_per_sel - actual_cps
        print(f"  ✓ {saved:.2f} fewer clicks/selection than the N+1 upper bound — "
              f"some words committed early (API suggested them before all letters "
              f"were typed)")
    else:
        extra = actual_cps - upper_bound_clicks_per_sel
        print(f"  ~ {extra:.2f} extra clicks/selection above the N+1 upper bound — "
              f"some words needed undo (API didn't return target word)")

    print(f"\nCorrection Rate: {row['Correction Rate (%)']:.1f}%  "
          f"(expected 0% if API returns all target words)")
    print(f"Word Pred Usage: {row['Word Prediction Usage (%)']:.1f}%  "
          f"(fraction of selections where API returned the target word directly)")
    print(f"Entry Rate:      {row['Entry Rate (wpm)']:.2f} wpm")
    print(f"Error Rate:      {row['Error Rate (%)']:.1f}%  (expected 0%)")

    # Hard assertion: error rate (edit distance between typed and target) must be 0
    # because even if we undo and retry, we only commit correct words.
    assert row['Error Rate (%)'] == 0.0, \
        f"Error rate should be 0% for a perfect user (got {row['Error Rate (%)']:.2f}%)"
    print("\n✓ Error rate == 0%: typed text exactly matches target (up to what was typed)")

    return row


if __name__ == "__main__":
    print("OneClick perfect-user sanity check\n")

    # Phrase 1: short, very common words — API should return all of them
    run_perfect_user("the cat", verbose=True)

    # Phrase 2: slightly longer phrase
    run_perfect_user("she is happy", verbose=True)
