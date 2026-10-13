from __future__ import annotations

import json

import tempfile

import unittest

from pathlib import Path

import numpy as np

import pandas as pd

from User_Simulation.evaluation.historical_oneclick.global_sweep import build_global_curve_points, build_global_summary, rank_global_summary, validate_stage_outputs

from User_Simulation.evaluation.historical_oneclick.phase1 import condition_path

def synthetic_phrase_results() -> tuple[pd.DataFrame, list[dict]]:
    combos = json.loads(Path(__file__).with_name("fixtures").joinpath("global_combos.json").read_text())
    completion = {
        ("A", "s06_e06"): [30.0, 80.0],
        ("B", "s06_e06"): [40.0, np.nan],
        ("A", "s06_e10"): [20.0, 50.0],
        ("B", "s06_e10"): [25.0, 60.0],
    }
    rows = []
    for combo in combos:
        for user_id in ["A", "B"]:
            for phrase_index, completion_time in enumerate(
                completion[(user_id, combo["combo_id"])]
            ):
                completed = np.isfinite(completion_time)
                target = f"phrase {phrase_index}"
                attempt_time = (
                    float(completion_time) if completed else 120.0
                )
                rows.append(
                    {
                        "user_id": user_id,
                        "trial": 0,
                        "Comparison Phrase ID": f"p{phrase_index}",
                        "Target Phrase": target,
                        "Typed Text": target if completed else "partial",
                        "phrase_completed": bool(completed),
                        "phrase_failure_reason": (
                            np.nan if completed else "word_click_budget_letters"
                        ),
                        "phrase_failure_stage": (
                            np.nan if completed else "letters"
                        ),
                        "simulated_attempt_time_s": attempt_time,
                        "simulated_completion_time_s": (
                            float(completion_time) if completed else np.nan
                        ),
                        "letter_clock_time_s": attempt_time,
                        "target_enter_clock_time_s": 0.0,
                        "undo_clock_time_s": 0.0,
                        "simulated_time_accounting_error_s": 0.0,
                        "paired_click_schedule_id": f"{user_id}_trial_00",
                        **combo,
                    }
                )
    return pd.DataFrame(rows), combos

class GlobalDesignTests(unittest.TestCase):











    def test_global_metrics_and_ranking_are_lexicographic(self):
        phrase_results, _ = synthetic_phrase_results()
        per_user, global_summary = build_global_summary(
            phrase_results,
            reliability_floor=0.50,
        )
        ranked = rank_global_summary(global_summary)

        self.assertEqual(ranked.iloc[0]["combo_id"], "s06_e10")
        diagonal = global_summary.set_index("combo_id").loc["s06_e06"]
        self.assertEqual(diagonal["users_meeting_reliability_floor"], 2)
        self.assertEqual(diagonal["worst_user_completion_rate"], 0.5)
        self.assertEqual(diagonal["macro_phrase_completion_rate"], 0.75)
        self.assertEqual(per_user.groupby("combo_id").user_id.nunique().min(), 2)

class GlobalValidationTests(unittest.TestCase):
    def test_global_curves_and_stage_validation(self):
        phrase_results, combos = synthetic_phrase_results()
        per_user, global_summary = build_global_summary(
            phrase_results,
            reliability_floor=0.50,
        )
        curves = build_global_curve_points(phrase_results)

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            rows = []
            for combo in combos:
                for user_id in ["A", "B"]:
                    path = condition_path(
                        run_dir,
                        user_id,
                        combo["space_period_index"],
                        combo["enter_period_index"],
                        0,
                    )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    subset = phrase_results[
                        (phrase_results["combo_id"] == combo["combo_id"])
                        & (phrase_results["user_id"] == user_id)
                    ]
                    subset.to_csv(path, index=False)
                    rows.append(
                        {
                            "stage": "screen",
                            "user_id": user_id,
                            **combo,
                            "trial": 0,
                            "status": "completed",
                            "condition_origin": "test",
                            "condition_file": str(path.relative_to(run_dir)),
                        }
                    )
            manifest = pd.DataFrame(rows)
            validate_stage_outputs(
                phrase_results,
                per_user,
                global_summary,
                curves,
                manifest,
                users=["A", "B"],
                combos=combos,
                trials=[0],
                phrase_count=2,
            )
