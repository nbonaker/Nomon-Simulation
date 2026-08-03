"""Tests for the global OneClick Space/Enter heatmap experiment."""

from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from User_Simulation.evaluation.evaluation_oneclick_clock_speed_tradeoff import (
    phrase_set_checksum,
)
from User_Simulation.evaluation.evaluation_oneclick_global_space_enter_sweep import (
    DEFAULT_PERIOD_INDICES,
    build_global_curve_points,
    build_global_summary,
    build_grid_combos,
    freeze_shortlist,
    import_reusable_condition,
    rank_global_summary,
    run_global_sweep,
    validate_stage_outputs,
)
from User_Simulation.evaluation.evaluation_oneclick_space_enter_phase1 import (
    build_combo_records,
    condition_path,
)
from User_Simulation.evaluation.synthetic_profiles import (
    CLICK_OFFSET_COL,
    CLOCK_PERIOD_COL,
    DEAD_TIME_COL,
)


def synthetic_phrase_results() -> tuple[pd.DataFrame, list[dict]]:
    combos = build_combo_records(
        [6, 10],
        [(6, 6), (6, 10)],
    )
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
    def test_default_grid_has_36_unique_combinations(self):
        combos = build_grid_combos(list(DEFAULT_PERIOD_INDICES))

        self.assertEqual(len(combos), 36)
        self.assertEqual(len({combo["combo_id"] for combo in combos}), 36)
        self.assertEqual(sum(combo["is_diagonal"] for combo in combos), 6)
        self.assertEqual(36 * 6, 216)
        self.assertEqual(216 * 20, 4_320)
        self.assertEqual(5 * 6 * 5, 150)
        self.assertEqual(150 * 20, 3_000)

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

    def test_frozen_shortlist_cannot_change_after_creation(self):
        summary = pd.DataFrame(
            [
                {
                    "combo_id": "a",
                    "global_rank": 1,
                    "space_period_index": 0,
                    "enter_period_index": 0,
                },
                {
                    "combo_id": "b",
                    "global_rank": 2,
                    "space_period_index": 2,
                    "enter_period_index": 2,
                },
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            frozen = freeze_shortlist(run_dir, summary, 1, "checksum")
            self.assertEqual(frozen["combo_id"].tolist(), ["a"])

            changed = summary.iloc[::-1].reset_index(drop=True)
            with self.assertRaisesRegex(ValueError, "Frozen shortlist differs"):
                freeze_shortlist(run_dir, changed, 1, "checksum")


class GlobalReuseTests(unittest.TestCase):
    def test_legacy_diagonal_condition_is_normalized_and_reused(self):
        combo = build_combo_records([6], [(6, 6)])[0]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            phase1 = root / "phase1"
            phase2 = root / "phase2"
            baseline = root / "baseline"
            run_dir = root / "run"
            source = (
                baseline
                / "conditions"
                / "user_A"
                / "period_06"
                / "trial_00.csv"
            )
            source.parent.mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "user_id": "A",
                        "trial": 0,
                        "clock_period_index": 6,
                        "clock_period_s": combo["space_period_s"],
                        "Comparison Phrase ID": "p1",
                        "Target Phrase": "target",
                        "Typed Text": "target",
                        "phrase_completed": True,
                        "simulated_attempt_time_s": 1.0,
                        "simulated_completion_time_s": 1.0,
                        "letter_clock_time_s": 0.4,
                        "target_enter_clock_time_s": 0.6,
                        "undo_clock_time_s": 0.0,
                        "simulated_time_accounting_error_s": 0.0,
                    }
                ]
            ).to_csv(source, index=False)

            audit = import_reusable_condition(
                run_dir,
                phase1,
                phase2,
                baseline,
                "A",
                combo,
                0,
                ["p1"],
                {
                    "phase1": {},
                    "phase2": {},
                    "clock_speed_baseline": {"source": "baseline"},
                },
            )
            destination = condition_path(run_dir, "A", 6, 6, 0)
            imported = pd.read_csv(destination)

            self.assertEqual(audit["condition_origin"], "clock_speed_baseline")
            self.assertEqual(imported.loc[0, "combo_id"], "s06_e06")
            self.assertEqual(imported.loc[0, "space_period_index"], 6)
            self.assertEqual(imported.loc[0, "enter_period_index"], 6)


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


class GlobalRunnerIntegrationTests(unittest.TestCase):
    def test_mocked_all_stage_run_writes_screen_and_confirmation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            phase1 = root / "phase1"
            phase2 = root / "phase2"
            baseline = root / "baseline"
            output_dir = root / "outputs"
            for directory in [phase1, phase2, baseline]:
                directory.mkdir()
            phrase_set = pd.DataFrame(
                [
                    {
                        "Comparison Phrase ID": "p1",
                        "Session Num": 1,
                        "Phrase Num": 1,
                        "Phrase Text": "a",
                        "all_words_prediction_reachable": True,
                    }
                ]
            )
            phrase_set.to_csv(phase1 / "common_phrase_set.csv", index=False)
            checksum = phrase_set_checksum(phrase_set)
            compatible_config = {
                "phrase_set_checksum": checksum,
                "max_word_attempts": 5,
                "max_enter_attempts": 5,
                "max_clicks_per_word": 30,
                "undo_mode": "protected",
                "dead_time_mode": "zero_active_dead_time",
                "phrase_time_ceiling_s": None,
            }
            for directory in [phase1, phase2, baseline]:
                (directory / "run_config.json").write_text(
                    json.dumps(compatible_config),
                    encoding="utf-8",
                )
            for user_id in ["A", "B"]:
                for trial in range(2):
                    schedule = (
                        baseline
                        / "paired_click_schedules"
                        / f"user_{user_id}_trial_{trial:02d}.csv"
                    )
                    schedule.parent.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame(
                        {
                            "Session Num": [np.nan, 1],
                            CLOCK_PERIOD_COL: [2.2, 2.2],
                            CLICK_OFFSET_COL: [0.0, 0.01 * trial],
                            DEAD_TIME_COL: [np.nan, 0.0],
                        }
                    ).to_csv(schedule, index=False)

            raw_result = pd.DataFrame(
                [
                    {
                        "Comparison Phrase ID": "p1",
                        "Target Phrase": "a",
                        "Typed Text": "a",
                        "Phrase Completed": True,
                        "Completion Fraction": 1.0,
                        "Num Clicks": 2,
                        "Num Corrections": 0,
                        "Correction Rate (%)": 0.0,
                        "Error Rate (%)": 0.0,
                        "Simulated Attempt Time (s)": 1.0,
                        "Simulated Completion Time (s)": 1.0,
                        "Letter Clock Time (s)": 0.4,
                        "Target Enter Clock Time (s)": 0.6,
                        "Undo Clock Time (s)": 0.0,
                        "Simulated Time Accounting Error (s)": 0.0,
                    }
                ]
            )
            args = Namespace(
                phase="all",
                users="A,B",
                period_indices="6,10",
                combo_pairs="6:6,6:10,10:10",
                confirmation_trials=2,
                phrases=1,
                shortlist_size=3,
                reliability_floor=0.80,
                max_word_attempts=5,
                max_enter_attempts=5,
                max_clicks_per_word=30,
                seed=12345,
                oneclick_cache_dir=root / "cache",
                output_dir=output_dir,
                phase1_run_dir=phase1,
                phase2_run_dir=phase2,
                baseline_run_dir=baseline,
                resume_run_dir=None,
                verbose=False,
            )
            with patch(
                "User_Simulation.evaluation."
                "evaluation_oneclick_global_space_enter_sweep.run_oneclick",
                return_value=raw_result,
            ) as mocked_run:
                run_dir = run_global_sweep(args)

            screen = pd.read_csv(run_dir / "screen_phrase_results.csv")
            confirmed = pd.read_csv(run_dir / "confirmed_phrase_results.csv")
            shortlist = pd.read_csv(run_dir / "frozen_shortlist.csv")
            selection = pd.read_csv(run_dir / "global_selection_summary.csv")

            self.assertEqual(mocked_run.call_count, 12)
            self.assertEqual(len(screen), 6)
            self.assertEqual(len(confirmed), 12)
            self.assertEqual(len(shortlist), 3)
            self.assertEqual(len(selection), 1)
            self.assertTrue(
                pd.read_csv(run_dir / "screen_manifest.csv")[
                    "status"
                ].eq("completed").all()
            )
            self.assertTrue(
                pd.read_csv(run_dir / "confirmation_manifest.csv")[
                    "status"
                ].eq("completed").all()
            )


if __name__ == "__main__":
    unittest.main()
