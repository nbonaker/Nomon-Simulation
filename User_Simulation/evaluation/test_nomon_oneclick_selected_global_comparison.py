from __future__ import annotations

import unittest
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from Nomon_Text.imagineville_lm import ImaginevilleLM
from OneClick_Text.language_model import LanguageModel
from User_Simulation.evaluation.evaluation_nomon_oneclick_selected_global_comparison import (
    build_curve_points,
    build_paired_results,
    run_comparison,
    select_heldout_phrases,
    validate_condition,
)
from User_Simulation.simulated_user_text import SimulatedUser


def _audit_row(phrase_id: str, length: int) -> dict:
    return {
        "phrase_id": phrase_id,
        "phrase_text": f"phrase {phrase_id}",
        "all_words_prediction_reachable": True,
        "target_character_count": length,
    }


class HeldoutPhraseTests(unittest.TestCase):
    def test_selection_is_deterministic_and_excludes_tuning_phrases(self):
        audit = pd.DataFrame(
            [_audit_row(f"real_{index:03d}", index) for index in range(1, 9)]
        )
        tuning = pd.DataFrame({"phrase_id": ["real_002", "real_006"]})

        first = select_heldout_phrases(audit, tuning, phrase_count=3, seed=54321)
        second = select_heldout_phrases(audit, tuning, phrase_count=3, seed=54321)

        self.assertTrue(first.equals(second))
        self.assertFalse(set(first["phrase_id"]) & set(tuning["phrase_id"]))
        self.assertEqual(
            first["Comparison Phrase ID"].tolist(),
            ["heldout_phrase_01", "heldout_phrase_02", "heldout_phrase_03"],
        )


class CompletionSummaryTests(unittest.TestCase):
    def test_failed_phrases_remain_in_curve_denominator(self):
        frame = pd.DataFrame(
            [
                {
                    "user_id": "A",
                    "system": system,
                    "system_label": label,
                    "phrase_completed": True,
                    "simulated_completion_time_s": 30.0,
                }
                for system, label in [
                    ("og_nomon", "OG Nomon"),
                    ("oneclick", "OneClick"),
                ]
            ]
            + [
                {
                    "user_id": "A",
                    "system": system,
                    "system_label": label,
                    "phrase_completed": False,
                    "simulated_completion_time_s": np.nan,
                }
                for system, label in [
                    ("og_nomon", "OG Nomon"),
                    ("oneclick", "OneClick"),
                ]
            ]
        )

        curves = build_curve_points(frame)
        endpoints = curves[
            (curves["scope_user_id"] == "A")
            & (curves["event_type"] == "endpoint")
        ]
        self.assertTrue(endpoints["phrase_attempts"].eq(2).all())
        self.assertTrue(endpoints["cumulative_completion_rate"].eq(0.5).all())

    def test_paired_metrics_only_use_mutually_completed_trials(self):
        rows = []
        for system in ["og_nomon", "oneclick"]:
            rows.extend(
                [
                    {
                        "user_id": "A",
                        "trial": 0,
                        "system": system,
                        "Comparison Phrase ID": "p1",
                        "Target Phrase": "one",
                        "paired_click_schedule_id": "A_trial_00",
                        "paired_offset_checksum": "same",
                        "phrase_completed": True,
                        "num_clicks": 20 if system == "og_nomon" else 10,
                        "num_corrections": 0,
                        "correction_rate_percent": 0.0,
                        "simulated_attempt_time_s": (
                            50.0 if system == "og_nomon" else 40.0
                        ),
                        "simulated_completion_time_s": (
                            50.0 if system == "og_nomon" else 40.0
                        ),
                        "phrase_failure_reason": "",
                    },
                    {
                        "user_id": "A",
                        "trial": 0,
                        "system": system,
                        "Comparison Phrase ID": "p2",
                        "Target Phrase": "two",
                        "paired_click_schedule_id": "A_trial_00",
                        "paired_offset_checksum": "same",
                        "phrase_completed": system == "og_nomon",
                        "num_clicks": 30 if system == "og_nomon" else 5,
                        "num_corrections": 0,
                        "correction_rate_percent": 0.0,
                        "simulated_attempt_time_s": 60.0,
                        "simulated_completion_time_s": (
                            60.0 if system == "og_nomon" else np.nan
                        ),
                        "phrase_failure_reason": (
                            ""
                            if system == "og_nomon"
                            else "word_attempts_exhausted"
                        ),
                    },
                ]
            )

        paired = build_paired_results(pd.DataFrame(rows))

        self.assertEqual(
            paired["paired_outcome"].tolist(),
            ["both_completed", "og_only"],
        )
        self.assertAlmostEqual(paired.loc[0, "paired_click_reduction_percent"], 50.0)
        self.assertTrue(np.isnan(paired.loc[1, "paired_click_reduction_percent"]))


class ConditionValidationTests(unittest.TestCase):
    def test_exact_text_and_oneclick_time_accounting_are_enforced(self):
        frame = pd.DataFrame(
            [
                {
                    "user_id": "A",
                    "trial": 0,
                    "system": "oneclick",
                    "Comparison Phrase ID": "p1",
                    "Target Phrase": "exact phrase",
                    "Typed Text": "exact phrase ",
                    "phrase_completed": True,
                    "simulated_attempt_time_s": 6.0,
                    "simulated_completion_time_s": 6.0,
                    "paired_click_schedule_id": "A_trial_00",
                    "paired_offset_checksum": "checksum",
                    "space_clock_period_s": 6.0,
                    "enter_clock_period_s": 6.0,
                    "letter_clock_time_s": 2.0,
                    "target_enter_clock_time_s": 3.0,
                    "undo_clock_time_s": 1.0,
                }
            ]
        )
        validate_condition(frame, "A", "oneclick", 0, {"p1"})

        frame.loc[0, "Typed Text"] = "wrong phrase"
        with self.assertRaisesRegex(ValueError, "exactly match"):
            validate_condition(frame, "A", "oneclick", 0, {"p1"})


class LanguageModelReliabilityTests(unittest.TestCase):
    def test_oneclick_strict_mode_raises_instead_of_falling_back(self):
        strict = LanguageModel({"strict_errors": True})
        with patch.object(strict.session, "get", side_effect=OSError("offline")):
            with self.assertRaisesRegex(RuntimeError, "character language-model"):
                strict.get_key_probs("context")

        permissive = LanguageModel()
        with patch.object(permissive.session, "get", side_effect=OSError("offline")):
            self.assertEqual(
                len(permissive.get_key_probs("context")),
                len(permissive.key_chars),
            )

    def test_imagineville_client_retries_transient_requests(self):
        lm = ImaginevilleLM()
        self.assertEqual(lm.session.get_adapter("https://").max_retries.total, 3)


class OldNomonTimingTests(unittest.TestCase):
    def test_phrase_results_export_attempt_and_completion_time(self):
        simulated_user = SimulatedUser.__new__(SimulatedUser)
        simulated_user.keyboard = SimpleNamespace(
            typed="exact phrase",
            sim_time=SimpleNamespace(time=lambda: 10.0),
            lm=SimpleNamespace(),
        )
        simulated_user.start_time = 4.0
        simulated_user.num_selections_phrase = 1
        simulated_user.num_word_selections_phrase = 0
        simulated_user.num_clicks_phrase = 1
        simulated_user.num_corrections_phrase = 0

        result = simulated_user.calculate_phrase_results("exact phrase", "test")

        self.assertIs(result["Phrase Completed"], True)
        self.assertAlmostEqual(result["Simulated Attempt Time (s)"], 6.0)
        self.assertAlmostEqual(result["Simulated Completion Time (s)"], 6.0)

        simulated_user.keyboard.typed = "wrong"
        failed = simulated_user.calculate_phrase_results("exact phrase", "test")
        self.assertIs(failed["Phrase Completed"], False)
        self.assertTrue(np.isnan(failed["Simulated Completion Time (s)"]))


class RunnerIntegrationTests(unittest.TestCase):
    @staticmethod
    def _raw_results(phrase_df: pd.DataFrame, system: str) -> pd.DataFrame:
        rows = []
        for phrase in phrase_df.to_dict("records"):
            row = {
                "Comparison Phrase ID": phrase["Comparison Phrase ID"],
                "Target Phrase": phrase["Phrase Text"],
                "Typed Text": phrase["Phrase Text"],
                "Num Clicks": 10 if system == "og_nomon" else 6,
                "Num Corrections": 0,
                "Correction Rate (%)": 0.0,
                "Error Rate (%)": 0.0,
                "Phrase Completed": True,
                "Completion Fraction": 1.0,
                "Simulated Attempt Time (s)": 12.0 if system == "og_nomon" else 9.0,
                "Simulated Completion Time (s)": (
                    12.0 if system == "og_nomon" else 9.0
                ),
            }
            if system == "oneclick":
                row.update(
                    {
                        "Letter Clock Time (s)": 5.0,
                        "Target Enter Clock Time (s)": 4.0,
                        "Undo Clock Time (s)": 0.0,
                        "Simulated Time Accounting Error (s)": 0.0,
                    }
                )
            rows.append(row)
        return pd.DataFrame(rows)

    def test_mocked_run_writes_and_resumes_all_outputs(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            selection = root / "selection"
            selection.mkdir()
            tuning = pd.DataFrame(
                [
                    {"phrase_id": "tuning_1", "phrase_text": "tuning one"},
                    {"phrase_id": "tuning_2", "phrase_text": "tuning two"},
                ]
            )
            audit = pd.DataFrame(
                [
                    _audit_row("tuning_1", 10),
                    _audit_row("tuning_2", 11),
                    _audit_row("heldout_1", 12),
                    _audit_row("heldout_2", 13),
                    _audit_row("heldout_3", 14),
                ]
            )
            tuning.to_csv(selection / "common_phrase_set.csv", index=False)
            audit.to_csv(
                selection / "common_phrase_reachability_audit.csv",
                index=False,
            )
            pd.DataFrame(
                [
                    {"phrase_id": phrase_id, "target_word": "word"}
                    for phrase_id in audit["phrase_id"]
                ]
            ).to_csv(
                selection / "common_phrase_word_reachability_audit.csv",
                index=False,
            )
            (selection / "global_selection_summary.csv").write_text(
                "space_period_s,enter_period_s\n6.0,6.0\n",
                encoding="utf-8",
            )
            args = Namespace(
                users="A",
                trials=1,
                phrase_count=2,
                seed=54321,
                validation_fraction=0.2,
                clicks_per_phrase=5,
                calibration_clicks=2,
                max_word_attempts=5,
                max_enter_attempts=5,
                max_clicks_per_word=30,
                lm_backend="imagineville",
                lm_size="tiny",
                lm_cache_dir=root / "og_cache",
                oneclick_cache_dir=root / "oneclick_cache",
                selection_run_dir=selection,
                resume_run_dir=None,
                verbose=False,
                output_dir=root / "outputs",
            )

            def fake_og(**kwargs):
                return self._raw_results(kwargs["phrase_df"], "og_nomon")

            def fake_oneclick(**kwargs):
                self.assertEqual(kwargs["fixed_space_clock_period_s"], 6.0)
                self.assertEqual(kwargs["fixed_enter_clock_period_s"], 6.0)
                self.assertIs(kwargs["strict_lm_errors"], True)
                return self._raw_results(kwargs["phrase_df"], "oneclick")

            with patch(
                "User_Simulation.evaluation."
                "evaluation_nomon_oneclick_selected_global_comparison."
                "run_old_nomon_bootstrap",
                side_effect=fake_og,
            ) as old_runner, patch(
                "User_Simulation.evaluation."
                "evaluation_nomon_oneclick_selected_global_comparison.run_oneclick",
                side_effect=fake_oneclick,
            ) as oneclick_runner:
                run_dir = run_comparison(args)

            self.assertEqual(old_runner.call_count, 1)
            self.assertEqual(oneclick_runner.call_count, 1)
            self.assertEqual(len(pd.read_csv(run_dir / "manifest.csv")), 2)
            self.assertEqual(
                len(pd.read_csv(run_dir / "system_phrase_results.csv")),
                4,
            )
            self.assertTrue(
                (run_dir / "plots" / "comparison_dashboard.png").is_file()
            )
            self.assertTrue(
                (run_dir / "completion_by_time_curve_points.csv").is_file()
            )

            args.resume_run_dir = run_dir
            with patch(
                "User_Simulation.evaluation."
                "evaluation_nomon_oneclick_selected_global_comparison."
                "run_old_nomon_bootstrap"
            ) as resumed_old, patch(
                "User_Simulation.evaluation."
                "evaluation_nomon_oneclick_selected_global_comparison.run_oneclick"
            ) as resumed_oneclick:
                resumed = run_comparison(args)
            self.assertEqual(resumed.resolve(), run_dir.resolve())
            resumed_old.assert_not_called()
            resumed_oneclick.assert_not_called()


if __name__ == "__main__":
    unittest.main()
