import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from OneClick_Simulation.examples.text_simulation.run_config_sweep import (
    SUMMARY_COLUMNS,
    SweepValues,
    run_config_sweep,
)


class ConfigSweepExecutionTests(unittest.TestCase):
    def test_sweep_writes_detailed_and_clean_summary_tables(self):
        values = SweepValues(
            clock_period=(2.0,),
            use_click_offset=(False,),
            delay_learning_mode=("enter_only",),
            word_clock_mode=("fixed", "adaptive"),
            sigma_margin=(2.5,),
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            study_module = (
                "OneClick_Simulation.examples.text_simulation.run_full_study"
            )
            with patch(f"{study_module}.prepare_study_inputs") as prepare_inputs, patch(
                f"{study_module}.run_full_study"
            ) as study_runner:
                prepare_inputs.return_value = {"shared": "inputs"}
                study_runner.return_value = {
                    "summary": pd.DataFrame(),
                    "user_results": pd.DataFrame(
                        {
                            "user_id": ["A", "A", "B", "P"],
                            "Successful Word Click Count": [4, 2, 3, 1],
                            "Successful Word Character Count": [8, 4, 3, 2],
                            "Active Typing Time (s)": [10.0, 14.0, 6.0, 4.0],
                            "Corrective Undo Action Count": [1, 0, 0, 0],
                            "Completed Word Count": [2, 1, 1, 1],
                            "Failed Word Count": [0, 1, 1, 0],
                            "Target Word Count": [2, 2, 2, 1],
                            "Enter Misselection Count": [1, 1, 1, 0],
                            "Enter Press Count": [4, 2, 2, 1],
                            "diagnostic": ["one", "two", "three", "four"],
                        }
                    ),
                }
                configs = run_config_sweep(
                    values,
                    dry_run=False,
                    output_directory=temporary_directory,
                )

            self.assertEqual(len(configs), 2)
            prepare_inputs.assert_called_once_with(Path(temporary_directory).resolve())
            self.assertEqual(study_runner.call_count, 2)
            root = Path(temporary_directory)
            self.assertTrue((root / "sweep_config.csv").is_file())
            self.assertFalse((root / "all_results.csv").exists())
            results = pd.read_csv(root / "phrase_results.csv")
            self.assertIn("diagnostic", results.columns)
            self.assertEqual(len(results), 8)
            user_summary = pd.read_csv(root / "summary_by_user_config.csv")
            config_summary = pd.read_csv(root / "summary_by_config.csv")
            self.assertEqual(user_summary.columns.tolist(), SUMMARY_COLUMNS)
            self.assertEqual(config_summary.columns.tolist(), SUMMARY_COLUMNS)
            self.assertEqual(len(user_summary), 6)
            self.assertEqual(len(config_summary), 4)
            first_a = user_summary[
                (user_summary["config_id"] == "config_001")
                & (user_summary["user_id"] == "A")
            ].iloc[0]
            self.assertEqual(first_a["Clicks per Character"], 0.5)
            self.assertEqual(first_a["Active Typing Time per Phrase"], 12.0)
            first_real_mean = config_summary[
                (config_summary["config_id"] == "config_001")
                & (config_summary["user_id"] == "MEAN_REAL_USERS")
            ].iloc[0]
            self.assertEqual(first_real_mean["Clicks per Character"], 0.75)
            self.assertEqual(first_real_mean["Active Typing Time per Phrase"], 9.0)
            self.assertFalse(any(path.is_dir() for path in root.iterdir()))


if __name__ == "__main__":
    unittest.main()
