import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from types import SimpleNamespace

from OneClick_Simulation.examples.text_simulation.run_config_sweep import (
    ADAPTIVE_SIGMA_MARGIN,
    BASELINE_CLOCK_PERIOD,
    BASELINE_NGRAM_MODEL_PATH,
    BASELINE_VOCABULARY_PATH,
    SIGMA_MARGIN_SWEEP_VALUES,
    SUMMARY_COLUMNS,
    algorithm_experiment_configs,
    baseline_config,
    main,
    run_config_sweep,
    sigma_margin_experiment_configs,
)


class ConfigSweepExecutionTests(unittest.TestCase):
    def test_sigma_margin_sweep_changes_only_adaptive_margin(self):
        configs = sigma_margin_experiment_configs()

        self.assertEqual(len(configs), 1 + len(SIGMA_MARGIN_SWEEP_VALUES))
        self.assertEqual(configs[0], baseline_config())
        self.assertEqual(
            tuple(config.sigma_margin for config in configs[1:]),
            SIGMA_MARGIN_SWEEP_VALUES,
        )
        self.assertEqual(
            [config.algorithm_condition for config in configs[1:]],
            [
                "adaptive_sigma_1_5",
                "adaptive_sigma_2_0",
                "adaptive_sigma_2_5",
                "adaptive_sigma_3_0",
                "adaptive_sigma_3_5",
            ],
        )
        for config in configs[1:]:
            self.assertEqual(config.clock_period, BASELINE_CLOCK_PERIOD)
            self.assertFalse(config.use_click_offset)
            self.assertEqual(config.delay_learning_mode, "enter_only")
            self.assertEqual(config.word_clock_mode, "adaptive")
            self.assertEqual(config.prediction_priority_mode, "alternating")

    def test_algorithm_conditions_include_isolated_and_combined_changes(self):
        configs = algorithm_experiment_configs()

        self.assertEqual(
            [config.algorithm_condition for config in configs],
            [
                "baseline",
                "enter_offset_compensation",
                "separate_space_enter_models",
                "adaptive_word_clocks",
                "combined_offset_separate_models",
            ],
        )
        config = configs[0]
        self.assertEqual(config.clock_period, BASELINE_CLOCK_PERIOD)
        self.assertFalse(config.use_click_offset)
        self.assertEqual(config.delay_learning_mode, "enter_only")
        self.assertEqual(config.word_clock_mode, "fixed")
        self.assertEqual(config.prediction_priority_mode, "legacy")
        self.assertIsNone(config.sigma_margin)

        offset, separate, adaptive, combined = configs[1:]
        self.assertTrue(offset.use_click_offset)
        self.assertEqual(offset.delay_learning_mode, "enter_only")
        self.assertEqual(offset.word_clock_mode, "fixed")
        self.assertFalse(separate.use_click_offset)
        self.assertEqual(separate.delay_learning_mode, "separate_space_enter")
        self.assertEqual(separate.word_clock_mode, "fixed")
        self.assertFalse(adaptive.use_click_offset)
        self.assertEqual(adaptive.delay_learning_mode, "enter_only")
        self.assertEqual(adaptive.word_clock_mode, "adaptive")
        self.assertEqual(adaptive.prediction_priority_mode, "alternating")
        self.assertEqual(adaptive.sigma_margin, ADAPTIVE_SIGMA_MARGIN)
        self.assertTrue(combined.use_click_offset)
        self.assertEqual(combined.delay_learning_mode, "separate_space_enter")
        self.assertEqual(combined.word_clock_mode, "fixed")
        self.assertEqual(combined.prediction_priority_mode, "legacy")
        self.assertIsNone(combined.sigma_margin)
        self.assertTrue(
            all(
                config.prediction_priority_mode == "legacy"
                for config in (configs[0], offset, separate, combined)
            )
        )
        self.assertTrue(
            all(config.clock_period == BASELINE_CLOCK_PERIOD for config in configs)
        )

    def test_baseline_cli_pins_ngram_inputs_and_writes_manifest(self):
        module = (
            "OneClick_Simulation.examples.text_simulation.run_config_sweep"
        )
        configs = [baseline_config()]
        with tempfile.TemporaryDirectory() as directory:
            with patch(
                f"{module}.run_config_sweep",
                return_value=configs,
            ) as sweep_runner:
                main(["--baseline", "--output-directory", directory])

            options = sweep_runner.call_args.kwargs
            self.assertFalse(options["dry_run"])
            self.assertIsNone(options["study_users"])
            self.assertIsNone(options["phrase_limit"])
            self.assertEqual(options["lm_backend"], "ngram")
            self.assertEqual(
                options["lm_model_path"],
                str(BASELINE_NGRAM_MODEL_PATH),
            )
            self.assertEqual(
                options["lm_vocabulary_path"],
                str(BASELINE_VOCABULARY_PATH),
            )
            record = json.loads(
                (Path(directory) / "baseline_config.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(record["configuration_id"], "config_001")
        self.assertEqual(record["algorithm_condition"], "baseline")
        self.assertEqual(record["clock_period"], BASELINE_CLOCK_PERIOD)
        self.assertEqual(record["lm_backend"], "ngram")
        self.assertEqual(record["maximum_word_clocks"], 8)
        self.assertEqual(record["study_users"], "A,B,C,D,F,G")
        self.assertEqual(record["synthetic_user"], "P")
        self.assertEqual(record["phrase_count"], 10)

    def test_sweep_writes_detailed_and_clean_summary_tables(self):
        configs = algorithm_experiment_configs()[:2]

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
                            "Num Clicks": [8, 4, 6, 2],
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
                    configs,
                    dry_run=False,
                    output_directory=temporary_directory,
                    language_model=SimpleNamespace(
                        recognizer_nbest=1000,
                        word_search_values={},
                    ),
                )

            self.assertEqual(len(configs), 2)
            prepare_inputs.assert_called_once_with(Path(temporary_directory).resolve())
            self.assertEqual(study_runner.call_count, 2)
            shared_models = {
                id(call.kwargs["language_model"])
                for call in study_runner.call_args_list
            }
            self.assertEqual(len(shared_models), 1)
            root = Path(temporary_directory)
            self.assertTrue((root / "sweep_config.csv").is_file())
            self.assertTrue((root / "sweep_report.md").is_file())
            report = (root / "sweep_report.md").read_text(encoding="utf-8")
            self.assertIn("QuickClick Algorithm Experiment Report", report)
            self.assertIn("## Cumulative real-user results", report)
            self.assertIn(
                "| Baseline | 62.50% | 1.500 | 9.00 s | 0.167 | 41.67% |",
                report,
            )
            self.assertIn("Algorithm-condition results", report)
            self.assertNotIn("Real-user factor averages", report)
            self.assertIn("Synthetic user P (kept separate)", report)
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
            self.assertEqual(first_a["Clicks per Character"], 1.0)
            self.assertEqual(first_a["Active Typing Time per Phrase"], 12.0)
            first_real_mean = config_summary[
                (config_summary["config_id"] == "config_001")
                & (config_summary["user_id"] == "MEAN_REAL_USERS")
            ].iloc[0]
            self.assertEqual(first_real_mean["Clicks per Character"], 1.5)
            self.assertEqual(first_real_mean["Active Typing Time per Phrase"], 9.0)
            self.assertFalse(any(path.is_dir() for path in root.iterdir()))

    def test_smoke_cli_uses_all_algorithm_conditions_on_one_phrase(self):
        module = (
            "OneClick_Simulation.examples.text_simulation.run_config_sweep"
        )
        with patch(f"{module}.run_config_sweep") as sweep_runner:
            main(
                [
                    "--smoke",
                    "--lm-model-path",
                    "/tmp/model",
                    "--lm-vocabulary-path",
                    "/tmp/vocabulary",
                ]
            )

        options = sweep_runner.call_args.kwargs
        configs = options["configs"]
        self.assertEqual(len(configs), 5)
        self.assertEqual(options["study_users"], ("A",))
        self.assertEqual(options["phrase_limit"], 1)
        self.assertFalse(options["dry_run"])
        self.assertEqual(options["lm_backend"], "ngram")
        self.assertEqual(options["lm_vocabulary_path"], "/tmp/vocabulary")

    def test_run_cli_uses_full_algorithm_study_scope(self):
        module = (
            "OneClick_Simulation.examples.text_simulation.run_config_sweep"
        )
        with patch(f"{module}.run_config_sweep") as sweep_runner:
            main(["--run", "--lm-model-path", "/tmp/model"])

        options = sweep_runner.call_args.kwargs
        configs = options["configs"]
        self.assertEqual(len(configs), 5)
        self.assertIsNone(options["study_users"])
        self.assertIsNone(options["phrase_limit"])
        self.assertFalse(options["dry_run"])
        self.assertEqual(options["lm_backend"], "ngram")

    def test_sigma_sweep_cli_uses_all_users_and_complete_corpus(self):
        module = (
            "OneClick_Simulation.examples.text_simulation.run_config_sweep"
        )
        with patch(f"{module}.run_config_sweep") as sweep_runner:
            main(["--sigma-sweep", "--lm-model-path", "/tmp/model"])

        options = sweep_runner.call_args.kwargs
        configs = options["configs"]
        self.assertEqual(len(configs), 6)
        self.assertEqual(
            tuple(config.sigma_margin for config in configs[1:]),
            SIGMA_MARGIN_SWEEP_VALUES,
        )
        self.assertIsNone(options["study_users"])
        self.assertIsNone(options["phrase_limit"])
        self.assertFalse(options["dry_run"])
        self.assertIn("sigma-margin-study-", options["output_directory"])


if __name__ == "__main__":
    unittest.main()
