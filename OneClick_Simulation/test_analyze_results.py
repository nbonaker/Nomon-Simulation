import tempfile
import unittest
from pathlib import Path

import pandas as pd

from OneClick_Simulation.examples.text_simulation.analyze_results import (
    analyze_results,
    summarize_all_users,
    summarize_failure_reasons,
    summarize_users,
)
from OneClick_Simulation.examples.text_simulation.run_full_study import (
    FIXED_IV_PHRASE_COUNT,
    REAL_USERS,
    audit_click_stream_result,
    block_bootstrap_click_stream,
    bootstrap_seed_for_user,
    build_summary,
    corpus_maximum_click_budget,
    load_fixed_iv_phrases,
    load_real_user_clicks,
    preflight_click_stream,
)


def phrase_row(user, **overrides):
    row = {
        "Configuration": "test",
        "user_id": user,
        "Num Clicks": 10,
        "Num Selections": 5,
        "Successful Word Click Count": 8,
        "Successful Word Selection Count": 4,
        "Successful Word Character Count": 16,
        "Target Word Count": 2,
        "Completed Word Count": 2,
        "Failed Word Count": 0,
        "Corrective Undo Action Count": 1,
        "Enter Press Count": 4,
        "Enter Misselection Count": 1,
        "Prediction Selection Count": 2,
        "Prefix Prediction Selection Count": 1,
        "Best Prediction Selection Count": 1,
        "Argmax Prediction Selection Count": 0,
        "Prediction Selection Events": "[]",
        "Active Typing Time (s)": 12.0,
        "Phrase Completed": True,
        "Phrase Failure Reason": "",
        "Failure Reason Counts": "{}",
        "Failure Events": "[]",
    }
    row.update(overrides)
    return row


class AnalysisMetricTests(unittest.TestCase):
    def test_real_users_start_without_picture_task_calibration_rows(self):
        for user_id in REAL_USERS:
            clicks = load_real_user_clicks(user_id)
            self.assertFalse(clicks["Session Num"].isna().any(), user_id)
            self.assertEqual(
                len(clicks),
                int(clicks["Original Session Num"].notna().sum()),
                user_id,
            )

    def test_fixed_corpus_uses_configured_existing_iv_labels_in_order(self):
        phrases = load_fixed_iv_phrases()

        self.assertEqual(len(phrases), FIXED_IV_PHRASE_COUNT)
        self.assertEqual(phrases.iloc[0]["Comparison Phrase ID"], "mt_iv1")
        self.assertEqual(phrases["Phrase Type"].unique().tolist(), ["iv"])

    def test_click_stream_preflight_and_runtime_exhaustion_location(self):
        phrases = load_fixed_iv_phrases()
        click_df = pd.DataFrame({"Session Num": [1.0]})
        report = preflight_click_stream("A", click_df, phrases)

        self.assertFalse(report["Minimum Sufficiency Passed"])

        partial = pd.DataFrame(
            [
                {
                    "Comparison Phrase ID": "mt_iv1",
                    "Num Clicks": 100,
                    "Failure Events": (
                        '[{"reason": "click_stream_exhausted_letters"}]'
                    ),
                }
            ]
        )
        audited = audit_click_stream_result(report, partial, phrases)

        self.assertTrue(audited["Click Stream Exhausted"])
        self.assertEqual(audited["Exhaustion Phrase ID"], "mt_iv1")
        self.assertEqual(audited["First Unattempted Phrase ID"], "mt_iv2")

    def test_block_bootstrap_is_deterministic_and_stays_within_sessions(self):
        calibration = pd.DataFrame(
            {
                "Session Num": [float("nan")],
                "Click Time Relative (s)": [0.0],
            }
        )
        text = pd.DataFrame(
            {
                "Session Num": [1.0] * 6 + [2.0] * 6,
                "Click Time Relative (s)": list(range(6)) + list(range(10, 16)),
                "Original Session Num": [1.0] * 6 + [2.0] * 6,
                "Bootstrap Source Row": list(range(12)),
            }
        )
        clicks = pd.concat([calibration, text], ignore_index=True, sort=False)

        first = block_bootstrap_click_stream(
            clicks, required_clicks=25, block_size=3, seed=42
        )
        second = block_bootstrap_click_stream(
            clicks, required_clicks=25, block_size=3, seed=42
        )

        pd.testing.assert_frame_equal(first, second)
        schedule = first[first["Session Num"].notna()]
        self.assertEqual(len(schedule), 25)
        self.assertEqual(int(first["Session Num"].isna().sum()), 1)
        self.assertEqual(schedule["Bootstrap Schedule Row"].tolist(), list(range(25)))
        for _, block in schedule.groupby("Bootstrap Block ID", sort=False):
            self.assertEqual(block["Original Session Num"].nunique(), 1)
            self.assertTrue(
                (block["Bootstrap Source Row"].diff().dropna() == 1).all()
            )
            self.assertEqual(
                block["Bootstrap Position in Block"].tolist(),
                list(range(len(block))),
            )

    def test_bootstrap_fills_full_fixed_corpus_worst_case_budget(self):
        phrases = load_fixed_iv_phrases()
        required = corpus_maximum_click_budget(phrases)
        source = pd.DataFrame(
            {
                "Session Num": [1.0] * 25,
                "Click Time Relative (s)": list(range(25)),
            }
        )

        schedule = block_bootstrap_click_stream(
            source,
            required_clicks=required,
            block_size=20,
            seed=bootstrap_seed_for_user("A"),
        )
        report = preflight_click_stream("A", schedule, phrases)

        self.assertEqual(
            required,
            int(phrases["Phrase Text"].str.split().str.len().sum()) * 30,
        )
        self.assertEqual(report["Available Click Rows"], required)
        self.assertTrue(report["Full Corpus Guaranteed"])

    def test_user_metrics_are_ratios_of_raw_totals(self):
        raw = pd.DataFrame(
            [
                phrase_row("A"),
                phrase_row(
                    "A",
                    **{
                        "Successful Word Click Count": 4,
                        "Successful Word Selection Count": 2,
                        "Successful Word Character Count": 8,
                        "Target Word Count": 1,
                        "Completed Word Count": 1,
                        "Corrective Undo Action Count": 0,
                        "Enter Press Count": 2,
                        "Enter Misselection Count": 1,
                        "Active Typing Time (s)": 8.0,
                    },
                ),
            ]
        )

        row = summarize_users(raw).iloc[0]

        self.assertNotIn("Click Burden (clicks/successful selection)", row.index)
        self.assertEqual(row["Clicks per Character"], 0.5)
        self.assertEqual(row["Active Typing Time (s/phrase)"], 10.0)
        self.assertAlmostEqual(
            row["Correction Rate (undoes/successful word)"], 1 / 3
        )
        self.assertAlmostEqual(row["Enter Misselection Rate"], 2 / 6)
        self.assertEqual(row["Completion Rate"], 1.0)
        self.assertEqual(row["Prefix Prediction Usage (%)"], 50.0)
        self.assertEqual(row["Best Prediction Usage (%)"], 50.0)
        self.assertEqual(row["Argmax Prediction Usage (%)"], 0.0)

    def test_aggregate_macro_averages_users_and_sums_counts(self):
        raw = pd.DataFrame(
            [
                phrase_row("A"),
                phrase_row(
                    "B",
                    **{
                        "Successful Word Click Count": 3,
                        "Successful Word Selection Count": 1,
                        "Successful Word Character Count": 6,
                        "Target Word Count": 2,
                        "Completed Word Count": 1,
                        "Failed Word Count": 1,
                        "Corrective Undo Action Count": 0,
                        "Enter Press Count": 2,
                        "Enter Misselection Count": 1,
                        "Active Typing Time (s)": 6.0,
                        "Phrase Completed": False,
                        "Phrase Failure Reason": "target_not_displayed",
                        "Failure Reason Counts": '{"target_not_displayed": 1}',
                        "Failure Events": (
                            '[{"reason": "target_not_displayed", '
                            '"stage": "word_prediction", '
                            '"limit": "max_word_attempts", '
                            '"guard": "word_attempts_exhausted"}]'
                        ),
                    },
                ),
            ]
        )
        per_user = summarize_users(raw)

        aggregate = summarize_all_users(per_user).iloc[0]

        self.assertEqual(aggregate["Users Included"], 2)
        self.assertEqual(aggregate["Completed Word Count"], 3)
        self.assertNotIn(
            "Click Burden (clicks/successful selection)", aggregate.index
        )
        self.assertEqual(aggregate["Clicks per Character"], 0.5)
        self.assertEqual(aggregate["Completion Rate"], 0.75)

    def test_structured_failure_counts_preserve_multiple_failures(self):
        raw = pd.DataFrame(
            [
                phrase_row(
                    "A",
                    **{
                        "Phrase Completed": False,
                        "Phrase Failure Reason": "target_not_displayed",
                        "Failure Reason Counts": (
                            '{"target_not_displayed": 2, '
                            '"word_click_budget_letters": 1}'
                        ),
                        "Failure Events": (
                            '[{"reason": "target_not_displayed"}, '
                            '{"reason": "target_not_displayed"}, '
                            '{"reason": "word_click_budget_letters"}]'
                        ),
                    },
                )
            ]
        )

        failures = summarize_failure_reasons(raw).set_index("Failure Reason")

        self.assertEqual(failures.loc["target_not_displayed", "Failure Count"], 2)
        self.assertEqual(failures.loc["word_click_budget_letters", "Failure Count"], 1)

    def test_full_study_mean_keeps_perfect_user_separate(self):
        raw = pd.DataFrame(
            [
                phrase_row("A"),
                phrase_row(
                    "B",
                    **{
                        "Successful Word Click Count": 3,
                        "Successful Word Selection Count": 1,
                        "Successful Word Character Count": 6,
                    },
                ),
                phrase_row(
                    "P",
                    **{
                        "Successful Word Click Count": 100,
                        "Successful Word Selection Count": 1,
                    },
                ),
            ]
        )
        per_user = summarize_users(raw)

        summary = build_summary(per_user.to_dict("records")).set_index("User")

        self.assertIn("P", summary.index)
        self.assertIn("MEAN_REAL_USERS", summary.index)
        self.assertNotIn(
            "Click Burden (clicks/successful selection)", summary.columns
        )
        self.assertEqual(summary.loc["MEAN_REAL_USERS", "Clicks per Character"], 0.5)

    def test_analysis_writes_all_output_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp)
            pd.DataFrame([phrase_row("A")]).drop(
                columns=["Configuration"]
            ).to_csv(results_dir / "user_A.csv", index=False)

            per_user, aggregate, failures = analyze_results(results_dir)

            self.assertEqual(len(per_user), 1)
            self.assertEqual(len(aggregate), 1)
            self.assertEqual(len(failures), 0)
            self.assertTrue((results_dir / "analysis" / "metrics_per_user.csv").is_file())
            self.assertTrue((results_dir / "analysis" / "metrics_aggregate.csv").is_file())
            self.assertTrue((results_dir / "analysis" / "failure_reasons.csv").is_file())


if __name__ == "__main__":
    unittest.main()
