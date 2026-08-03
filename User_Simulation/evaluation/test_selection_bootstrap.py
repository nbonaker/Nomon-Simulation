"""Focused tests for regime-aware selection bootstrap behavior."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from User_Simulation.evaluation.evaluation_baseline import load_text_click_data
from User_Simulation.evaluation.synthetic_profiles import (
    generate_synthetic_click_df,
    infer_stable_regime,
)
from User_Simulation.simulated_user_text import ClickUtil


def bootstrap_click_df() -> pd.DataFrame:
    rows = []
    for group_id, selection_type, offsets in [
        (1, "character", [0.01, 0.02]),
        (2, "character", [0.11, 0.12, 0.13]),
        (3, "word_prediction", [0.21, 0.22]),
    ]:
        for click_num, offset in enumerate(offsets, start=1):
            rows.append(
                {
                    "Synthetic Group ID": group_id,
                    "Synthetic Group Click Num": click_num,
                    "Synthetic Selection Type": selection_type,
                    "Synthetic Dead Time Clipped": False,
                    "Click Time Relative (s)": offset,
                    "Clock Period (s)": 1.5,
                    "Dead Time (s)": 1.0,
                }
            )
    return pd.DataFrame(rows)


def parent() -> SimpleNamespace:
    return SimpleNamespace(
        trial_num=0,
        session_num=1,
        phrase_num=1,
        current_original_phrase_num=1,
        selection_bootstrap_events=[],
    )


class StableRegimeTests(unittest.TestCase):
    def test_repository_users_use_expected_stable_sessions(self):
        user_a = load_text_click_data("A")
        user_b = load_text_click_data("B")

        period_a, _, sessions_a = infer_stable_regime(user_a[user_a["Session Num"] <= 5])
        period_b, _, sessions_b = infer_stable_regime(user_b[user_b["Session Num"] <= 13])

        self.assertAlmostEqual(period_a, 2.207)
        self.assertEqual(sessions_a, [1, 2, 3, 4, 5])
        self.assertAlmostEqual(period_b, 1.48)
        self.assertEqual(sessions_b, [8, 9, 10, 11, 12, 13])

    def test_generator_preserves_complete_source_group_values(self):
        train_click_df = pd.DataFrame(
            {
                "Session Num": [1, 1, 1],
                "Phrase Num": [1, 1, 1],
                "Selection Num": [1, 1, 1],
                "Click Num": [1, 2, 3],
                "Selection": ["a", "a", "a"],
                "Click Time Relative (s)": [0.03, -0.04, 0.05],
                "Clock Period (s)": [1.5, 1.5, 1.5],
                "Dead Time (s)": [20.0, 1.0, 2.0],
            }
        )
        phrase_df = pd.DataFrame(
            {"Session Num": [2], "Phrase Num": [1], "Phrase Text": ["test"]}
        )
        profile = {
            "bootstrap_source_sessions": [1],
            "dead_time_clip_max_s": 10.0,
            "click_offset_mean_s": 0.0,
            "click_offset_sd_s": 0.0,
            "click_offset_clip_min_s": 0.0,
            "click_offset_clip_max_s": 0.0,
            "clock_period_mean_s": 1.5,
            "clock_period_sd_s": 0.0,
            "clock_period_min_s": 1.5,
            "clock_period_max_s": 1.5,
        }

        generated = generate_synthetic_click_df(
            profile,
            phrase_df,
            trial=0,
            rng=np.random.default_rng(3),
            clicks_per_phrase=1,
            calibration_clicks=1,
            train_click_df=train_click_df,
        )
        active = generated[generated["Session Num"].notna()].sort_values(
            "Synthetic Group Click Num"
        )

        self.assertEqual(len(active), 3)
        self.assertEqual(active["Click Time Relative (s)"].tolist(), [0.03, -0.04, 0.05])
        self.assertEqual(active["Clock Period (s)"].tolist(), [1.5, 1.5, 1.5])
        self.assertEqual(active["Dead Time (s)"].tolist(), [10.0, 1.0, 2.0])
        self.assertEqual(active["Synthetic Group Click Num"].tolist(), [1, 2, 3])


class ClickUtilSelectionBootstrapTests(unittest.TestCase):
    def make_click_util(self, seed: int = 7) -> ClickUtil:
        return ClickUtil(parent(), bootstrap_click_df(), [], "playthrough", seed=seed)

    def test_early_selection_discards_unused_group_clicks(self):
        click_util = self.make_click_util()
        click_util.begin_selection("character")
        first_click = click_util.sample()
        click_util.end_selection("correct")

        click_util.begin_selection("character")
        next_selection_click = click_util.sample()

        self.assertEqual(first_click["Synthetic Group Click Num"], 1)
        self.assertEqual(next_selection_click["Synthetic Group Click Num"], 1)
        self.assertTrue(click_util.parent.selection_bootstrap_events[0]["early_selection"])

    def test_exhausted_group_continues_with_same_type(self):
        click_util = self.make_click_util(seed=2)
        click_util.begin_selection("word_prediction")

        first = click_util.sample()
        second = click_util.sample()
        continuation = click_util.sample()
        click_util.end_selection("correct")

        self.assertEqual([first["Synthetic Group Click Num"], second["Synthetic Group Click Num"]], [1, 2])
        self.assertEqual(continuation["Synthetic Group Click Num"], 1)
        event = click_util.parent.selection_bootstrap_events[0]
        self.assertEqual(event["continuation_group_count"], 1)
        self.assertEqual(event["fallback_group_count"], 0)

    def test_same_seed_reproduces_group_sequence(self):
        first_util = self.make_click_util(seed=11)
        second_util = self.make_click_util(seed=11)

        sequences = []
        for click_util in [first_util, second_util]:
            group_ids = []
            for _ in range(6):
                click_util.begin_selection("character")
                group_ids.append(int(click_util.sample()["Synthetic Group ID"]))
                click_util.end_selection("correct")
            sequences.append(group_ids)

        self.assertEqual(sequences[0], sequences[1])

    def test_ordered_replay_does_not_borrow_from_next_group(self):
        replay_df = bootstrap_click_df().copy()
        replay_df = replay_df[replay_df["Synthetic Selection Type"] == "character"].copy()
        replay_df["Session Num"] = 1
        replay_df["Phrase Num"] = 1
        replay_df["Synthetic Sampling Mode"] = "ordered_replay"
        replay_df["Synthetic Replay Sequence"] = replay_df["Synthetic Group ID"]
        replay_df["Synthetic Source Selection"] = "a"
        replay_df["Synthetic Source Session Num"] = 1
        replay_df["Synthetic Source Phrase Num"] = 1
        replay_df["Synthetic Source Selection Num"] = replay_df["Synthetic Group ID"]

        click_util = ClickUtil(parent(), replay_df, [], "playthrough", seed=3)
        click_util.begin_selection("character", target_text="a")
        first_group_id = int(click_util.sample()["Synthetic Group ID"])
        while click_util.sample() is not None:
            pass
        click_util.end_selection("no_selection")

        click_util.begin_selection("character", target_text="b")
        second_group_first_click = click_util.sample()

        self.assertEqual(first_group_id, 1)
        self.assertEqual(int(second_group_first_click["Synthetic Group ID"]), 2)
        self.assertEqual(int(second_group_first_click["Synthetic Group Click Num"]), 1)
        first_event = click_util.parent.selection_bootstrap_events[0]
        self.assertEqual(first_event["continuation_group_count"], 0)
        self.assertEqual(first_event["exhausted_group_count"], 1)


if __name__ == "__main__":
    unittest.main()
