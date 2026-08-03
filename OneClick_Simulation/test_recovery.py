import math
import unittest
from types import SimpleNamespace

from OneClick_Simulation.simulated_user import SimulatedUser
from OneClick_Text import kconfig
from OneClick_Text.keyboard import Keyboard


class FakeKeyboard:
    target_index = 7

    def __init__(self):
        self.typed = ""
        self.context = ""
        self.typed_versions = []
        self.restore_count = 0
        self.reset_count = 0
        self.prepare_undo_count = 0
        self.sim_time = SimpleNamespace(time=lambda: 0.0)

    def update_word_list(self):
        pass

    def prepare_undo_round(self, undo_only=False):
        self.prepare_undo_count += 1
        self.update_word_list()

    def word_clock_index(self, _target_word):
        return self.target_index

    def capture_word_attempt_state(self):
        return (self.typed, self.context, list(self.typed_versions))

    def restore_word_attempt_state(self, snapshot):
        self.typed, self.context, versions = snapshot
        self.typed_versions = list(versions)
        self.restore_count += 1

    def commit_word(self, word, confirmed_correct=False):
        self.typed_versions.append(self.typed)
        self.typed += word + " "
        self.context = self.typed

    def undo_word(self):
        self.typed = self.typed_versions.pop()
        self.context = self.typed

    def _reset_letter_round(self):
        self.reset_count += 1


def recovery_sim(enter_results):
    sim = SimulatedUser()
    sim.keyboard = FakeKeyboard()
    sim.click_util = SimpleNamespace(clicks_remaining=100)
    sim.max_word_attempts = 5
    sim.max_enter_attempts = 5
    sim.max_clicks_per_word = 30
    sim.undo_mode = "protected"
    sim.stop_phrase_on_failed_word = True
    sim.perfect_letter_observations = False
    sim._clear_phrase_tracking()
    sim.letter_press_count = 0
    results = iter(enter_results)

    def press_letter(_index):
        sim.letter_press_count += 1
        sim.num_clicks_phrase += 1
        sim.num_letter_presses_phrase += 1
        return True

    def press_enter(_index, action_kind="target_enter"):
        sim.num_clicks_phrase += 1
        return next(results)

    sim._press_letter = press_letter
    sim._press_enter = press_enter
    return sim


class WordAttemptSnapshotTests(unittest.TestCase):
    def test_undo_only_round_removes_prediction_competitors(self):
        initialized = []
        keyboard = Keyboard.__new__(Keyboard)
        keyboard.words_by_letter = {"a": ["another"]}
        keyboard.best_words = ["word"]
        keyboard.argmax_word = "wrong"
        keyboard.valid_word_indices = [1, 2, kconfig.undo_word_index]
        keyboard.sim_time = SimpleNamespace(time=lambda: 4.0)
        keyboard.word_clock_util = SimpleNamespace(
            init_round=lambda indices: initialized.append(list(indices)),
            latest_time=0.0,
        )

        keyboard.prepare_undo_round(undo_only=True)

        self.assertEqual(keyboard.words_by_letter, {})
        self.assertEqual(keyboard.best_words, [])
        self.assertEqual(keyboard.argmax_word, "")
        self.assertEqual(keyboard.valid_word_indices, [kconfig.undo_word_index])
        self.assertEqual(initialized, [[kconfig.undo_word_index]])
        self.assertEqual(keyboard.word_clock_util.latest_time, 4.0)

    def test_snapshot_round_trip_is_deep_copied(self):
        keyboard = Keyboard.__new__(Keyboard)
        keyboard.bc = SimpleNamespace(
            clock_inf=SimpleNamespace(observations=[[1.0, 2.0], [3.0, 4.0]])
        )
        keyboard.words_by_letter = {"c": ["cat", "car"]}
        keyboard.best_words = ["ca"]
        keyboard.argmax_word = "ca"
        keyboard.valid_word_indices = [1, 2, kconfig.undo_word_index]
        keyboard.word_clock_util = SimpleNamespace(
            cur_hours={1: 10, 2: 20}, latest_time=0.0
        )
        keyboard.sim_time = SimpleNamespace(time=lambda: 12.5)
        keyboard.typed = "the "
        keyboard.context = "the "
        keyboard.typed_versions = [""]
        keyboard._last_enter_time_in = 0.1
        keyboard._last_commit_updated_delay = True

        snapshot = keyboard.capture_word_attempt_state()
        keyboard.bc.clock_inf.observations[0][0] = 99.0
        keyboard.words_by_letter["c"].append("can")
        keyboard.word_clock_util.cur_hours[1] = 99
        keyboard.typed_versions.append("changed")

        keyboard.restore_word_attempt_state(snapshot)

        self.assertEqual(keyboard.bc.clock_inf.observations, [[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(keyboard.words_by_letter, {"c": ["cat", "car"]})
        self.assertEqual(keyboard.word_clock_util.cur_hours, {1: 10, 2: 20})
        self.assertEqual(keyboard.word_clock_util.latest_time, 12.5)
        self.assertEqual(keyboard.typed_versions, [""])
        self.assertIsNone(keyboard._last_enter_time_in)
        self.assertFalse(keyboard._last_commit_updated_delay)

        snapshot.observations[0][0] = -1.0
        snapshot.words_by_letter["c"].append("copy")
        self.assertEqual(keyboard.bc.clock_inf.observations[0][0], 1.0)
        self.assertEqual(keyboard.words_by_letter["c"], ["cat", "car"])

    def test_only_confirmed_correct_commit_updates_delay_model(self):
        updates = []
        delay_model = SimpleNamespace(
            update=lambda value: updates.append(value),
            rollback=lambda: updates.append("rollback"),
        )
        keyboard = Keyboard.__new__(Keyboard)
        keyboard.typed = ""
        keyboard.context = ""
        keyboard.typed_versions = []
        keyboard.bc = SimpleNamespace(clock_inf=SimpleNamespace(delay_model=delay_model))
        keyboard._reset_letter_round = lambda: None
        keyboard._last_commit_updated_delay = False

        keyboard._last_enter_time_in = 0.25
        keyboard.commit_word("wrong", confirmed_correct=False)
        self.assertEqual(updates, [])

        keyboard._last_enter_time_in = 0.15
        keyboard.commit_word("right", confirmed_correct=True)
        self.assertEqual(updates, [0.15])


class RecoveryFlowTests(unittest.TestCase):
    def test_trailing_space_does_not_hide_missing_final_letter(self):
        sim = recovery_sim([])
        sim.keyboard.typed = "i'm getting one for mya no "

        phrase_result = sim._calculate_phrase_results(
            "i'm getting one for mya now",
            "?",
        )

        self.assertFalse(phrase_result["Phrase Completed"])
        self.assertTrue(math.isnan(phrase_result["Simulated Completion Time (s)"]))
        self.assertEqual(phrase_result["Phrase Failure Reason"], "final_text_mismatch")

    def test_enter_miss_retries_without_repeating_letters(self):
        sim = recovery_sim(
            [
                (None, kconfig.undo_word_index),
                ("cat", FakeKeyboard.target_index),
            ]
        )

        sim.type_phrase("cat")

        self.assertEqual(sim.keyboard.typed, "cat ")
        self.assertEqual(sim.letter_press_count, 1)
        self.assertEqual(sim.num_word_attempts_phrase, 1)
        self.assertEqual(sim.num_enter_retries_phrase, 1)
        self.assertEqual(sim.num_clicks_phrase, 3)

    def test_wrong_commit_undo_restore_retries_only_enter(self):
        sim = recovery_sim(
            [
                ("good", 9),
                (None, kconfig.undo_word_index),
                ("cat", FakeKeyboard.target_index),
            ]
        )

        sim.type_phrase("cat")

        self.assertEqual(sim.keyboard.typed, "cat ")
        self.assertEqual(sim.letter_press_count, 1)
        self.assertEqual(sim.keyboard.restore_count, 1)
        self.assertEqual(sim.num_wrong_word_commits_phrase, 1)
        self.assertEqual(sim.num_undo_attempts_phrase, 1)
        self.assertEqual(sim.num_enter_retries_phrase, 1)
        phrase_result = sim._calculate_phrase_results("cat", "?")
        self.assertEqual(phrase_result["Typed Text"], "cat ")
        self.assertEqual(phrase_result["Completed Word Count"], 1)
        self.assertTrue(phrase_result["Phrase Completed"])

    def test_mistimed_undo_suppresses_competitor_and_retries(self):
        sim = recovery_sim(
            [
                ("good", 9),
                ("bad", 10),
                (None, kconfig.undo_word_index),
                ("cat", FakeKeyboard.target_index),
            ]
        )

        sim.type_phrase("cat")

        self.assertEqual(sim.keyboard.typed, "cat ")
        self.assertEqual(sim.num_wrong_word_commits_phrase, 1)
        self.assertEqual(sim.num_undo_attempts_phrase, 2)
        self.assertEqual(sim.num_undo_failures_phrase, 0)
        self.assertEqual(sim.keyboard.restore_count, 1)
        self.assertEqual(sim.keyboard.prepare_undo_count, 1)

    def test_protected_undo_exhaustion_stops_without_context_corruption(self):
        sim = recovery_sim(
            [
                ("wrong", 9),
                ("competitor-one", 10),
                ("competitor-two", 11),
            ]
        )
        sim.max_clicks_per_word = 4

        sim.type_phrase("cat")

        self.assertEqual(sim.num_failed_words_phrase, 1)
        self.assertEqual(sim.num_undo_failures_phrase, 1)
        self.assertEqual(sim.num_wrong_word_commits_phrase, 1)
        self.assertEqual(sim.num_undo_attempts_phrase, 2)
        self.assertEqual(sim.keyboard.restore_count, 0)
        self.assertEqual(sim.keyboard.typed, "wrong ")
        phrase_result = sim._calculate_phrase_results("cat", "?")
        self.assertEqual(
            phrase_result["Phrase Failure Reason"],
            "undo_click_budget_exhausted",
        )
        self.assertEqual(phrase_result["Phrase Failure Stage"], "undo")
        self.assertEqual(phrase_result["Phrase Failure Limit"], "max_clicks_per_word")
        self.assertEqual(
            phrase_result["Phrase Failure Guard"],
            "undo_click_budget_exhausted",
        )
        self.assertEqual(phrase_result["Failed Target Word"], "cat")
        self.assertEqual(phrase_result["Failed Word Position"], 1)
        self.assertEqual(phrase_result["Failure Word Click Count"], 4)
        self.assertEqual(phrase_result["Failure Target Enter Attempt Count"], 1)
        self.assertEqual(phrase_result["Failure Undo Attempt Count"], 2)

    def test_target_not_displayed_is_reported_after_word_attempt_limit(self):
        sim = recovery_sim([(None, kconfig.undo_word_index)])
        sim.keyboard.target_index = kconfig.undo_word_index
        sim.max_word_attempts = 1
        sim.max_enter_attempts = 1

        sim.type_phrase("cat")
        phrase_result = sim._calculate_phrase_results("cat", "?")

        self.assertEqual(phrase_result["Phrase Failure Reason"], "target_not_displayed")
        self.assertEqual(phrase_result["Phrase Failure Stage"], "word_prediction")
        self.assertEqual(phrase_result["Phrase Failure Limit"], "max_word_attempts")
        self.assertEqual(
            phrase_result["Phrase Failure Guard"],
            "word_attempts_exhausted",
        )
        self.assertFalse(phrase_result["Failed Target Was Displayed"])
        self.assertEqual(phrase_result["Failure Letter Press Count"], 3)
        self.assertEqual(phrase_result["Failure Target Enter Attempt Count"], 1)

    def test_target_enter_retry_exhaustion_is_reported_exactly(self):
        sim = recovery_sim(
            [
                (None, kconfig.undo_word_index),
                (None, kconfig.undo_word_index),
            ]
        )
        sim.max_word_attempts = 1
        sim.max_enter_attempts = 2

        sim.type_phrase("cat")
        phrase_result = sim._calculate_phrase_results("cat", "?")

        self.assertEqual(
            phrase_result["Phrase Failure Reason"],
            "target_enter_retries_exhausted",
        )
        self.assertEqual(phrase_result["Phrase Failure Stage"], "target_enter")
        self.assertEqual(phrase_result["Phrase Failure Limit"], "max_word_attempts")
        self.assertEqual(
            phrase_result["Phrase Failure Guard"],
            "word_attempts_exhausted",
        )
        self.assertTrue(phrase_result["Failed Target Was Displayed"])
        self.assertEqual(phrase_result["Failure Letter Press Count"], 1)
        self.assertEqual(phrase_result["Failure Target Enter Attempt Count"], 2)


if __name__ == "__main__":
    unittest.main()
