import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from .bridge_simulated_user import BridgeSimulatedUser
from .bridge_study import run_study


class StudyTests(unittest.TestCase):
    def test_corpus_consumes_each_user_stream_in_order_and_saves_failures(self):
        user = BridgeSimulatedUser()
        seen = []

        def run_phrase(target, source, backend):
            sample = source.sample()
            seen.append(sample.offset_s)
            failure = ("word_press_budget", None, "prediction_failure")[len(seen) - 1]
            return {
                "target": target, "final_text": "", "completed": failure is None,
                "failure_reason": failure, "simulated_elapsed_s": 1, "typing_elapsed_s": 1,
                "counts": {"presses": 1, "space": 1, "enter": 0, "wrong_commits": 0, "undo_attempts": 0},
                "trace": {"events": []},
            }

        user.run_phrase = run_phrase
        with TemporaryDirectory() as temp:
            root = Path(temp)
            summary = run_study(root, user, object, users=["A", "F"], phrase_count=2,
                                backend_settings={"model": "test"})
            rows = json.loads((root / "phrases.json").read_text())
            self.assertEqual([(row["user"], row["schedule_start"], row["schedule_end_exclusive"])
                              for row in rows], [("A", 0, 1), ("A", 1, 2), ("F", 0, 1)])
            expected = []
            for uid, number in (("A", 2), ("F", 1)):
                with (root / f"user_{uid}_click_schedule.csv").open(newline="") as stream:
                    reader = csv.DictReader(stream)
                    expected.extend(float(next(reader)["Click Time Relative (s)"]) for _ in range(number))
            self.assertEqual(seen, expected)
            self.assertEqual(summary["attempted"], 3)
            self.assertEqual(summary["unattempted"], 1)
            self.assertEqual(summary["completed"], 1)
            self.assertEqual(summary["budget_stops"], 1)
            self.assertEqual(summary["failure_reasons"], {"word_press_budget": 1, "prediction_failure": 1})
            for row in rows:
                self.assertTrue((root / row["artifacts"] / "result.json").is_file())
                self.assertTrue((root / row["artifacts"] / "trace.json").is_file())


if __name__ == "__main__":
    unittest.main()
