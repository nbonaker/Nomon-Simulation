from __future__ import annotations

import unittest

from types import SimpleNamespace

import numpy as np

from Nomon_Text.imagineville_lm import ImaginevilleLM

from OneClick_Text.language_model import LanguageModel

from User_Simulation.simulated_user_text import SimulatedUser

class LanguageModelReliabilityTests(unittest.TestCase):
    def test_oneclick_local_model_errors_are_not_silently_replaced(self):
        model = SimpleNamespace(
            predict_characters=lambda **_kwargs: (_ for _ in ()).throw(
                RuntimeError("local inference failed")
            )
        )
        language_model = LanguageModel(model)

        with self.assertRaisesRegex(RuntimeError, "local inference failed"):
            language_model.get_key_probs("context")

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
