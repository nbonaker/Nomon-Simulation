import math
import unittest
from types import SimpleNamespace

from OneClick_Text import kconfig
from OneClick_Text.language_model import LanguageModel


def _prediction(*, character=None, word=None, lower=None, retained=-20.0):
    prediction = SimpleNamespace(
        probability=SimpleNamespace(
            log_lower_bound=lower,
            log_retained_joint_mass=retained,
        )
    )
    if character is not None:
        prediction.character = character
    if word is not None:
        prediction.word = word
    return prediction


class FakeTextSlingerModel:
    def __init__(self):
        self.character_calls = []
        self.word_calls = []
        self.character_predictions = []
        self.word_predictions = []

    def predict_characters(self, **kwargs):
        self.character_calls.append(kwargs)
        return SimpleNamespace(predictions=tuple(self.character_predictions))

    def predict_words(self, **kwargs):
        self.word_calls.append(kwargs)
        return SimpleNamespace(predictions=tuple(self.word_predictions))


class LocalLanguageModelTests(unittest.TestCase):
    def test_empty_context_is_uniform_without_calling_model(self):
        model = FakeTextSlingerModel()
        adapter = LanguageModel(model)

        result = adapter.get_key_probs("")

        self.assertFalse(model.character_calls)
        self.assertEqual(len(result), len(kconfig.key_chars))
        self.assertTrue(all(value == result[0] for value in result))

    def test_character_results_are_clamped_normalized_and_key_ordered(self):
        model = FakeTextSlingerModel()
        model.character_predictions = [
            _prediction(character="b", lower=math.log(0.8)),
            _prediction(character="a", lower=math.log(0.2)),
        ]
        adapter = LanguageModel(model)

        result = adapter.get_key_probs("hello ")

        self.assertAlmostEqual(sum(math.exp(value) for value in result), 1.0)
        self.assertGreater(result[kconfig.key_chars.index("b")], result[0])
        self.assertEqual(
            model.character_calls[0]["output_characters"], tuple(kconfig.key_chars)
        )

    def test_word_call_receives_one_input_event_per_raw_clock_row(self):
        model = FakeTextSlingerModel()
        adapter = LanguageModel(model)
        rows = [
            [float(index) for index in range(len(kconfig.key_chars))],
            [-float(index) for index in range(len(kconfig.key_chars))],
        ]

        adapter.get_word_predictions("left ", rows)

        call = model.word_calls[0]
        self.assertEqual(len(call["input_sequence"]), 2)
        self.assertEqual(
            call["input_sequence"][0].alternatives,
            tuple(zip(kconfig.key_chars, rows[0])),
        )
        self.assertEqual(call["input_channel"].complete_endpoint_probability, 0.5)
        self.assertEqual(call["input_channel"].channel_insertion_probability, 0.0)
        self.assertEqual(call["input_channel"].channel_deletion_probability, 0.0)

    def test_mixed_results_split_deduplicate_and_prefer_lower_bound(self):
        model = FakeTextSlingerModel()
        model.word_predictions = [
            _prediction(word="cat", lower=-0.1, retained=-10.0),
            _prediction(word="CAT", lower=-0.2),
            _prediction(word="car", lower=None, retained=-0.3),
            _prediction(word="ca", lower=-0.4),
            _prediction(word="c", lower=-0.5),
            SimpleNamespace(word="bad"),
        ]
        adapter = LanguageModel(model)

        prefix, best = adapter.get_word_predictions(
            "", [[0.0] * len(kconfig.key_chars)] * 2
        )

        self.assertEqual(prefix, [
            {"text": "cat", "logprob": -0.1},
            {"text": "car", "logprob": -0.3},
        ])
        self.assertEqual(best, [{"text": "ca", "logprob": -0.4}])
        self.assertEqual(len(model.word_calls), 1)

    def test_malformed_observation_fails_before_model_call(self):
        model = FakeTextSlingerModel()
        adapter = LanguageModel(model)
        with self.assertRaisesRegex(ValueError, "expected 27"):
            adapter.get_word_predictions("", [[0.0]])
        self.assertFalse(model.word_calls)


if __name__ == "__main__":
    unittest.main()
