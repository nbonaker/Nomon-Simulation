import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from OneClick_Text import kconfig
from OneClick_Text.language_model import (
    LanguageModel,
    NGRAM_BACKEND,
    NGRAM_MODEL_ALPHABET,
    language_model_metadata,
    load_local_language_model,
)
from textslinger.ngram import (
    ConfigPredictCharactersNGram,
    ConfigPredictWordsNGram,
)


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

    def test_ngram_backend_uses_ngram_configs_and_fixed_word_list(self):
        model = FakeTextSlingerModel()
        word_list = object()
        adapter = LanguageModel(
            model,
            backend=NGRAM_BACKEND,
            word_list=word_list,
        )

        adapter.get_key_probs("hello ")
        adapter.get_word_predictions(
            "hello ",
            [[0.0] * len(kconfig.key_chars)],
        )

        self.assertIsInstance(
            model.character_calls[0]["config"],
            ConfigPredictCharactersNGram,
        )
        self.assertIsInstance(
            model.word_calls[0]["config"],
            ConfigPredictWordsNGram,
        )
        self.assertIs(model.word_calls[0]["word_list"], word_list)

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

    def test_character_predictions_are_cached_by_context(self):
        model = FakeTextSlingerModel()
        model.character_predictions = [
            _prediction(character="a", lower=math.log(0.8)),
        ]
        adapter = LanguageModel(model)

        first = adapter.get_key_probs("same context ")
        first[0] = 123.0
        second = adapter.get_key_probs("same context ")

        self.assertEqual(len(model.character_calls), 1)
        self.assertNotEqual(second[0], 123.0)
        self.assertEqual(adapter.result_cache_stats()["character"]["hits"], 1)

    def test_character_transition_matrix_is_ordered_cached_and_defensive(self):
        model = FakeTextSlingerModel()
        adapter = LanguageModel(model)
        requested_contexts = []

        def key_probs(context):
            requested_contexts.append(context)
            source_index = kconfig.key_chars.index(context[-1])
            return [0.0 if index == source_index else -100.0 for index in range(27)]

        with patch.object(adapter, "get_key_probs", side_effect=key_probs):
            first = adapter.get_character_transition_log_probs("left ")
            first[0][0] = 123.0
            second = adapter.get_character_transition_log_probs("left ")
            adapter.get_character_transition_log_probs("other ")

        alphabet_size = len(kconfig.key_chars)
        self.assertEqual(len(first), alphabet_size)
        self.assertTrue(all(len(row) == alphabet_size for row in first))
        for source_index, row in enumerate(second):
            expected = [
                0.0 if index == source_index else -100.0
                for index in range(alphabet_size)
            ]
            self.assertEqual(row, expected)
        self.assertEqual(second[0][0], 0.0)
        self.assertEqual(requested_contexts[:alphabet_size], [
            "left " + character for character in kconfig.key_chars
        ])
        stats = adapter.result_cache_stats()["character_transition"]
        self.assertEqual((stats["hits"], stats["misses"]), (1, 2))


    def test_word_predictions_are_cached_by_exact_context_and_observations(self):
        model = FakeTextSlingerModel()
        model.word_predictions = [_prediction(word="cat", lower=-0.1)]
        adapter = LanguageModel(model)
        rows = [[0.0] * len(kconfig.key_chars)]

        first, _ = adapter.get_word_predictions("left ", rows)
        first[0]["text"] = "changed"
        second, _ = adapter.get_word_predictions("left ", rows)
        adapter.get_word_predictions("different ", rows)

        self.assertEqual(second[0]["text"], "cat")
        self.assertEqual(len(model.word_calls), 2)
        stats = adapter.result_cache_stats()["word"]
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 2)

    def test_ngram_loader_records_model_and_vocabulary_metadata(self):
        model = FakeTextSlingerModel()
        word_list = object()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "lm_char_medium.kenlm"
            vocabulary_path = root / "vocab_lower_100k.txt"
            model_path.write_bytes(b"fake kenlm")
            vocabulary_path.write_text("cat\ndog\n", encoding="utf-8")
            module = "OneClick_Text.language_model"
            with (
                patch(f"{module}.NGramLanguageModel", return_value=model) as loader,
                patch(f"{module}.WordList.from_file", return_value=word_list),
            ):
                adapter = load_local_language_model(
                    model_path,
                    backend=NGRAM_BACKEND,
                    vocabulary_path=vocabulary_path,
                )
                metadata = language_model_metadata(adapter)

        loader.assert_called_once_with(
            lm_path=str(model_path.resolve()),
            model_alphabet=NGRAM_MODEL_ALPHABET,
            space_character="<sp>",
        )
        self.assertEqual(adapter.backend, NGRAM_BACKEND)
        self.assertIs(adapter.word_list, word_list)
        self.assertEqual(adapter.resolved_device, "cpu")
        self.assertIsNone(adapter.precision)
        self.assertEqual(metadata["model_family"], NGRAM_BACKEND)
        self.assertEqual(metadata["model_source"], "local_file")
        self.assertEqual(metadata["model_path"], str(model_path.resolve()))
        self.assertEqual(
            metadata["vocabulary_path"],
            str(vocabulary_path.resolve()),
        )
        self.assertEqual(len(metadata["model_sha256"]), 64)
        self.assertEqual(len(metadata["vocabulary_sha256"]), 64)

    def test_ngram_loader_requires_a_vocabulary(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "lm_char_medium.kenlm"
            model_path.write_bytes(b"fake kenlm")
            with self.assertRaisesRegex(ValueError, "vocabulary_path is required"):
                load_local_language_model(
                    model_path,
                    backend=NGRAM_BACKEND,
                )


if __name__ == "__main__":
    unittest.main()
