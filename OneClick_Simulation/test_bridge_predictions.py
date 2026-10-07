"""Model-free adapter/scheduler contracts, plus opt-in local n-gram integration."""
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from OneClick_Bridge import OneClickEngineBridge
from .bridge_predictions import ALPHABET, FixedCharacterTransitions, TextSlingerNGramBackend
from .bridge_simulated_user import (BridgeSimulatedUser, ClickSample, FixedPredictionBackend,
                                    ZeroNoiseClickSource, replay_trace, snapshot_digest)
from .test_bridge_simulated_user import CHARACTERS, words, presses

FIXTURES = Path(__file__).with_name('bridge_fixtures')


def fixture_backend():
    return FixedPredictionBackend.from_dict(json.loads((FIXTURES / 'hello_world.json').read_text()))


def effect():
    return {'type': 'predict-words', 'left': 'hello ', 'numPrefix': 2, 'numBest': 1,
            'distribs': [{'distrib': [{'text': c, 'logProb': -i} for i, c in reversed(list(enumerate(ALPHABET)))]}]}


class FakeModel:
    key_chars = tuple(ALPHABET)
    def __init__(self):
        self.calls = []
        self.response = (words(['abc', 'def', 'ghi'])['prefix'], words(best=['a', 'b'])['best'])
    def get_key_probs(self, context):
        self.calls.append(context)
        return [-0.1] + [-100.0] * 26
    def get_word_predictions(self, context, observations, **options):
        self.calls.append((context, observations, options))
        return copy.deepcopy(self.response)


class AdapterTests(unittest.TestCase):
    def test_effect_translation_reorders_observations_and_forwards_limits(self):
        model = FakeModel()
        backend = TextSlingerNGramBackend(model)
        request = effect()
        original = copy.deepcopy(request)
        result = backend.predict(request)
        self.assertEqual(model.calls, [('hello ', [[-i for i in range(27)]],
                                       {'prefix_limit': 2, 'best_limit': 1, 'strict_scores': True})])
        self.assertEqual([x['text'] for x in result['prefix']], ['abc', 'def'])
        self.assertEqual([x['text'] for x in result['best']], ['a'])
        self.assertEqual(request, original)
        row = backend.predict({'type': 'predict-characters', 'left': 'h', 'index': 7})
        self.assertEqual(model.calls[-1], 'h')
        self.assertEqual(''.join(x['token'] for x in row['results']), ALPHABET)
        self.assertEqual(row['results'][1]['logProb'], -100)

    def test_nonfinite_words_and_invalid_observations(self):
        model = FakeModel()
        backend = TextSlingerNGramBackend(model)
        model.response = ([{'text': 'impossible', 'logprob': -math.inf}, {'text': 'abc', 'logprob': -1}], [])
        self.assertEqual(backend.predict(effect()), {'prefix': [{'text': 'abc', 'logprob': -1}], 'best': []})
        self.assertEqual(backend.metadata()['dropped_negative_infinity_candidates'], 1)
        for invalid in [math.nan, math.inf]:
            model.response = ([{'text': 'abc', 'logprob': invalid}], [])
            with self.assertRaisesRegex(ValueError, 'finite'):
                backend.predict(effect())
        for mutation in ('missing', 'duplicate', 'nan', 'unknown'):
            request = effect()
            row = request['distribs'][0]['distrib']
            if mutation == 'missing': row.pop()
            elif mutation == 'duplicate': row.append(row[0])
            elif mutation == 'nan': row[0]['logProb'] = math.nan
            else: row[0]['text'] = '!'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                backend.predict(request)
        with self.assertRaisesRegex(ValueError, 'Model file'):
            TextSlingerNGramBackend.from_local('/missing/model', '/missing/vocab')

    def test_fixed_transition_validation_and_detachment(self):
        source = FixedCharacterTransitions(CHARACTERS)
        response = source.predict({'type': 'predict-characters', 'left': 'a'})
        response['results'][0]['logProb'] = 123
        self.assertEqual(source.predict({'type': 'predict-characters', 'left': 'a'}), CHARACTERS)
        for bad in ({}, {'a': CHARACTERS}, {'results': []}, {'results': CHARACTERS['results'][:-1]}):
            with self.assertRaises(ValueError): FixedCharacterTransitions(bad)
        bad = copy.deepcopy(CHARACTERS)
        bad['results'][0]['logProb'] = -math.inf
        with self.assertRaises(ValueError): FixedCharacterTransitions(bad)

    def test_fixture_and_replay_imports_are_model_free(self):
        code = """
import sys
from OneClick_Simulation.bridge_pilot import main
from OneClick_Simulation.bridge_predictions import TextSlingerNGramBackend
assert not any(name.startswith(('textslinger', 'numpy', 'torch', 'OneClick_Text')) for name in sys.modules)
"""
        subprocess.run([sys.executable, '-c', code], check=True)


class PredictionSchedulingTests(unittest.TestCase):
    def run_fixture(self, **options):
        return BridgeSimulatedUser(**options).run_phrase('hello world', ZeroNoiseClickSource(), fixture_backend())

    def test_zero_latency_matches_preintegration_event_digest(self):
        baseline = json.loads((FIXTURES / 'zero_latency_baseline.json').read_text())
        result = self.run_fixture()
        self.assertTrue(result['completed'], result['error'])
        self.assertEqual(len(result['trace']['events']), baseline['event_count'])
        self.assertEqual(snapshot_digest(result['trace']['events']), baseline['events_sha256'])
        self.assertEqual(result['simulated_elapsed_s'], 9.06)
        self.assertEqual(result['startup_elapsed_s'], 0)
        self.assertEqual(result['prediction_wait_s'], 0)
        # Version-1 traces without the new optional metadata still replay.
        legacy = {key: result['trace'][key] for key in ('version', 'engine_config', 'engine_metadata', 'events')}
        self.assertEqual(replay_trace(legacy)['snapshot'], result['final_snapshot'])

    def test_startup_arrival_order_and_frame_origin(self):
        for character_latency, word_latency in [(0.13, 0.0), (0.0, 0.13), (0.13, 0.13)]:
            with self.subTest(character=character_latency, word=word_latency):
                result = self.run_fixture(character_prediction_latency_s=character_latency,
                                          word_prediction_latency_s=word_latency)
                self.assertTrue(result['completed'], result['error'])
                events = [r['event'] for r in result['trace']['events']]
                initial = events[1:29]
                expected = ['prediction-response'] + ['character-response'] * 27 if word_latency < character_latency else ['character-response'] * 27 + ['prediction-response']
                self.assertEqual([e['type'] for e in initial], expected)
                self.assertEqual([e['index'] for e in initial if e['type'] == 'character-response'], list(range(27)))
                self.assertAlmostEqual(result['startup_elapsed_s'], 0.13)
                first_frame = next(e for e in events if e['type'] == 'frame')
                self.assertAlmostEqual(first_frame['time'], 0.17)
                self.assertTrue(all(a['time'] <= b['time'] for a,b in zip(events, events[1:])))
                self.assertAlmostEqual(result['simulated_elapsed_s'], result['startup_elapsed_s'] + result['typing_elapsed_s'])
                self.assertEqual(replay_trace(result['trace'])['snapshot'], result['final_snapshot'])

    def test_delayed_frames_waits_and_response_frame_tie(self):
        # First Space is on a half-frame; .10 latency puts its response on a frame.
        result = self.run_fixture(character_prediction_latency_s=0.2, word_prediction_latency_s=0.1)
        self.assertTrue(result['completed'], result['error'])
        records = result['trace']['events']
        requests = result['trace']['prediction_requests']
        for request in requests:
            self.assertEqual(request['status'], 'delivered')
            latency = .2 if request['effect']['type'] == 'predict-characters' else .1
            self.assertAlmostEqual(request['due_time'], request['requested_time'] + latency)
            self.assertAlmostEqual(request['delivered_time'], request['due_time'])
        action = presses(result)[0]
        start = records.index(action)
        response_index = next(i for i in range(start+1, len(records)) if records[i]['event']['type'] == 'prediction-response')
        response = records[response_index]['event']
        self.assertGreater(response_index, start+2)
        self.assertEqual([records[i]['event'].get('group') for i in (response_index-2, response_index-1)], ['letters', 'words'])
        self.assertAlmostEqual(records[response_index-1]['event']['time'], response['time'])
        next_press = presses(result)[1]
        self.assertGreater(next_press['event']['time'], response['time'])
        self.assertEqual(next_press['intent']['clock_id'], 12)  # hello arrived after Space
        self.assertAlmostEqual(result['typing_prediction_wait_s'], 0.4)
        self.assertAlmostEqual(result['prediction_wait_s'], 0.6)
        self.assertEqual(replay_trace(result['trace'])['snapshot'], result['final_snapshot'])

    def test_matrix_sources_preserve_engine_floor_and_normalization(self):
        rows = {c: {'results': [{'token': x, 'logProb': -0.1 if x == c else -100} for x in ALPHABET]} for c in ALPHABET}
        backend = fixture_backend()
        original_predict = backend.predict
        def words_only(effect):
            self.assertEqual(effect['type'], 'predict-words')
            return original_predict(effect)
        backend.predict = words_only
        result = BridgeSimulatedUser(character_transition_source='fixed', fixed_character_responses=rows).run_phrase(
            'hello world', ZeroNoiseClickSource(), backend)
        self.assertTrue(result['completed'], result['error'])
        meta = result['prediction_metadata']
        self.assertEqual(meta['character_responses'], rows)
        matrix = meta['effective_transition_matrix']
        expected_norm = math.log(math.exp(-.1) + 26*.01)
        for i, row in enumerate(matrix):
            self.assertEqual(len(row), 27)
            self.assertAlmostEqual(sum(map(math.exp, row)), 1)
            for j, value in enumerate(row):
                self.assertAlmostEqual(value, (-.1 if i == j else math.log(.01)) - expected_norm)
        self.assertEqual(meta['effective_transition_matrix_sha256'], snapshot_digest(matrix))
        backend_matrix = self.run_fixture()['prediction_metadata']['effective_transition_matrix']
        self.assertNotEqual(matrix, backend_matrix)
        shared = self.run_fixture(character_transition_source='fixed', fixed_character_responses=CHARACTERS)
        self.assertEqual(shared['prediction_metadata']['effective_transition_matrix'], backend_matrix)

    def test_nonfinite_backend_payload_retains_serializable_failure(self):
        backend = fixture_backend()
        original = backend.predict
        def bad_after_space(effect):
            if effect['type'] == 'predict-words' and effect['distribs']:
                return {'prefix': [{'text': 'hello', 'logprob': math.nan}], 'best': []}
            return original(effect)
        backend.predict = bad_after_space
        result = BridgeSimulatedUser().run_phrase('hello', ZeroNoiseClickSource(), backend)
        self.assertEqual(result['failure_reason'], 'prediction_failure')
        self.assertTrue(result['final_text_known'])
        self.assertEqual(result['trace']['events'][-1]['event']['type'], 'space')
        json.dumps(result, allow_nan=False)
        self.assertEqual(replay_trace(result['trace'])['snapshot'], result['final_snapshot'])

    def test_invalid_settings_and_backend_failure_cleanup(self):
        for field in ('word_prediction_latency_s', 'character_prediction_latency_s'):
            for bad in (-1, math.inf, math.nan, True):
                with self.assertRaises(ValueError): BridgeSimulatedUser(**{field: bad})
        live = []
        def factory(**kwargs):
            bridge = OneClickEngineBridge(**kwargs)
            live.append(bridge)
            return bridge
        backend = fixture_backend()
        original = backend.predict
        def fail_after_space(effect):
            if effect['type'] == 'predict-words' and effect['distribs']:
                raise ValueError('model failed')
            return original(effect)
        backend.predict = fail_after_space
        with patch('OneClick_Simulation.bridge_simulated_user.OneClickEngineBridge', side_effect=factory):
            result = BridgeSimulatedUser(word_prediction_latency_s=.2).run_phrase('hello', ZeroNoiseClickSource(), backend)
        self.assertEqual(result['failure_reason'], 'prediction_failure')
        self.assertTrue(result['final_text_known'])
        self.assertEqual(result['trace']['prediction_requests'][-1]['status'], 'error')
        self.assertIsNotNone(live[0]._proc.poll())
        self.assertTrue(all(not t.is_alive() for t in live[0]._threads))
        self.assertEqual(replay_trace(result['trace'])['snapshot'], result['final_snapshot'])


@unittest.skipUnless(os.environ.get('QUICKCLICK_TEST_NGRAM') == '1', 'Set QUICKCLICK_TEST_NGRAM=1 for local model integration')
class RealNGramTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        resources = Path(__file__).resolve().parents[1] / 'Nomon_Text' / 'resources'
        cls.backend = TextSlingerNGramBackend.from_local(resources / 'lm_char_tiny.kenlm', resources / 'vocab_lower_100k.txt')

    def test_word_sentence_and_noisy_correction(self):
        for phrase, miss in [('hello', False), ('hello world', False), ('hello', True)]:
            with self.subTest(phrase=phrase, miss=miss):
                user = BridgeSimulatedUser(word_prediction_latency_s=.1, character_prediction_latency_s=.2)
                class MissFirstEnter:
                    missed = False
                    def sample(source):
                        target = user._matching_word('hello')
                        if miss and not source.missed and user.state['observations'] and target is not None:
                            source.missed = True
                            group = user.state['words']
                            wrong = next(i for i in user.state['valid_word_indices'] if i not in (target, user._undo_id()) and group['clocks'][i]['label'] != 'hello')
                            return ClickSample((group['phases'][target] - group['phases'][wrong]) * group['period'] / group['num_divs_time'])
                        return ClickSample()
                result = user.run_phrase(phrase, MissFirstEnter(), self.backend)
                self.assertTrue(result['completed'], result['error'] or result['failure_reason'])
                self.assertEqual(result['final_text'], phrase + ' ')
                if miss:
                    self.assertGreaterEqual(result['counts']['wrong_commits'], 1)
                    self.assertGreaterEqual(result['counts']['undo_attempts'], 1)
                self.assertEqual(replay_trace(result['trace'])['snapshot'], result['final_snapshot'])
                metadata = result['prediction_metadata']['backend']
                self.assertEqual(metadata['model_family'], 'ngram')
                self.assertEqual(len(metadata['model_sha256']), 64)
                self.assertEqual(len(metadata['vocabulary_sha256']), 64)


if __name__ == '__main__':
    unittest.main()
