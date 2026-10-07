import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from OneClick_Bridge import OneClickEngineBridge
from OneClick_Simulation.bridge_simulated_user import (
    BridgeSimulatedUser, ClickSample, FixedPredictionBackend, FrameScheduler,
    SequenceClickSource, ZeroNoiseClickSource, replay_trace, snapshot_digest,
)
from OneClick_Simulation.bridge_pilot import main

CHARACTERS = {'results': [{'token': c, 'logProb': -3.3} for c in "abcdefghijklmnopqrstuvwxyz'"]}


def words(prefix=(), best=()):
    return {'prefix': [{'text': text, 'logprob': -0.1 - i} for i, text in enumerate(prefix)],
            'best': [{'text': text, 'logprob': -0.1 - i} for i, text in enumerate(best)]}


def backend(mapping):
    return FixedPredictionBackend(CHARACTERS, mapping)


def presses(result):
    return [record for record in result['trace']['events'] if record['event']['type'] in ('space', 'enter')]


class PilotTests(unittest.TestCase):
    def run_word(self, mapping=None, samples=None, **options):
        mapping = mapping or {('', 0): words(), ('', 1): words(['hello']), ('hello ', 0): words()}
        return BridgeSimulatedUser(**options).run_phrase('hello', samples or ZeroNoiseClickSource(), backend(mapping))

    def test_word_sentence_and_replay(self):
        one = self.run_word()
        self.assertTrue(one['completed'], one['error'])
        self.assertEqual(one['counts']['space'], 1)
        self.assertEqual(presses(one)[1]['intent']['clock_id'], 12)
        fixture = json.loads(Path(__file__).with_name('bridge_fixtures').joinpath('hello_world.json').read_text())
        result = BridgeSimulatedUser().run_phrase('  HELLO\n world ', ZeroNoiseClickSource(), FixedPredictionBackend.from_dict(fixture))
        self.assertTrue(result['completed'], result['error'])
        self.assertEqual(result['target'], 'hello world')
        self.assertEqual(result['final_text'], 'hello world ')
        self.assertEqual(result['counts']['presses'], 4)
        replay = replay_trace(result['trace'])
        self.assertTrue(replay['complete_trace'])
        self.assertEqual(replay['snapshot'], result['final_snapshot'])
        self.assertEqual(replay['verified_events'], len(result['trace']['events']))
        events = result['trace']['events']
        self.assertTrue(all(a['event']['time'] <= b['event']['time'] for a, b in zip(events, events[1:])))
        changed = copy.deepcopy(result['trace'])
        changed['engine_metadata']['sourceSha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            replay_trace(changed)

    def test_best_and_literal_use_real_ids_and_no_observation_overwrite(self):
        for best, expected in [(['a'], 81), ([], 84)]:
            with self.subTest(best=best):
                result = BridgeSimulatedUser().run_phrase('a', SequenceClickSource([ClickSample(0.05), ClickSample()]),
                    backend({('', 0): words(), ('', 1): words(best=best), ('a ', 0): words()}))
                self.assertTrue(result['completed'], result['error'])
                self.assertEqual(presses(result)[1]['intent']['clock_id'], expected)
                self.assertEqual(presses(result)[1]['observed']['selection']['id'], expected)
                # Recover the actual observation by replaying only recorded events.
                with OneClickEngineBridge() as engine:
                    for record in result['trace']['events']:
                        state = engine.dispatch(record['event'])['snapshot']
                        self.assertEqual(snapshot_digest(state), record['snapshot_sha256'])
                        if record['event']['type'] == 'space':
                            row = state['observations'][0]
                            self.assertEqual(len(row), 27)
                            self.assertTrue(all(math.isfinite(x) for x in row))
                            self.assertNotEqual(row[0], 0)  # actual Gaussian likelihood, not perfect-letter replacement
                            self.assertEqual(max(range(27), key=row.__getitem__), 0)
                            break

    def test_word_undo_is_not_confused_with_the_command_clock(self):
        fixtures = {('', i): words() for i in range(5)}
        fixtures[('undo ', 0)] = words()
        result = BridgeSimulatedUser().run_phrase('undo',
            SequenceClickSource([ClickSample(0.05)] * 4 + [ClickSample()]), backend(fixtures))
        self.assertTrue(result['completed'], result['error'])
        self.assertEqual(result['counts']['space'], 4)
        self.assertEqual(presses(result)[-1]['observed']['selection']['id'], 84)

    def test_midword_undo_retypes_discarded_observations(self):
        result = self.run_word(samples=SequenceClickSource([ClickSample(), ClickSample(-2.4), ClickSample(), ClickSample()]))
        self.assertTrue(result['completed'], result['error'])
        actions = presses(result)
        self.assertEqual([r['event']['type'] for r in actions], ['space', 'enter', 'space', 'enter'])
        self.assertEqual(actions[1]['observed']['selection']['id'], 85)
        self.assertEqual(actions[1]['observed']['observation_count'], 0)
        self.assertEqual(result['attempts_by_word'], [2])
        self.assertEqual(result['final_snapshot']['delay_model']['n_samples'], 1)

    def test_wrong_commit_and_missed_undo_cascade_then_retype(self):
        fixtures = {('', 0): words(), ('', 1): words(['hello', 'wrong']),
                    ('wrong ', 0): words(['oops']), ('wrong oops ', 0): words(), ('hello ', 0): words()}
        samples = SequenceClickSource([ClickSample(), ClickSample(-0.88), ClickSample(1.8),
                                       ClickSample(), ClickSample(), ClickSample(), ClickSample()])
        result = self.run_word(fixtures, samples)
        self.assertTrue(result['completed'], result)
        actions = presses(result)
        self.assertEqual([r['observed']['typed'] for r in actions if r['event']['type'] == 'enter'],
                         ['wrong ', 'wrong oops ', 'wrong ', '', 'hello '])
        self.assertEqual(result['counts']['wrong_commits'], 2)
        self.assertEqual(result['counts']['undo_attempts'], 3)
        self.assertEqual(result['presses_by_word'], [7])
        self.assertEqual(result['attempts_by_word'], [2])
        # Browser learns from every wrong commit and has only one rollback slot.
        self.assertEqual(result['final_snapshot']['delay_model']['n_samples'], 2)
        self.assertEqual(replay_trace(result['trace'])['snapshot'], result['final_snapshot'])

    def test_policy_revisits_a_shorter_valid_prefix_without_resetting_budgets(self):
        # Observation-policy test: an externally observed rewind removed a correct
        # word. The real engine's Undo behavior is covered by bridge/integration tests.
        user = BridgeSimulatedUser()
        user.words = ['a', 'b']
        user.prefixes = ['', 'a ', 'a b ']
        user.attempts = [1, 0]
        user.presses_by_word = [2, 0]
        user.state = {'typed': 'a ', 'letters': {'clocks': [{'id': 0, 'label': 'a'}, {'id': 1, 'label': 'b'}]}}
        calls = []
        outcomes = iter(['', 'a ', 'a b '])
        def press(group, clock_id, action, target):
            calls.append((action, target, user.current_word))
            user.presses_by_word[user.current_word] += 1
            if action == 'target_enter':
                user.state['typed'] = next(outcomes)
        user._press = press
        user._matching_word = lambda target: 81
        user._type_phrase()
        self.assertEqual([target for action, target, _ in calls if action == 'letter'], ['b', 'a', 'b'])
        self.assertEqual(user.attempts, [2, 2])
        self.assertEqual(user.presses_by_word, [4, 4])

    def test_prediction_failure_exhaustion_and_all_budgets(self):
        missing = self.run_word({('', 0): words()})
        self.assertEqual(missing['failure_reason'], 'prediction_failure')
        self.assertIn("('', 1)", missing['error']['message'])
        self.assertTrue(missing['final_text_known'])
        exhausted = self.run_word(samples=SequenceClickSource([]))
        self.assertEqual(exhausted['failure_reason'], 'click_source_exhausted')
        for parameter, reason in [('max_presses_per_word', 'word_press_budget'), ('max_presses_per_phrase', 'phrase_press_budget')]:
            with self.subTest(parameter=parameter):
                result = self.run_word(**{parameter: 1})
                self.assertEqual(result['failure_reason'], reason)
                self.assertEqual(result['counts']['presses'], 1)
        attempt = self.run_word(samples=SequenceClickSource([ClickSample(), ClickSample(-2.4)]), max_attempts_per_word=1)
        self.assertEqual(attempt['failure_reason'], 'word_attempt_budget')
        # The same user can start another phrase without retaining failure diagnostics.
        user = BridgeSimulatedUser()
        user.run_phrase('hello', ZeroNoiseClickSource(), backend({('', 0): words()}))
        success = user.run_phrase('hello', ZeroNoiseClickSource(), backend({('', 0): words(), ('', 1): words(['hello']), ('hello ', 0): words()}))
        self.assertIsNone(success['error'])

    def test_invalid_targets_before_worker_and_invalid_samples(self):
        with patch('OneClick_Simulation.bridge_simulated_user.OneClickEngineBridge') as constructor:
            for text in ['', 'hi!', '123', 'café']:
                with self.assertRaises(ValueError):
                    BridgeSimulatedUser().run_phrase(text, ZeroNoiseClickSource(), backend({}))
            constructor.assert_not_called()
        for sample in [ClickSample(math.nan), ClickSample(math.inf), ClickSample(0, -1), ClickSample(0, math.inf)]:
            result = self.run_word(samples=SequenceClickSource([sample]))
            self.assertEqual(result['failure_reason'], 'invalid_click_sample')
            self.assertEqual(result['counts']['presses'], 0)

    def test_recorded_samples_keep_fixed_period_and_consume_in_order(self):
        source = SequenceClickSource([{'Click Time Relative (s)': '0.05', 'Dead Time (s)': '7.3', 'Clock Period (s)': '2.2'},
                                      {'Click Time Relative (s)': '0', 'Dead Time (s)': ''}])
        result = self.run_word(samples=source)
        self.assertTrue(result['completed'], result['error'])
        self.assertEqual(presses(result)[0]['intent']['source_period_s'], 2.2)
        self.assertEqual(presses(result)[0]['intent']['dead_rotations_s'], 7.28)
        self.assertEqual(result['final_snapshot']['letters']['period'], 3.64)
        self.assertIsNone(source.sample())

    def test_worker_failure_retains_trace_and_cleans_up(self):
        live = []
        real = OneClickEngineBridge
        def factory(**kwargs):
            bridge = real(**kwargs)
            live.append(bridge)
            return bridge
        class KillOnSample:
            def sample(self):
                live[0]._proc.kill()
                live[0]._proc.wait()
                return ClickSample()
        with patch('OneClick_Simulation.bridge_simulated_user.OneClickEngineBridge', side_effect=factory):
            result = self.run_word(samples=KillOnSample())
        self.assertEqual(result['failure_reason'], 'bridge_failure')
        self.assertFalse(result['final_text_known'])
        self.assertEqual(result['trace']['events'][-1]['status'], 'error')
        self.assertIsNotNone(live[0]._proc.poll())
        self.assertTrue(all(not thread.is_alive() for thread in live[0]._threads))
        partial = replay_trace(result['trace'])
        self.assertFalse(partial['complete_trace'])
        self.assertEqual(partial['snapshot'], result['final_snapshot'])

    def test_cli_writes_separate_result_and_replay_files(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(main(['--output-dir', directory]), 0)
            result = json.loads((Path(directory) / 'result.json').read_text())
            trace = json.loads((Path(directory) / 'trace.json').read_text())
            self.assertNotIn('trace', result)
            self.assertEqual(result['final_text'], 'hello world ')
            self.assertEqual(replay_trace(trace)['snapshot'], result['final_snapshot'])


class SchedulerTests(unittest.TestCase):
    def test_residual_next_noon_early_wrap_and_dead_rotations(self):
        group = {'period': 2.0, 'num_divs_time': 50, 'latest_time': 10.0, 'phases': [10]}
        target, dead = FrameScheduler.press_time(10.03, group, 0, ClickSample(0.08))
        self.assertAlmostEqual(target, 10.68)  # uses frame time, not now + phase delay
        self.assertEqual(dead, 0)
        early, _ = FrameScheduler.press_time(10.59, group, 0, ClickSample(-0.08))
        self.assertAlmostEqual(early, 12.52)
        delayed, dead = FrameScheduler.press_time(10.03, group, 0, ClickSample(0.08, 4.3))
        self.assertAlmostEqual(delayed, 14.68)
        self.assertEqual(dead, 4)
        passed, _ = FrameScheduler.press_time(10.61, group, 0, ClickSample())
        self.assertAlmostEqual(passed, 12.6)

    def test_frame_press_tie_order_and_no_reaiming(self):
        events = []
        group = {'period': 2, 'num_divs_time': 50}
        scheduler = FrameScheduler({'letters': group, 'words': group}, events.append)
        scheduler.advance_to(0.12)
        events.append({'type': 'space', 'time': scheduler.now})
        self.assertEqual([(event.get('group'), event['time']) for event in events],
                         [('letters', 0.04), ('words', 0.04), ('letters', 0.08), ('words', 0.08),
                          ('letters', 0.12), ('words', 0.12), (None, 0.12)])
        self.assertEqual(scheduler.frame_index, 3)


if __name__ == '__main__':
    unittest.main()
