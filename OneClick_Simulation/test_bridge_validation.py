"""Acceptance artifacts and deliberate corruption checks against the real bridge."""
import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from OneClick_Bridge import OneClickEngineBridge
from .bridge_validate import main, audit_directory, check_scenario
from .bridge_validation import validate_run, _press_timing
from .bridge_validation_scenarios import fixed_scenarios
from .bridge_simulated_user import (BridgeSimulatedUser, ClickSample, FixedPredictionBackend,
                                   FrameScheduler, SequenceClickSource, replay_trace)


class ValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        with contextlib.redirect_stdout(io.StringIO()):
            code = main(['--output-dir', str(cls.root)])
        if code:
            raise AssertionError((cls.root / 'summary.json').read_text())
        cls.cases = {}
        for spec in fixed_scenarios():
            directory = cls.root / spec['name']
            cls.cases[spec['name']] = tuple(json.loads((directory / name).read_text()) for name in ('result.json', 'trace.json', 'validation.json'))

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def pair(self, name='word'):
        result, trace, _ = self.cases[name]
        return copy.deepcopy(result), copy.deepcopy(trace)

    def run_timed_word(self, samples, **options):
        spec = fixed_scenarios()[0]
        return BridgeSimulatedUser(**options).run_phrase(
            spec['target'], SequenceClickSource(samples),
            FixedPredictionBackend(spec['characters'], spec['word_responses']))

    def test_timing_sample_corruption_is_rejected_at_press(self):
        cases = [('offset_s', 1000, 'press.time'), ('offset_s', .08, 'press.time'),
                 ('dead_time_s', -10, 'intent.dead_time_s'),
                 ('dead_time_s', 7.3, 'intent.dead_rotations_s'),
                 ('dead_rotations_s', 1200, 'intent.dead_rotations_s'),
                 ('source_period_s', -1, 'intent.source_period_s'),
                 ('offset_s', True, 'intent.offset_s')]
        for field, value, error_field in cases:
            with self.subTest(field=field, value=value):
                result, trace = self.pair()
                index = next(i for i, r in enumerate(trace['events']) if r['event']['type'] == 'space')
                trace['events'][index]['intent'][field] = value
                report = validate_run(result, trace)
                self.assertFalse(report['valid'])
                self.assertEqual(report['errors'][0]['field'], error_field)
                self.assertEqual(report['errors'][0]['event_index'], index)

    def test_action_choice_checkpoint_cannot_be_replaced_by_later_frames(self):
        for field in ('event_index', 'time'):
            result, trace = self.pair()
            index = next(i for i, r in enumerate(trace['events']) if r['event']['type'] == 'space')
            checkpoint = trace['events'][index]['intent']['scheduled_from']
            checkpoint[field] = index - 1 if field == 'event_index' else trace['events'][index-1]['event']['time']
            report = validate_run(result, trace)
            self.assertFalse(report['valid'])
            self.assertEqual(report['errors'][0]['field'], 'intent.scheduled_from.' + field)
            self.assertEqual(report['errors'][0]['event_index'], index)

    def test_independent_auditor_catches_scheduler_ignoring_sample(self):
        actual_scheduler = FrameScheduler.press_time
        for sample, expected_field in [(ClickSample(.08), 'press.time'),
                                       (ClickSample(0, 7.3), 'intent.dead_rotations_s')]:
            with self.subTest(sample=sample):
                # Simulate a scheduler regression with fully genuine engine events,
                # hashes, counters, and predictions. Replay alone would still pass.
                def ignore_sample(now, group, clock_id, sample):
                    return actual_scheduler(now, group, clock_id, ClickSample())
                with patch.object(FrameScheduler, 'press_time', side_effect=ignore_sample):
                    result = self.run_timed_word([sample, ClickSample()])
                self.assertTrue(result['completed'], result['error'])
                self.assertTrue(replay_trace(result['trace'])['complete_trace'])
                report = validate_run(result, result['trace'])
                self.assertFalse(report['valid'])
                self.assertEqual(report['errors'][0]['field'], expected_field)

    def test_valid_early_late_dead_time_and_frame_tie_schedules(self):
        for offset, dead, options in [(.05, 7.3, {}), (-.05, 0, {}),
                                      (0, 0, {}), (.02, 0, {'rotate_index': 20}),
                                      (0, 0, {'character_prediction_latency_s': .2,
                                              'word_prediction_latency_s': .1})]:
            with self.subTest(offset=offset, dead=dead, options=options):
                result = self.run_timed_word([ClickSample(offset, dead, 2.2), ClickSample()], **options)
                self.assertTrue(result['completed'], result['error'])
                report = validate_run(result, result['trace'])
                self.assertTrue(report['valid'], report['errors'])
                schedule = report['reconstructed']['press_schedule']
                self.assertEqual(len(schedule), 2)
                self.assertEqual(schedule[0]['offset_s'], offset)
                self.assertAlmostEqual(schedule[0]['time'], schedule[0]['expected_time'])
                if dead:
                    self.assertAlmostEqual(schedule[0]['dead_rotations_s'], 7.28)
                self.assertTrue(all(item['time'] >= item['scheduled_from']['time'] + .001 - 1e-12 for item in schedule))

    def test_auditor_derives_known_noon_wrap_and_dead_time_cases(self):
        # Hand-specified cases use a last-frame time different from action time.
        group = {'period': 2., 'num_divs_time': 50, 'latest_time': 10., 'phases': [10],
                 'clocks': [{'id': 0, 'active': True}]}
        for now, offset, dead, expected, rotations in [(10.03, .08, 0, 10.68, 0),
                                                      (10.59, -.08, 0, 12.52, 0),
                                                      (10.03, .08, 4.3, 14.68, 4),
                                                      (10.6, 0, 0, 12.6, 0)]:
            for kind in ('space', 'enter'):
                intent = {'clock_id': 0, 'offset_s': offset, 'dead_time_s': dead,
                          'dead_rotations_s': rotations, 'source_period_s': 6.,
                          'scheduled_from': {'event_index': 4, 'time': now}}
                decision = (4, now, {'letters': group, 'words': group})
                report = _press_timing(intent, {'type': kind, 'time': expected}, decision, 10., .04, 5)
                self.assertAlmostEqual(report['expected_time'], expected)

    def test_auditor_rejects_press_before_minimum_interval(self):
        actual_scheduler = FrameScheduler.press_time
        calls = 0
        def too_soon(now, group, clock_id, sample):
            nonlocal calls
            calls += 1
            return (now + .0005, 0) if calls == 1 else actual_scheduler(now, group, clock_id, sample)
        with patch.object(FrameScheduler, 'press_time', side_effect=too_soon):
            result = self.run_timed_word([ClickSample(), ClickSample()])
        self.assertTrue(result['completed'], result['error'])
        self.assertTrue(replay_trace(result['trace'])['complete_trace'])
        report = validate_run(result, result['trace'])
        self.assertFalse(report['valid'])
        self.assertEqual(report['errors'][0]['field'], 'press.minimum_interval')

    def test_settings_are_saved_and_custom_budget_boundaries_pass(self):
        settings = {'max_attempts_per_word': 1, 'max_presses_per_word': 2, 'max_presses_per_phrase': 2}
        result = self.run_timed_word([ClickSample(), ClickSample()], **settings)
        self.assertTrue(result['completed'])
        self.assertEqual(result['simulation_settings'], settings)
        self.assertEqual(result['trace']['simulation_settings'], settings)
        report = validate_run(result, result['trace'])
        self.assertTrue(report['valid'], report['errors'])
        self.assertEqual(report['reconstructed']['simulation_settings'], settings)
        result['simulation_settings']['max_presses_per_word'] = 99
        self.assertEqual(result['trace']['simulation_settings'], settings)
        failed = self.run_timed_word([ClickSample(), ClickSample()], max_presses_per_phrase=1)
        self.assertEqual(failed['failure_reason'], 'phrase_press_budget')
        self.assertEqual(failed['simulation_settings']['max_presses_per_phrase'], 1)
        self.assertEqual(failed['trace']['simulation_settings'], failed['simulation_settings'])

    def test_invalid_inconsistent_and_exceeded_budgets_are_rejected(self):
        result, trace = self.pair()
        result['simulation_settings']['max_presses_per_word'] += 1
        self.assertEqual(validate_run(result, trace)['errors'][0]['field'], 'simulation_settings')
        for key in ('max_attempts_per_word', 'max_presses_per_word', 'max_presses_per_phrase'):
            for value in (0, -1, True, 1.5, '2'):
                with self.subTest(key=key, value=value):
                    result, trace = self.pair()
                    for artifact in (result, trace):
                        artifact['simulation_settings'][key] = value
                    report = validate_run(result, trace)
                    self.assertFalse(report['valid'])
                    self.assertEqual(report['errors'][0]['field'], 'simulation_settings.' + key)
            result, trace = self.pair('midword_undo' if key == 'max_attempts_per_word' else 'word')
            for artifact in (result, trace):
                artifact['simulation_settings'][key] = 1
            report = validate_run(result, trace)
            self.assertFalse(report['valid'])
            self.assertEqual(report['errors'][0]['field'], 'budget.' + key)
            self.assertIsInstance(report['errors'][0]['event_index'], int)

    def test_older_traces_replay_but_cannot_claim_full_timing_audit(self):
        result, trace = self.pair()
        for artifact in (result, trace):
            del artifact['simulation_settings']
        for record in trace['events']:
            if record['intent']:
                del record['intent']['scheduled_from']
        self.assertTrue(replay_trace(trace)['complete_trace'])
        self.assertEqual(validate_run(result, trace)['errors'][0]['field'], 'simulation_settings')
        result, trace = self.pair()
        index = next(i for i, r in enumerate(trace['events']) if r['event']['type'] == 'space')
        del trace['events'][index]['intent']['scheduled_from']
        self.assertEqual(validate_run(result, trace)['errors'][0]['field'], 'intent.scheduled_from')

    def test_all_scenarios_have_independent_expected_outcomes(self):
        summary = json.loads((self.root / 'summary.json').read_text())
        self.assertTrue(summary['valid'])
        self.assertTrue(summary['finished'])
        self.assertTrue(all(s['complete_trace'] for s in summary['scenarios']))
        self.assertEqual([s['scenario'] for s in summary['scenarios']], [s['name'] for s in fixed_scenarios()])
        for name, (result, trace, report) in self.cases.items():
            self.assertTrue(report['valid'], name)
            self.assertTrue(report['complete_trace'])
            self.assertEqual(report['verified_events'], len(trace['events']))
            self.assertNotIn('trace', result)
            self.assertEqual(report['reconstructed']['counts'], result['counts'])
        missed = self.cases['missed_undo'][2]['reconstructed']
        self.assertEqual(missed['counts']['undo_attempts'], 3)
        self.assertEqual(missed['actual_undo_selections'], 2)
        midword = self.cases['midword_undo'][2]['reconstructed']['selections'][0]
        self.assertEqual(midword['observations_before'], 1)
        self.assertEqual(midword['observations_after'], 0)
        delayed = self.cases['incorrect_second_word_delayed'][2]['reconstructed']
        self.assertAlmostEqual(delayed['startup_elapsed_s'], .2)
        self.assertAlmostEqual(delayed['typing_prediction_wait_s'], .7)
        self.assertAlmostEqual(delayed['prediction_wait_s'], .9)

    def test_result_corruption_is_rejected(self):
        for field in ('counts', 'attempts_by_word', 'presses_by_word', 'final_text', 'final_snapshot',
                      'completed', 'final_text_known', 'failure_reason', 'error',
                      'simulated_elapsed_s', 'startup_elapsed_s', 'typing_elapsed_s',
                      'prediction_wait_s', 'typing_prediction_wait_s'):
            with self.subTest(field=field):
                result, trace = self.pair()
                if field == 'counts': result[field]['enter'] += 1
                elif field in ('attempts_by_word', 'presses_by_word'): result[field][0] += 1
                elif field == 'final_text': result[field] = 'wrong '
                elif field == 'final_snapshot': result[field]['typed'] = 'wrong '
                elif field in ('completed', 'final_text_known'): result[field] = False
                elif field in ('failure_reason', 'error'): result[field] = 'invented failure'
                else: result[field] += .2
                report = validate_run(result, trace)
                self.assertFalse(report['valid'])
                self.assertEqual(report['errors'][0]['field'], field)
                self.assertTrue(report['complete_trace'])
        result, trace = self.pair()
        result['counts']['space'] = True  # bool is not an integer counter
        self.assertFalse(validate_run(result, trace)['valid'])

    def test_trace_corruption_is_rejected_at_its_event(self):
        for field in ('selection', 'text', 'observations', 'snapshot', 'effects', 'intent', 'time'):
            with self.subTest(field=field):
                result, trace = self.pair()
                index = next(i for i, r in enumerate(trace['events']) if r['event']['type'] == 'enter')
                record = trace['events'][index]
                if field == 'selection': record['observed']['selection']['id'] = 85
                elif field == 'text': record['observed']['typed'] = 'wrong '
                elif field == 'observations': record['observed']['observation_count'] = 20
                elif field == 'snapshot': record['snapshot_sha256'] = '0'*64
                elif field == 'effects': record['effects'] = []
                elif field == 'intent': record['intent']['word_position'] = 4
                else: record['event']['time'] += .5
                report = validate_run(result, trace)
                self.assertFalse(report['valid'])
                self.assertEqual(report['errors'][0]['event_index'], index)
                self.assertEqual(report['verified_events'], index)

    def test_response_order_and_pending_press_checks(self):
        result, trace = self.pair('incorrect_second_word_delayed')
        trace['events'][1], trace['events'][2] = trace['events'][2], trace['events'][1]
        self.assertFalse(validate_run(result, trace)['valid'])
        result, trace = self.pair()
        response_index = next(i for i, r in enumerate(trace['events']) if r['event']['type'] == 'prediction-response')
        # Put a press before the final startup response without advancing time.
        press = copy.deepcopy(next(r for r in trace['events'] if r['event']['type'] == 'space'))
        press['event']['time'] = trace['events'][response_index]['event']['time']
        trace['events'].insert(response_index, press)
        report = validate_run(result, trace)
        self.assertFalse(report['valid'])
        self.assertEqual(report['errors'][0]['field'], 'press.pending_predictions')

    def test_request_timing_and_matrix_metadata_corruption(self):
        for field in ('requested_time', 'due_time', 'delivered_time'):
            result, trace = self.pair()
            trace['prediction_requests'][0][field] += .3
            report = validate_run(result, trace)
            self.assertFalse(report['valid'])
            self.assertEqual(report['errors'][0]['field'], 'request.' + field)
        for field in ('effective_transition_matrix', 'effective_transition_matrix_sha256', 'character_responses'):
            result, trace = self.pair()
            metadata = trace['prediction_metadata']
            if field == 'effective_transition_matrix': metadata[field][0][0] += 1
            elif field == 'effective_transition_matrix_sha256': metadata[field] = '0'*64
            else: metadata[field]['a']['results'][0]['logProb'] += 1
            result['prediction_metadata'] = copy.deepcopy(metadata)  # both copies wrong
            report = validate_run(result, trace)
            self.assertFalse(report['valid'])
            self.assertEqual(report['errors'][0]['field'], field)
        result, trace = self.pair()
        trace['engine_metadata']['sourceSha256'] = '0'*64
        result['engine_metadata'] = copy.deepcopy(trace['engine_metadata'])
        self.assertFalse(validate_run(result, trace)['valid'])

    def test_truncated_and_failed_traces_never_pass(self):
        for stop in (0, 1, 28, -1):
            result, trace = self.pair()
            trace['events'] = trace['events'][:stop]
            self.assertFalse(validate_run(result, trace)['valid'])
        result, trace = self.pair()
        trace['events'][-1]['status'] = 'error'
        report = validate_run(result, trace)
        self.assertFalse(report['valid'])
        self.assertFalse(report['complete_trace'])
        self.assertEqual(report['verified_events'], len(trace['events'])-1)
        self.assertEqual(report['last_verified_text'], 'hello ')

    def test_scenario_expectations_are_not_taken_from_result_counters(self):
        report = copy.deepcopy(self.cases['word'][2])
        self.assertFalse(check_scenario(report, {'selection_ids': [85]})['valid'])
        report = copy.deepcopy(self.cases['word'][2])
        self.assertFalse(check_scenario(report, {'requires_correction': True})['valid'])

    def test_worker_cleanup_after_replay_error_and_mismatch(self):
        live = []
        def factory(**kwargs):
            bridge = OneClickEngineBridge(**kwargs)
            live.append(bridge)
            return bridge
        result, trace = self.pair()
        trace['events'][0]['snapshot_sha256'] = 'broken'
        with patch('OneClick_Simulation.bridge_validation.OneClickEngineBridge', side_effect=factory):
            self.assertFalse(validate_run(result, trace)['valid'])
        self.assertIsNotNone(live[-1]._proc.poll())
        self.assertTrue(all(not t.is_alive() for t in live[-1]._threads))
        def crashing_factory(**kwargs):
            bridge = factory(**kwargs)
            bridge._proc.kill()
            bridge._proc.wait()
            return bridge
        result, trace = self.pair()
        with patch('OneClick_Simulation.bridge_validation.OneClickEngineBridge', side_effect=crashing_factory):
            report = validate_run(result, trace)
        self.assertFalse(report['valid'])
        self.assertEqual(report['verified_events'], 0)
        self.assertTrue(all(not t.is_alive() for t in live[-1]._threads))

    def test_disk_audit_needs_no_models_and_reports_malformed_files(self):
        subprocess.run([sys.executable, '-S', '-m', 'OneClick_Simulation.bridge_validate', '--audit', str(self.root / 'word')],
                       check=True, capture_output=True, text=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'result.json').write_text('{invalid')
            self.assertFalse(audit_directory(path)['valid'])
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(['--audit', directory]), 1)
            self.assertFalse((path / 'validation.json').exists())  # auditing is read-only

    def test_cli_returns_nonzero_on_failed_required_scenario(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['--engine-repo', '/missing/checkout', '--output-dir', directory]), 1)
            summary = json.loads((Path(directory) / 'summary.json').read_text())
            self.assertFalse(summary['valid'])
            self.assertTrue(summary['finished'])
            self.assertEqual(len(summary['scenarios']), 6)
            self.assertTrue(all(not s['valid'] for s in summary['scenarios']))


if __name__ == '__main__':
    unittest.main()
