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
from .bridge_validation import validate_run
from .bridge_validation_scenarios import fixed_scenarios


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
