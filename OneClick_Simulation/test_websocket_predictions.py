"""Codec parity, controlled WebSocket I/O, timing, failure, and offline replay checks."""
from __future__ import annotations

import asyncio
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from .websocket_codec import ALPHABET, prediction_request, prediction_response, strict_loads
from .websocket_predictions import WebSocketPredictionBackend, WebSocketBackendError
from .bridge_simulated_user import BridgeSimulatedUser, ZeroNoiseClickSource, SequenceClickSource, ClickSample, replay_trace
from .bridge_validation import validate_run
from .bridge_validation_scenarios import fixed_scenarios
from OneClick_Bridge import BridgeError

ROOT = Path(__file__).resolve().parents[2]
BROWSER_REPO = Path(os.environ.get('QUICKCLICK_BROWSER_REPO', ROOT / 'Nomon-One-Click'))
BROWSER_REPLAY = BROWSER_REPO / 'tests/predictions/codec-replay.mjs'
FIXTURES = Path(__file__).with_name('bridge_fixtures') / 'websocket'
HAVE_WEBSOCKETS = importlib.util.find_spec('websockets') is not None


def word_effect(count=1):
    fixture = json.loads((FIXTURES / 'protocol-v1.json').read_text())
    row = {'distrib': [dict(text=c, logProb=p) for c, p in zip(ALPHABET, fixture['observation_scores'])]}
    return dict(type='predict-words', left='hello ', distribs=[copy.deepcopy(row) for _ in range(count)], numPrefix=3, numBest=3)


def envelope(request, data, **overrides):
    return dict(version=1, request_id=request['request_id'], session_id=request['session_id'],
                revision=request['revision'], model=request['model'], operation=request['operation'],
                status='ok', data=data, timings={'service_ms': 2.5, 'inference_ms': 2, 'queue_ms': .5}, **overrides)


class CodecTests(unittest.TestCase):
    def check_fixture_results(self):
        protocol = json.loads((FIXTURES / 'protocol-v1.json').read_text())
        words = json.loads((FIXTURES / protocol['word_fixture']).read_text())
        identity = protocol['identity']
        kwargs = dict(session_id=identity['sessionId'], request_id=identity['requestId'], revision=identity['revision'])
        cases, expected = [], []
        for i, c in enumerate(ALPHABET):
            effect = dict(type='predict-characters', left=c, index=i)
            data = {'characters': {'predictions': [dict(character=char, probability=dict(log_retained_joint_mass=p))
                                                 for char, p in zip(ALPHABET, protocol['character_scores'])]}}
            cases.append(dict(effect=effect, identity=identity, data=data))
            expected.append(dict(type='character-response', index=i, data={'results': [
                dict(token=char, logProb=p) for char, p in zip(ALPHABET, protocol['expected_character_scores'])]}))
        for example in words['cases']:
            cases.append(dict(effect=word_effect(example['observation_count']), identity=identity,
                              data={'words': {'predictions': words['candidates']}}))
            expected.append(dict(type='prediction-response', data=example['expected']))
        results = []
        for case, golden in zip(cases, expected):
            request = prediction_request(case['effect'], **kwargs)
            for key, value in protocol['request_defaults'].items():
                self.assertEqual(request[key], value)
            event = prediction_response(case['effect'], request, envelope(request, case['data']))
            self.assertEqual(event, golden)
            results.append(dict(request=request, event=event))
        return cases, results

    def test_fixtures_match_independent_expected_values(self):
        self.check_fixture_results()

    @unittest.skipUnless(BROWSER_REPLAY.is_file(), 'optional browser prediction codec checkout unavailable')
    def test_fixtures_match_browser_codec(self):
        cases, results = self.check_fixture_results()
        node = subprocess.run(['node', str(BROWSER_REPLAY)],
                              input=json.dumps(cases), capture_output=True, text=True, check=True, timeout=15)
        self.assertEqual(results, json.loads(node.stdout))

    def test_bad_scores_identities_observations_and_json_are_rejected(self):
        kwargs = dict(session_id='s', request_id='r', revision=1)
        for bad in (float('nan'), float('inf'), -float('inf'), True, '1', None):
            effect = word_effect(); effect['distribs'][0]['distrib'][0]['logProb'] = bad
            with self.assertRaises((ValueError, TypeError)):
                prediction_request(effect, **kwargs)
        effect = word_effect(); effect['distribs'][0]['distrib'][0]['text'] = 'b'
        with self.assertRaises(ValueError): prediction_request(effect, **kwargs)
        effect = word_effect(); request = prediction_request(effect, **kwargs)
        response = envelope(request, {'words': {'predictions': []}})
        for key in ('version', 'request_id', 'session_id', 'revision', 'model', 'operation'):
            bad = copy.deepcopy(response); bad[key] = 'wrong'
            with self.assertRaises(ValueError): prediction_response(effect, request, bad)
        for bad in (float('nan'), float('inf'), True, '1', None):
            bad_response = copy.deepcopy(response)
            bad_response['data']['words']['predictions'] = [dict(word='hello', probability=dict(log_retained_joint_mass=bad))]
            with self.assertRaises(ValueError): prediction_response(effect, request, bad_response)
        for raw in ('{"x":NaN}', '{"x":Infinity}', '{"x":1e999}', '{"x":1,"x":2}'):
            with self.assertRaises(ValueError): strict_loads(raw)

    def test_configuration_and_missing_optional_dependency_fail_actionably(self):
        for url in ('ws://localhost:8766', 'http://user:pass@host', 'http://host/path', 'http://host?token=x', 'http://host:0'):
            with self.assertRaises(ValueError): WebSocketPredictionBackend(url)
        with self.assertRaises(ValueError): WebSocketPredictionBackend(model=None)
        for timeout in (-1, 0, float('nan'), float('inf'), True):
            with self.assertRaises(ValueError): WebSocketPredictionBackend(timeout_s=timeout)
        backend = WebSocketPredictionBackend()
        with patch.dict(sys.modules, {'websockets.asyncio.client': None}):
            with self.assertRaisesRegex(WebSocketBackendError, 'requirements-websocket'):
                with backend: pass
        self.assertFalse(backend.thread.is_alive())

    def test_optional_client_is_not_imported_by_fixture_or_replay_modules(self):
        code = "import sys; import OneClick_Simulation.bridge_simulated_user; assert not any(m == 'websockets' or m.startswith('websockets.') for m in sys.modules); assert 'torch' not in sys.modules; assert 'textslinger' not in sys.modules"
        subprocess.run([sys.executable, '-c', code], check=True, timeout=10)


class TestServer:
    """Small actual WS service with deterministic, independently specified replies."""
    __test__ = False
    def __init__(self, fixture=None, mode='ok', delay=0):
        self.fixture, self.mode, self.delay = fixture, mode, delay
        self.requests = []
        self.connections = 0
        self.loop = asyncio.new_event_loop()
        self.ready = threading.Event()
        self.thread = threading.Thread(target=self._thread, daemon=True)
        self.error = None

    def __enter__(self):
        self.thread.start()
        if not self.ready.wait(5): raise RuntimeError('test server startup stalled')
        if self.error: raise self.error
        self.url = f'http://127.0.0.1:{self.port}'
        return self

    def _thread(self):
        asyncio.set_event_loop(self.loop)
        async def serve():
            from websockets.asyncio.server import serve
            async with serve(self.handler, '127.0.0.1', 0, subprotocols=['textslinger.v1'],
                             process_request=self.health, close_timeout=.2) as server:
                self.port = server.sockets[0].getsockname()[1]
                self.stop = asyncio.Event()
                self.ready.set()
                await self.stop.wait()
        try:
            self.loop.run_until_complete(serve())
        except Exception as exc:
            self.error = exc; self.ready.set()
        finally:
            self.loop.close()

    async def health(self, connection, request):
        if request.path == '/health':
            from websockets.http11 import Response
            from websockets.datastructures import Headers
            body = json.dumps({'status': 'ready', 'workers': [{'id': 'fixture', 'model': 'default', 'backend': 'mock'}]}).encode()
            return Response(200, 'OK', Headers({'Content-Length': str(len(body)), 'Connection': 'close'}), body)

    async def handler(self, socket):
        self.connections += 1
        if self.mode == 'setup-stall':
            await socket.wait_closed(); return
        await socket.send(json.dumps(dict(type='ready', version=2 if self.mode == 'version' else 1)))
        try:
            async for message in socket:
                request = json.loads(message)
                self.requests.append(request)
                if self.delay: await asyncio.sleep(self.delay)
                if self.mode == 'stall': await socket.wait_closed(); return
                if self.mode == 'disconnect': await socket.close(); return
                if self.mode == 'malformed': await socket.send('not-json'); continue
                if self.mode == 'duplicate': await socket.send('{"request_id":"a","request_id":"b"}'); continue
                if request['operation'] == 'characters':
                    rows = self.fixture['characters']['results'] if self.fixture else [dict(token=c, logProb=-3.3) for c in ALPHABET]
                    data = {'characters': {'predictions': [dict(character=r['token'], probability=dict(log_retained_joint_mass=r['logProb'])) for r in rows]}}
                else:
                    if self.fixture:
                        candidates = self.fixture['word_responses'][(request['left_context'], len(request['input_events']))]
                        predictions = [dict(word=p['text'], probability=dict(log_retained_joint_mass=p['logprob']))
                                       for category in ('prefix', 'best') for p in candidates[category]]
                    else:
                        predictions = [dict(word='hello', probability=dict(log_retained_joint_mass=-.1))]
                    data = {'words': {'predictions': predictions}}
                response = envelope(request, data)
                if self.mode == 'wrong-id': response['request_id'] = 'wrong'
                if self.mode == 'error': response.update(status='overloaded', error='test overload')
                if self.mode == 'nonfinite': response['data'] = {'words': {'predictions': [dict(word='hello', probability=dict(log_retained_joint_mass=float('nan')))]}}
                await socket.send(json.dumps(response))
        except Exception:
            # Client failure tests intentionally abort the transport.
            pass

    def __exit__(self, *_):
        if hasattr(self, 'stop'): self.loop.call_soon_threadsafe(self.stop.set)
        self.thread.join(5)
        if self.thread.is_alive(): raise RuntimeError('test service did not stop')


@unittest.skipUnless(HAVE_WEBSOCKETS, 'optional websockets client not installed')
class TransportTests(unittest.TestCase):
    def test_persistent_connection_provenance_ids_detached_results_and_cleanup(self):
        with TestServer() as server:
            backend = WebSocketPredictionBackend(server.url)
            with backend:
                first = backend.predict(word_effect()); first['prefix'].clear()
                second = backend.predict(word_effect())
                self.assertTrue(second['prefix'])
                self.assertEqual(backend.metadata()['server_health']['status'], 'ready')
                self.assertEqual(server.connections, 1)
                self.assertEqual(len({r['session_id'] for r in server.requests}), 1)
                self.assertEqual(len({r['request_id'] for r in server.requests}), 2)
                self.assertEqual([r['revision'] for r in server.requests], [1, 2])
                self.assertEqual(backend.last_exchange()['status'], 'ok')
                self.assertNotIn('target', server.requests[0])
            backend.close()
            self.assertFalse(backend.thread.is_alive())
            with self.assertRaises(WebSocketBackendError): backend.predict(word_effect())

    def test_malformed_error_disconnect_stall_and_protocol_failure_cleanup(self):
        for mode in ('version', 'setup-stall', 'malformed', 'duplicate', 'wrong-id', 'error', 'nonfinite', 'disconnect', 'stall'):
            with self.subTest(mode=mode), TestServer(mode=mode) as server:
                backend = WebSocketPredictionBackend(server.url, timeout_s=.15, connect_timeout_s=.3)
                started = time.monotonic()
                with self.assertRaises(WebSocketBackendError):
                    with backend:
                        backend.predict(word_effect())
                self.assertLess(time.monotonic() - started, 4)
                self.assertFalse(backend.thread.is_alive())
                self.assertIsNotNone(backend.metadata()['failure'])
                json.dumps(backend.last_exchange(), allow_nan=False)

    def test_independent_sessions_and_exception_exit(self):
        with TestServer() as server:
            a, b = WebSocketPredictionBackend(server.url), WebSocketPredictionBackend(server.url)
            with self.assertRaisesRegex(RuntimeError, 'intentional'):
                with a, b:
                    a.predict(word_effect()); b.predict(word_effect())
                    self.assertNotEqual(a.session_id, b.session_id)
                    raise RuntimeError('intentional')
            self.assertFalse(a.thread.is_alive()); self.assertFalse(b.thread.is_alive())

    def test_missing_service_is_not_retried_and_leaves_no_thread(self):
        with TestServer() as server:
            url = server.url
        backend = WebSocketPredictionBackend(url, connect_timeout_s=.3)
        with self.assertRaises(WebSocketBackendError):
            with backend: pass
        self.assertFalse(backend.thread.is_alive())
        self.assertEqual(backend.state, 'closed')


@unittest.skipUnless(HAVE_WEBSOCKETS, 'optional websockets client not installed')
class SimulationTests(unittest.TestCase):
    def run_spec(self, spec, delay=0):
        with TestServer(spec, delay=delay) as server:
            backend = WebSocketPredictionBackend(server.url)
            user = BridgeSimulatedUser(**spec['options'])
            result = user.run_phrase(spec['target'], SequenceClickSource([ClickSample(x) for x in spec['offsets']]), backend)
        self.assertFalse(backend.thread.is_alive())
        self.assertTrue(result['completed'], result['error'])
        trace = result.pop('trace')
        # Audit serialized data after the service is stopped.
        result, trace = json.loads(json.dumps([result, trace], allow_nan=False))
        report = validate_run(result, trace)
        self.assertTrue(report['valid'], report['errors'])
        self.assertTrue(replay_trace(trace)['complete_trace'])
        self.assertEqual(result['counts'], spec['expected']['counts'])
        return result, trace

    def test_all_six_acceptance_scenarios_over_websocket_and_offline_audit(self):
        for spec in fixed_scenarios():
            with self.subTest(scenario=spec['name']): self.run_spec(spec)

    def test_wall_clock_delay_does_not_change_simulated_time_or_engine_events(self):
        spec = fixed_scenarios()[1]
        fast, fast_trace = self.run_spec(spec)
        slow, slow_trace = self.run_spec(spec, delay=.005)
        self.assertEqual(fast['simulated_elapsed_s'], slow['simulated_elapsed_s'])
        self.assertEqual(fast_trace['events'], slow_trace['events'])
        self.assertGreater(sum(r['transport']['client_elapsed_s'] for r in slow_trace['prediction_requests']),
                           sum(r['transport']['client_elapsed_s'] for r in fast_trace['prediction_requests']))

    def test_fixed_character_matrix_override_keeps_only_words_on_the_socket(self):
        spec = fixed_scenarios()[0]
        with TestServer(spec) as server:
            backend = WebSocketPredictionBackend(server.url)
            user = BridgeSimulatedUser(character_transition_source='fixed', fixed_character_responses=spec['characters'])
            result = user.run_phrase('hello', ZeroNoiseClickSource(), backend)
        self.assertTrue(result['completed'], result['error'])
        self.assertTrue(all(r['operation'] == 'words' for r in server.requests))
        trace = result.pop('trace')
        self.assertTrue(validate_run(result, trace)['valid'])
        self.assertFalse(backend.thread.is_alive())

    def test_transport_tampering_is_rejected_without_a_server(self):
        result, trace = self.run_spec(fixed_scenarios()[0])
        for field in ('wire_request', 'wire_response', 'client_elapsed_s'):
            altered = copy.deepcopy(trace)
            transport = altered['prediction_requests'][-1]['transport']
            if field == 'wire_request': transport[field]['left_context'] = 'wrong '
            elif field == 'wire_response': transport[field]['revision'] = -1
            else: transport[field] = -1
            self.assertFalse(validate_run(result, altered)['valid'])

        for change in ('session', 'revision', 'duplicate-id'):
            altered = copy.deepcopy(trace)
            wire = altered['prediction_requests'][-1]['transport']['wire_request']
            reply = altered['prediction_requests'][-1]['transport']['wire_response']
            key = {'session': 'session_id', 'revision': 'revision', 'duplicate-id': 'request_id'}[change]
            value = altered['prediction_requests'][0]['transport']['wire_request'][key] if change == 'duplicate-id' else 'changed'
            wire[key] = reply[key] = value
            self.assertFalse(validate_run(result, altered)['valid'])

    def test_prediction_failure_preserves_diagnostics_and_closes_session(self):
        with TestServer(mode='malformed') as server:
            backend = WebSocketPredictionBackend(server.url)
            result = BridgeSimulatedUser().run_phrase('hello', ZeroNoiseClickSource(), backend)
        self.assertFalse(result['completed'])
        self.assertEqual(result['failure_reason'], 'prediction_failure')
        self.assertEqual(result['trace']['prediction_requests'][0]['transport']['status'], 'error')
        self.assertFalse(backend.thread.is_alive())
        self.assertFalse(validate_run({k:v for k,v in result.items() if k != 'trace'}, result['trace'])['valid'])

    def test_bridge_failure_closes_backend_and_reaps_node(self):
        with TestServer(fixed_scenarios()[0]) as server:
            backend = WebSocketPredictionBackend(server.url)
            user = BridgeSimulatedUser()
            with patch.object(user, '_type_phrase', side_effect=BridgeError('intentional bridge failure')):
                result = user.run_phrase('hello', ZeroNoiseClickSource(), backend)
        self.assertEqual(result['failure_reason'], 'bridge_failure')
        self.assertFalse(backend.thread.is_alive())
        self.assertIsNotNone(user.bridge._proc.poll())

    def test_invalid_target_does_not_open_connection_and_exhaustion_closes_it(self):
        backend = WebSocketPredictionBackend()
        with self.assertRaises(ValueError):
            BridgeSimulatedUser().run_phrase('hello!', ZeroNoiseClickSource(), backend)
        self.assertIsNone(backend.thread)
        with TestServer(fixed_scenarios()[0]) as server:
            backend = WebSocketPredictionBackend(server.url)
            result = BridgeSimulatedUser().run_phrase('hello', SequenceClickSource([]), backend)
        self.assertFalse(result['completed'])
        self.assertFalse(backend.thread.is_alive())

    def test_cli_saves_connection_failure_and_rejects_conflicting_options(self):
        with TestServer() as server:
            url = server.url
        with tempfile.TemporaryDirectory() as folder:
            call = subprocess.run([sys.executable, '-m', 'OneClick_Simulation.bridge_pilot', '--backend',
                'textslinger-websocket', '--prediction-server', url, '--output-dir', folder],
                text=True, capture_output=True, timeout=15)
            self.assertEqual(call.returncode, 1, call.stderr)
            result = json.loads(Path(folder, 'result.json').read_text())
            self.assertEqual(result['failure_reason'], 'prediction_failure')
            self.assertIsNotNone(result['error'])
        call = subprocess.run([sys.executable, '-m', 'OneClick_Simulation.bridge_pilot', '--backend',
            'textslinger-websocket', '--model', 'invalid.kenlm'], capture_output=True, text=True, timeout=10)
        self.assertEqual(call.returncode, 2)
        self.assertIn('omit local model options', call.stderr)


if __name__ == '__main__':
    unittest.main()
