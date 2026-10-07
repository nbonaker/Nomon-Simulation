"""Bridge contract/integration tests; require Node 22+ and the sibling JS checkout."""
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

from . import (OneClickEngineBridge, BridgeConfigurationError, BridgeProtocolError,
               BridgeRemoteError, BridgeTimeoutError, BridgeProcessError)
from .demo import run_demo

REPO = Path(__file__).resolve().parents[2] / "Nomon-One-Click"
NODE = shutil.which("node")


class BridgeTests(unittest.TestCase):
    def assert_stopped(self, bridge):
        self.assertIsNotNone(bridge._proc.poll())
        self.assertTrue(all(not thread.is_alive() for thread in bridge._threads))

    def compare(self, actual, expected, path="root", baseline=False):
        if isinstance(expected, dict):
            self.assertIsInstance(actual, dict, path)
            extra = set()
            if baseline and path.endswith(".snapshot"):
                extra = {"rotate_index", "layout_revision"}
            elif baseline and (path.endswith(".letters") or path.endswith(".words")):
                extra = {"clocks"}
            self.assertEqual(set(actual) - extra, set(expected), path)
            for key, value in expected.items():
                self.compare(actual[key], value, path + "." + key, baseline)
        elif isinstance(expected, list):
            self.assertIsInstance(actual, list, path)
            self.assertEqual(len(actual), len(expected), path)
            for i, value in enumerate(expected):
                self.compare(actual[i], value, f"{path}[{i}]", baseline)
        elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
            discrete = any(part in path for part in (".phases[", ".valid_word_indices[")) or path.endswith((".id", "_seq", ".phase"))
            if discrete:
                self.assertEqual(actual, expected, path)
            else:
                self.assertAlmostEqual(actual, expected, delta=1e-10, msg=path)
        else:
            self.assertEqual(actual, expected, path)

    def ready(self, engine):
        initial = engine.dispatch({"type": "initialize", "time": 0})
        for effect in initial["effects"]:
            if effect["type"] == "predict-characters":
                engine.dispatch({"type": "character-response", "time": 0, "index": effect["index"],
                                 "data": {"results": [{"token": c, "logProb": -3.3} for c in "abcdefghijklmnopqrstuvwxyz'"]}})
            else:
                engine.dispatch({"type": "prediction-response", "time": 0,
                                 "data": {"prefix": [{"text": "hello", "logprob": -0.1}], "best": []}})
        return engine.get_snapshot()

    def test_all_baselines_and_complete_direct_engine_results(self):
        traces = json.loads((REPO / "tests/engine/fixtures/baseline.json").read_text())
        # Direct execution is a separate process with no worker or wire envelopes.
        script = """
            import {readFileSync} from 'node:fs';
            const {createOneClickEngine}=await import(process.argv[1]);
            const traces=JSON.parse(readFileSync(0,'utf8'));
            const results=traces.map(steps=>{
                const engine=createOneClickEngine();
                return steps.map(events=>{let result;for(const event of events)result=engine.dispatch(event);return result;});
            });
            process.stdout.write(JSON.stringify(results));
        """
        direct = subprocess.run([NODE, "--input-type=module", "-e", script,
                                 (REPO / "js/oneclick/engine/index.mjs").as_uri()],
                                input=json.dumps([[step["events"] for step in trace["steps"]] for trace in traces]),
                                text=True, capture_output=True, check=True, timeout=10)
        expected_direct = json.loads(direct.stdout)
        count = 0
        for trace_index, trace in enumerate(traces):
            with self.subTest(trace=trace["name"]), OneClickEngineBridge() as engine:
                for step_index, step in enumerate(trace["steps"]):
                    for event in step["events"]:
                        result = engine.dispatch(event)
                    self.compare(result, {"snapshot": step["snapshot"], "effects": step["effects"]}, baseline=True)
                    self.compare(result, expected_direct[trace_index][step_index])
                    count += 1
            self.assert_stopped(engine)
        self.assertEqual(count, 164)

    def test_independence_detachment_unicode_and_read_only_snapshot(self):
        with OneClickEngineBridge() as first, OneClickEngineBridge(config={"rotateIndex": 20, "useClickOffset": True}) as second:
            self.assertNotEqual(first.pid, second.pid)
            self.assertFalse(first.get_snapshot()["ready"])
            with self.assertRaises(BridgeRemoteError):
                first.dispatch({"type": "space", "time": 0})  # open did not initialize
            self.assert_stopped(first)
            saved = self.ready(second)
            self.assertEqual(saved["letters"]["period"], 0.84)
            self.assertEqual(second.get_snapshot(), saved)
            state = second.dispatch({"type": "prediction-response", "time": 0,
                                     "data": {"prefix": [{"text": "café'\n☕", "logprob": -0.2}], "best": []}})["snapshot"]
            self.assertEqual(state["words"]["clocks"][6]["label"], "café'\n☕")
            state["words"]["phases"][0] = -100
            state["words_by_letter"].clear()
            self.assertNotEqual(second.get_snapshot()["words"]["phases"][0], -100)
            self.assertIn("c", second.get_snapshot()["words_by_letter"])
            original_metadata = second.metadata
            second.metadata["sourceSha256"] = "changed"
            self.assertEqual(second.metadata, original_metadata)
        self.assert_stopped(second)

    def test_two_active_sessions_do_not_share_state(self):
        with OneClickEngineBridge() as a, OneClickEngineBridge() as b:
            self.ready(a)
            before = self.ready(b)
            a.dispatch({"type": "space", "time": 0.1})
            self.assertEqual(b.get_snapshot(), before)

    def test_configuration_forwarding_matches_direct_engine(self):
        events = [{"type": "initialize", "time": 0},
                  {"type": "prediction-response", "time": 0, "data": {"prefix": [{"text": "a"}], "best": []}},
                  {"type": "prediction-response", "time": 0, "data": {"prefix": [{"text": "a"}], "best": []}}]
        # Repeated commits bootstrap the offset model, then exercise compensated selection.
        for i in range(4):
            events += [{"type": "enter", "time": i + 0.1}, events[1]]
        events += [{"type": "frame", "group": "words", "time": 4}, {"type": "enter", "time": 4.58}]
        script = """
            import {readFileSync} from 'node:fs';
            const {createOneClickEngine}=await import(process.argv[1]);
            const e=createOneClickEngine({rotateIndex:20,useClickOffset:true});
            process.stdout.write(JSON.stringify(JSON.parse(readFileSync(0,'utf8')).map(x=>e.dispatch(x))));
        """
        direct = subprocess.run([NODE, "--input-type=module", "-e", script, (REPO / "js/oneclick/engine/index.mjs").as_uri()],
                                input=json.dumps(events), text=True, capture_output=True, check=True, timeout=10)
        with OneClickEngineBridge(config={"rotateIndex": 20, "useClickOffset": True}) as engine:
            self.assertEqual([engine.dispatch(event) for event in events], json.loads(direct.stdout))
            self.assertEqual(engine.get_snapshot()["last_selection"]["id"], 85)
        with OneClickEngineBridge(config={"rotateIndex": 20, "useClickOffset": False}) as engine:
            for event in events:
                engine.dispatch(event)
            self.assertEqual(engine.get_snapshot()["last_selection"]["id"], 0)

    def test_local_non_finite_validation_does_not_send_an_event(self):
        with OneClickEngineBridge() as engine:
            before = self.ready(engine)
            for value in (math.nan, math.inf, -math.inf, 2**60):
                with self.assertRaises(ValueError):
                    engine.dispatch({"type": "space", "time": value})
                self.assertEqual(engine.get_snapshot(), before)

    def test_missing_checkout_node_and_old_node(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(BridgeConfigurationError, "worker not found"):
                OneClickEngineBridge(engine_repo=directory)
        with self.assertRaisesRegex(BridgeConfigurationError, "Node executable not found"):
            OneClickEngineBridge(node_executable="/missing/quickclick-node")
        with patch("OneClick_Bridge.client.subprocess.run", return_value=subprocess.CompletedProcess([], 0, b"v18.0.0\n", b"")):
            with self.assertRaisesRegex(BridgeConfigurationError, "Node 22"):
                OneClickEngineBridge()

    def test_working_directory_independent_and_source_fingerprint(self):
        with tempfile.TemporaryDirectory(prefix="quickclick path ") as directory:
            previous = Path.cwd()
            try:
                os.chdir(directory)
                with OneClickEngineBridge() as engine:
                    self.assertEqual(engine.engine_repo, REPO)
                    first = engine.metadata
            finally:
                os.chdir(previous)
            copied = Path(directory) / "checkout"
            shutil.copytree(REPO / "js/oneclick/engine", copied / "js/oneclick/engine")
            helper = copied / "js/oneclick/engine/math.mjs"
            helper.write_text(helper.read_text() + "\n// fingerprint probe\n")
            with OneClickEngineBridge(engine_repo=copied) as engine:
                self.assertNotEqual(first["sourceSha256"], engine.metadata["sourceSha256"])
                self.assertEqual(Path(engine.metadata["enginePath"]), (copied / "js/oneclick/engine/index.mjs").resolve())

    def test_context_exception_and_idempotent_close_reap_worker(self):
        with self.assertRaisesRegex(RuntimeError, "demo failure"):
            with OneClickEngineBridge() as engine:
                raise RuntimeError("demo failure")
        self.assert_stopped(engine)
        engine.close()
        with self.assertRaises(BridgeProcessError):
            engine.get_snapshot()

    def test_child_process_must_not_use_or_close_inherited_session(self):
        with OneClickEngineBridge() as engine:
            with patch("OneClick_Bridge.client.os.getpid", return_value=engine._owner_pid + 1):
                with self.assertRaises(BridgeProcessError):
                    engine.get_snapshot()
                with self.assertRaises(BridgeProcessError):
                    engine.close()
            self.assertFalse(engine.get_snapshot()["ready"])

    def fake_repo(self, handler, open_action=None, close_action=None):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        worker = root / "js/oneclick/engine/worker.mjs"
        worker.parent.mkdir(parents=True)
        metadata = {"protocolVersion": 1, "nodeVersion": "v22.0.0", "enginePath": "/fake/index.mjs",
                    "sourceSha256": "a" * 64, "sourceFiles": ["index.mjs"]}
        normal_open = f"send({{metadata:{json.dumps(metadata)}}});"
        normal_close = "process.stdout.write(JSON.stringify({id:req.id,ok:true,result:{closed:true}})+'\\n',()=>process.exit(0));"
        worker.write_text("""
            import {createInterface} from 'node:readline';
            import {writeFileSync} from 'node:fs';
            writeFileSync(new URL('../../../pid',import.meta.url),String(process.pid));
            const lines=createInterface({input:process.stdin});
            lines.on('line',line=>{
                const req=JSON.parse(line);
                const send=result=>process.stdout.write(JSON.stringify({id:req.id,ok:true,result})+'\\n');
                if(req.op==='open'){OPEN}
                else if(req.op==='close'){CLOSE}
                else {HANDLER}
            });
        """.replace("OPEN", open_action or normal_open).replace("CLOSE", close_action or normal_close).replace("HANDLER", handler))
        return root

    def test_crash_malformed_mismatched_id_and_remote_error_are_fatal(self):
        cases = [
            ("process.stderr.write('crash detail\\n');process.exit(23);", BridgeProcessError, "crash detail"),
            ("process.stdout.write('not JSON\\n');", BridgeProtocolError, "Invalid JSON"),
            ("process.stdout.write(JSON.stringify({id:req.id+1,ok:true,result:{}})+'\\n');", BridgeProtocolError, "request ID"),
            ("process.stdout.write(JSON.stringify({id:req.id,ok:false,error:{code:'ENGINE_ERROR',message:'bad event'}})+'\\n');", BridgeRemoteError, "bad event"),
            ("process.stdout.write('{\"id\":'+req.id+',\"ok\":true,\"result\":{\"value\":1e999}}\\n');", BridgeProtocolError, "finite"),
        ]
        for code, error_type, message in cases:
            with self.subTest(message=message):
                engine = OneClickEngineBridge(engine_repo=self.fake_repo(code))
                with self.assertRaisesRegex(error_type, message):
                    engine.get_snapshot()
                self.assert_stopped(engine)
                with self.assertRaises(BridgeProcessError):
                    engine.get_snapshot()
                engine.close()

    def test_stalled_request_and_partial_line_are_bounded(self):
        for code in ("setInterval(()=>{},1000);", "process.stdout.write('{');setInterval(()=>{},1000);"):
            engine = OneClickEngineBridge(engine_repo=self.fake_repo(code), timeout_s=0.4)
            started = time.monotonic()
            with self.assertRaises(BridgeTimeoutError):
                engine.get_snapshot()
            self.assertLess(time.monotonic() - started, 2)
            self.assert_stopped(engine)

    def test_blocked_stdin_write_is_bounded(self):
        root = self.fake_repo("", open_action="""
            send({metadata:{protocolVersion:1,nodeVersion:'v22.0.0',enginePath:'/fake/index.mjs',sourceSha256:'a'.repeat(64),sourceFiles:[]}});
            lines.close();process.stdin.pause();setInterval(()=>{},1000);
        """)
        engine = OneClickEngineBridge(engine_repo=root, timeout_s=0.4)
        started = time.monotonic()
        with self.assertRaises(BridgeTimeoutError):
            engine.dispatch({"type": "ignored", "data": "x" * (2 * 1024 * 1024)})
        self.assertLess(time.monotonic() - started, 2)
        self.assert_stopped(engine)

    def test_stderr_flood_is_drained_and_tail_is_bounded(self):
        code = "process.stderr.write('x'.repeat(2*1024*1024)+'TAIL_MARKER',()=>send({ready:false}));"
        with OneClickEngineBridge(engine_repo=self.fake_repo(code)) as engine:
            self.assertEqual(engine.get_snapshot(), {"ready": False})
        self.assert_stopped(engine)
        self.assertLessEqual(len(engine._stderr_tail), 64 * 1024)
        self.assertTrue(engine._stderr_tail.endswith(b"TAIL_MARKER"))

    def test_bad_handshake_and_stalled_startup_reap_workers(self):
        cases = [
            ("send({metadata:{protocolVersion:99}});", BridgeProtocolError, 3),
            ("setInterval(()=>{},1000);", BridgeTimeoutError, 0.4),
        ]
        for action, error, timeout in cases:
            root = self.fake_repo("", open_action=action)
            with self.assertRaises(error):
                OneClickEngineBridge(engine_repo=root, timeout_s=timeout)
            pid = int((root / "pid").read_text())
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)

    def test_close_has_two_second_grace_then_terminates(self):
        engine = OneClickEngineBridge(engine_repo=self.fake_repo("send({});", close_action="setInterval(()=>{},1000);"))
        started = time.monotonic()
        engine.close()
        elapsed = time.monotonic() - started
        self.assertGreaterEqual(elapsed, 1.9)
        self.assertLess(elapsed, 4)
        self.assert_stopped(engine)
        engine.close()

    def test_demo_completes_word_and_undo_without_live_backend(self):
        messages = []
        result = run_demo(emit=messages.append)
        self.assertEqual(result["typed"], "")
        self.assertTrue(any("committed 'hello '" in message for message in messages))
        self.assertTrue(any("text restored" in message for message in messages))


if __name__ == "__main__":
    unittest.main()
