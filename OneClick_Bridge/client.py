"""Synchronous JSON-lines RPC with bounded subprocess I/O (Python 3.10+)."""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import threading
import time
from typing import Any

PROTOCOL_VERSION = 1
MAX_RESPONSE_BYTES = 16 * 1024 * 1024
STDERR_TAIL_BYTES = 64 * 1024
MAX_SAFE_INTEGER = 2**53 - 1


class BridgeError(RuntimeError):
    """The bridge could not complete an operation."""


class BridgeConfigurationError(BridgeError):
    """Missing checkout, Node, or an unsupported configuration."""


class BridgeProtocolError(BridgeError):
    """The worker did not follow the agreed wire protocol."""


class BridgeRemoteError(BridgeError):
    """The worker reported an error; its session must not be reused."""


class BridgeTimeoutError(BridgeError):
    """The operation exceeded its wall-clock deadline."""


class BridgeProcessError(BridgeError):
    """The worker stopped or its pipes failed."""


def _validate_json(value: Any) -> None:
    """Avoid silent dict-key conversion and integer rounding on the JS side."""
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int):
        if abs(value) > MAX_SAFE_INTEGER:
            raise ValueError("Protocol v1 requires integers within JavaScript's safe-integer range")
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Protocol v1 requires finite JSON numbers")
    elif isinstance(value, list):
        for item in value:
            _validate_json(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("Protocol v1 requires string object keys")
            _validate_json(item)
    else:
        raise ValueError(f"Not a JSON value: {type(value).__name__}")


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON number: {value}")


class OneClickEngineBridge:
    """One engine session in one persistent Node worker.

    Construction starts the worker and opens the protocol, but does not dispatch
    initialize. Use as a context manager. A session is not shareable across
    processes or concurrent callers. Engine errors are fatal; events are never retried.
    """

    def __init__(self, engine_repo=None, node_executable="node", config=None, timeout_s=10):
        if not isinstance(timeout_s, (int, float)) or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be positive and finite")
        if config is not None and not isinstance(config, dict):
            raise ValueError("config must be a JSON object")
        _validate_json(config)
        self.timeout_s = float(timeout_s)
        self._owner_pid = os.getpid()
        self._proc = None
        self._state = "opening"
        self._id = 0
        self._metadata = {}
        self._lock = threading.Lock()
        self._stderr_lock = threading.Lock()
        self._stderr_tail = bytearray()
        self._requests = queue.Queue()
        self._threads = []
        deadline = time.monotonic() + self.timeout_s
        default_repo = Path(__file__).resolve().parents[2] / "Nomon-One-Click"
        self.engine_repo = Path(engine_repo).expanduser().resolve() if engine_repo is not None else default_repo
        worker = self.engine_repo / "js" / "oneclick" / "engine" / "worker.mjs"
        if not worker.is_file():
            raise BridgeConfigurationError(
                f"QuickClick worker not found: {worker}. Supply engine_repo pointing to the shared-engine checkout."
            )
        node = shutil.which(os.fspath(node_executable))
        if not node:
            raise BridgeConfigurationError(
                f"Node executable not found: {node_executable!r}. Install Node 22+ or supply node_executable."
            )
        try:
            version = subprocess.run(
                [node, "--version"], capture_output=True, timeout=self._remaining(deadline), check=False,
            )
            match = re.fullmatch(rb"v(\d+)\.\d+\.\d+[^\s]*\s*", version.stdout)
            if version.returncode or not match or int(match[1]) < 22:
                raise BridgeConfigurationError("Node 22 or newer is required; node --version returned "
                                               + repr(version.stdout.decode("utf-8", "replace").strip()))
            self._proc = subprocess.Popen(
                [node, str(worker)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, cwd=str(self.engine_repo), shell=False,
            )
            self._threads = [
                threading.Thread(target=self._io_loop, name="quickclick-rpc", daemon=True),
                threading.Thread(target=self._drain_stderr, name="quickclick-stderr", daemon=True),
            ]
            for thread in self._threads:
                thread.start()
            self._state = "open"
            result = self._request("open", {"protocolVersion": PROTOCOL_VERSION, "config": config or {}}, deadline)
            metadata = result.get("metadata") if isinstance(result, dict) else None
            if (not isinstance(metadata, dict) or type(metadata.get("protocolVersion")) is not int
                    or metadata["protocolVersion"] != PROTOCOL_VERSION
                    or not isinstance(metadata.get("nodeVersion"), str)
                    or not isinstance(metadata.get("enginePath"), str)
                    or not isinstance(metadata.get("sourceSha256"), str)
                    or not re.fullmatch(r"[0-9a-f]{64}", metadata["sourceSha256"])
                    or not isinstance(metadata.get("sourceFiles"), list)
                    or not all(isinstance(name, str) for name in metadata["sourceFiles"])):
                raise BridgeProtocolError("Incompatible protocol version or invalid engine metadata")
            self._metadata = metadata
        except subprocess.TimeoutExpired as exc:
            self._state = "failed"
            self._stop()
            raise BridgeTimeoutError("Timed out checking Node version") from exc
        except OSError as exc:
            self._state = "failed"
            self._stop()
            raise BridgeConfigurationError(f"Could not start Node: {exc}") from exc
        except BaseException:
            self._state = "failed"
            self._stop()
            raise

    @staticmethod
    def _remaining(deadline):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise BridgeTimeoutError("QuickClick request deadline exceeded")
        return remaining

    def _io_loop(self):
        # The caller never performs pipe writes or reads: even a full stdin pipe
        # or a response without a newline is covered by the caller's deadline.
        while True:
            task = self._requests.get()
            if task is None:
                return
            payload, reply = task
            try:
                self._proc.stdin.write(payload)
                self._proc.stdin.flush()
                line = self._proc.stdout.readline(MAX_RESPONSE_BYTES + 1)
                if not line:
                    raise BridgeProcessError("Node worker exited before sending a response")
                if len(line) > MAX_RESPONSE_BYTES or not line.endswith(b"\n"):
                    raise BridgeProtocolError("Worker response is oversized or missing its newline")
                reply.put((line, None))
            except (OSError, ValueError, BridgeError) as exc:
                reply.put((None, exc))
                return

    def _drain_stderr(self):
        try:
            while True:
                chunk = self._proc.stderr.read1(4096)
                if not chunk:
                    return
                with self._stderr_lock:
                    self._stderr_tail.extend(chunk)
                    del self._stderr_tail[:-STDERR_TAIL_BYTES]
        except (OSError, ValueError):
            return

    @property
    def metadata(self):
        return copy.deepcopy(self._metadata)

    @property
    def pid(self):
        """Worker PID for diagnostics; None if the worker was never started."""
        return self._proc.pid if self._proc is not None else None

    def _diagnostics(self):
        with self._stderr_lock:
            tail = self._stderr_tail.decode("utf-8", "replace").strip()
        return f"\nWorker stderr (tail):\n{tail}" if tail else ""

    def _request(self, op, fields=None, deadline=None):
        if os.getpid() != self._owner_pid:
            raise BridgeProcessError("Create a new bridge inside each child process; inherited bridges cannot be used")
        if self._state != "open":
            raise BridgeProcessError(f"Bridge is {self._state}; create a new session")
        if not self._lock.acquire(blocking=False):
            raise BridgeProcessError("Concurrent requests on one bridge are not supported")
        try:
            deadline = deadline if deadline is not None else time.monotonic() + self.timeout_s
            self._id += 1
            request = {"id": self._id, "op": op, **(fields or {})}
            # Local validation happens before sending and leaves the session usable.
            _validate_json(request)
            payload = (json.dumps(request, allow_nan=False, ensure_ascii=False) + "\n").encode("utf-8")
            reply = queue.Queue(maxsize=1)
            try:
                self._remaining(deadline)
                self._requests.put((payload, reply))
                line, error = reply.get(timeout=self._remaining(deadline))
                if error:
                    if isinstance(error, BridgeError):
                        raise error
                    raise BridgeProcessError(f"Worker pipe failed: {error}")
                try:
                    response = json.loads(line.decode("utf-8"), parse_constant=_reject_constant)
                    _validate_json(response)
                except (UnicodeError, ValueError, RecursionError) as exc:
                    raise BridgeProtocolError(f"Invalid JSON from worker: {exc}") from exc
                if (not isinstance(response, dict) or type(response.get("id")) is not int
                        or response["id"] != self._id or type(response.get("ok")) is not bool):
                    raise BridgeProtocolError("Invalid response envelope or mismatched request ID")
                if not response["ok"]:
                    error = response.get("error")
                    if not isinstance(error, dict) or not all(isinstance(error.get(k), str) for k in ("code", "message")):
                        raise BridgeProtocolError("Invalid worker error envelope")
                    raise BridgeRemoteError(f"{error['code']}: {error['message']}")
                if "result" not in response:
                    raise BridgeProtocolError("Response has no result")
                result = response["result"]
                if op == "dispatch" and (not isinstance(result, dict)
                        or not isinstance(result.get("snapshot"), dict) or not isinstance(result.get("effects"), list)):
                    raise BridgeProtocolError("Invalid dispatch result")
                if op == "snapshot" and not isinstance(result, dict):
                    raise BridgeProtocolError("Invalid snapshot result")
                if op == "close" and result != {"closed": True}:
                    raise BridgeProtocolError("Invalid close acknowledgement")
                return result
            except (queue.Empty, BridgeTimeoutError) as exc:
                self._state = "failed"
                self._stop()
                raise BridgeTimeoutError(f"Timed out during {op}; the session was terminated" + self._diagnostics()) from exc
            except BridgeError as exc:
                self._state = "failed"
                self._stop()
                raise type(exc)(str(exc) + self._diagnostics()) from exc
            except BaseException:
                self._state = "failed"
                self._stop()
                raise
        finally:
            self._lock.release()

    def dispatch(self, event):
        if not isinstance(event, dict):
            raise ValueError("event must be a JSON object")
        return self._request("dispatch", {"event": event})

    def get_snapshot(self):
        return self._request("snapshot")

    def _stop(self):
        if self._proc is None:
            return
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
        else:
            self._proc.wait()
        self._requests.put(None)
        for thread in self._threads:
            if thread is not threading.current_thread():
                thread.join(timeout=1)
        for stream in (self._proc.stdin, self._proc.stdout, self._proc.stderr):
            try:
                stream.close()
            except (OSError, ValueError):
                pass

    def close(self):
        if os.getpid() != self._owner_pid:
            raise BridgeProcessError("An inherited bridge must not close its parent's worker")
        if self._state == "closed":
            return
        deadline = time.monotonic() + 2.0
        try:
            if self._state == "open":
                self._request("close", deadline=deadline)
                try:
                    self._proc.wait(timeout=self._remaining(deadline))
                except (subprocess.TimeoutExpired, BridgeTimeoutError):
                    pass
        except BridgeError:
            # Cleanup is idempotent and must not mask an exception from the body.
            pass
        finally:
            self._state = "closed"
            self._stop()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
