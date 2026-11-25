"""Persistent WebSocket predictions; no TextSlinger/model imports in this client."""
from __future__ import annotations

import asyncio
import concurrent.futures
import copy
import importlib.metadata
import json
import math
import ssl
import threading
import time
from urllib.parse import urlsplit
import uuid

from .websocket_codec import prediction_request, prediction_response, strict_loads


class WebSocketBackendError(RuntimeError):
    pass


class WebSocketPredictionBackend:
    """One context-managed connection per phrase. A failed instance is not retried."""

    def __init__(self, server_url="http://127.0.0.1:8766", model="default", vocabulary_profile=None,
                 timeout_s=15, connect_timeout_s=10):
        parsed = urlsplit(server_url)
        if (parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password
                or parsed.path not in ("", "/") or parsed.query or parsed.fragment):
            raise ValueError("Prediction server must be an HTTP(S) origin without credentials or a path")
        if parsed.port is not None and not 1 <= parsed.port <= 65535:
            raise ValueError("Prediction server port must be 1..65535")
        self.port = parsed.port or (443 if parsed.scheme == "https" else 80)
        if model is None:
            raise ValueError("Model name is required")
        for label, value in (("model", model), ("vocabulary profile", vocabulary_profile)):
            if value is not None and (not isinstance(value, str) or not 1 <= len(value) <= 128):
                raise ValueError(f"Invalid {label}")
        for value in (timeout_s, connect_timeout_s):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("Prediction timeouts must be positive and finite")
        self.server_url = f"{parsed.scheme}://{parsed.netloc}"
        self.ws_url = f"{'wss' if parsed.scheme == 'https' else 'ws'}://{parsed.netloc}/api/v1/ws"
        self.hostname, self.secure = parsed.hostname, parsed.scheme == "https"
        self.model, self.vocabulary_profile = model, vocabulary_profile
        self.timeout_s, self.connect_timeout_s = float(timeout_s), float(connect_timeout_s)
        self.session_id, self.revision = str(uuid.uuid4()), 0
        self.state = "idle"
        self.loop = self.thread = self.connection = None
        self._last = None
        self._lock = threading.Lock()
        self.health = None
        self.health_error = None
        self.failure = None
        self.cleanup_error = None
        self.client_version = None
        self.dropped_negative_infinity = 0

    def __enter__(self):
        if self.state != "idle":
            raise WebSocketBackendError("Use a new WebSocket backend instance for each phrase")
        self.state = "opening"
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_loop, name=f"oneclick-ws-{self.session_id[:8]}", daemon=True)
        self.thread.start()
        try:
            self._call(self._open(), self.connect_timeout_s)
            self.state = "connected"
            return self
        except Exception as exc:
            self.failure = dict(type=type(exc).__name__, message=str(exc))
            self.close()
            raise WebSocketBackendError(f"Cannot open TextSlinger WebSocket at {self.server_url}: {exc}") from exc

    def __exit__(self, *_):
        self.close()

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_forever()
        finally:
            tasks = asyncio.all_tasks(self.loop)
            for task in tasks:
                task.cancel()
            if tasks:
                self.loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
            self.loop.close()

    def _call(self, coroutine, timeout):
        future = asyncio.run_coroutine_threadsafe(asyncio.wait_for(coroutine, timeout), self.loop)
        try:
            return future.result(timeout=timeout)
        except (concurrent.futures.TimeoutError, asyncio.TimeoutError) as exc:
            future.cancel()
            raise WebSocketBackendError(f"TextSlinger operation timed out after {timeout:g}s") from exc

    async def _open(self):
        try:
            from websockets.asyncio.client import connect
        except ImportError as exc:
            raise WebSocketBackendError("Install optional client dependencies with: python -m pip install -r requirements-websocket.txt") from exc
        self.client_version = importlib.metadata.version("websockets")
        major = int(self.client_version.split(".")[0])
        if not 14 <= major < 17:
            raise WebSocketBackendError("WebSocket client requires websockets>=14,<17")
        self.connection = await connect(self.ws_url, subprotocols=["textslinger.v1"],
                                        open_timeout=self.connect_timeout_s, close_timeout=1, max_size=16 * 1024 * 1024)
        ready = strict_loads(await self.connection.recv())
        if not isinstance(ready, dict) or ready.get("type") != "ready" or type(ready.get("version")) is not int or ready["version"] != 1:
            raise WebSocketBackendError("Invalid TextSlinger ready handshake or protocol version")
        # Optional provenance only. Failure here must not replace a working
        # prediction connection. Connection: close delimits the HTTP body.
        try:
            self.health = await asyncio.wait_for(self._health(), min(3, self.connect_timeout_s / 4))
        except Exception as exc:
            self.health_error = f"{type(exc).__name__}: {exc}"

    async def _health(self):
        writer = None
        try:
            reader, writer = await asyncio.open_connection(self.hostname, self.port,
                ssl=ssl.create_default_context() if self.secure else None)
            writer.write(f"GET /health HTTP/1.1\r\nHost: {urlsplit(self.server_url).netloc}\r\nConnection: close\r\n\r\n".encode("ascii"))
            await writer.drain()
            header = await reader.readuntil(b"\r\n\r\n")
            if header.split(b" ", 2)[1] != b"200":
                raise ValueError("Health endpoint did not return HTTP 200")
            body = bytearray()
            while True:
                chunk = await reader.read(65536)
                if not chunk:
                    break
                body.extend(chunk)
                if len(body) > 1024 * 1024:
                    raise ValueError("Health response exceeds 1 MiB")
            health = strict_loads(body)
            if not isinstance(health, dict):
                raise ValueError("Health response must be an object")
            return health
        finally:
            if writer is not None:
                writer.close()
                writer.transport.abort()

    async def _exchange(self, request):
        await self.connection.send(json.dumps(request, allow_nan=False, ensure_ascii=False))
        message = await self.connection.recv()
        if not isinstance(message, str):
            raise ValueError("TextSlinger must send JSON text frames")
        return message

    def predict(self, effect):
        if self.state != "connected":
            raise WebSocketBackendError("WebSocket backend is not connected or is no longer usable")
        if not self._lock.acquire(blocking=False):
            raise WebSocketBackendError("Only one outstanding prediction is supported")
        started = time.perf_counter()
        self._last = dict(status="pending")
        try:
            self.revision += 1
            request = prediction_request(effect, session_id=self.session_id, request_id=str(uuid.uuid4()),
                revision=self.revision, model=self.model, vocabulary_profile=self.vocabulary_profile)
            self._last["wire_request"] = copy.deepcopy(request)
            raw = self._call(self._exchange(request), self.timeout_s)
            self._last["raw_response_preview"] = raw[:4096]
            response = strict_loads(raw)
            self._last["wire_response"] = response
            result = prediction_response(effect, request, response)
            if effect["type"] == "predict-words":
                self.dropped_negative_infinity += sum(
                    item["probability"]["log_retained_joint_mass"] == "-inf"
                    for item in response["data"]["words"]["predictions"])
            self._last.pop("raw_response_preview", None)
            self._last.update(status="ok", client_elapsed_s=time.perf_counter() - started,
                              server_timings=copy.deepcopy(response.get("timings")))
            return result["data"]
        except Exception as exc:
            self.failure = dict(type=type(exc).__name__, message=str(exc))
            # Never persist invalid non-finite data in an otherwise strict trace.
            try:
                json.dumps(self._last.get("wire_response"), allow_nan=False)
            except (ValueError, TypeError):
                self._last.pop("wire_response", None)
            self._last.update(status="error", error=self.failure, client_elapsed_s=time.perf_counter() - started)
            self.close()
            raise WebSocketBackendError(f"TextSlinger prediction failed: {exc}") from exc
        finally:
            self._lock.release()

    def last_exchange(self):
        return copy.deepcopy(self._last)

    def metadata(self):
        return copy.deepcopy(dict(backend="textslinger-websocket", server_url=self.server_url, model=self.model,
            vocabulary_profile=self.vocabulary_profile, protocol_version=1, client_version=self.client_version,
            request_profile="quickclick-browser-v1", nbest=1000, timeout_s=self.timeout_s,
            server_health=self.health, health_error=self.health_error, failure=self.failure,
            cleanup_error=self.cleanup_error, dropped_negative_infinity_candidates=self.dropped_negative_infinity,
            candidate_count_scope="phrase", state=self.state))

    async def _disconnect(self):
        if self.connection is not None:
            try:
                await asyncio.wait_for(self.connection.close(), 1)
            finally:
                self.connection.transport.abort()

    def close(self):
        if self.state == "closed":
            return
        self.state = "closed"
        if self.loop is not None and self.thread is not None and self.thread.is_alive():
            try:
                self._call(self._disconnect(), 1.5)
            except Exception as exc:
                self.cleanup_error = str(exc)
                if self.connection is not None:
                    self.loop.call_soon_threadsafe(self.connection.transport.abort)
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join(timeout=2)
            if self.thread.is_alive():
                self.cleanup_error = "WebSocket event-loop thread did not stop"


def add_websocket_arguments(parser):
    parser.add_argument("--prediction-server", help="HTTP(S) origin; defaults to http://127.0.0.1:8766")
    parser.add_argument("--prediction-model", help="Server model name; defaults to default")
    parser.add_argument("--prediction-vocabulary", help="Optional server vocabulary profile, not a local path")
    parser.add_argument("--prediction-timeout-s", type=float, help="Wall-clock request deadline; defaults to 15 seconds")


def backend_from_arguments(args):
    return WebSocketPredictionBackend(args.prediction_server or "http://127.0.0.1:8766",
        model=args.prediction_model or "default", vocabulary_profile=args.prediction_vocabulary,
        timeout_s=args.prediction_timeout_s if args.prediction_timeout_s is not None else 15)
