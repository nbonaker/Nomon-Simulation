# Python–Node QuickClick bridge

This package lets Python drive the same JavaScript engine used by the OneClick browser interface. One client owns one persistent Node worker and one engine session. It uses only the Python standard library: importing it does not load NumPy, TextSlinger, or the existing Python keyboard.

The current simulator still uses its existing implementation. This bridge is a separately tested foundation for a later migration; it does not run predictions, choose clicks, schedule animation, or translate old simulator settings.

## Requirements and quick start

- Python 3.10+ and Node 22+ on macOS or Linux.
- A local `Nomon-One-Click` checkout containing `js/oneclick/engine/worker.mjs`.
- By default, that checkout is a sibling of `Nomon-Simulation`. Paths are resolved relative to this package, not your terminal's current directory. Nothing is downloaded or updated automatically.

From the simulator repository:

```sh
.venv/bin/python -m OneClick_Bridge.demo
.venv/bin/python -m unittest OneClick_Bridge.test_bridge -v
```

Or use any Python 3.10+ interpreter. No additional pip or npm install is needed for the bridge. To choose another checkout or Node binary:

```sh
python -m OneClick_Bridge.demo --engine-repo /path/to/Nomon-One-Click --node-executable /path/to/node
```

The demo supplies fixed character and word responses, aims a Space press at `h`, observes 27 letter likelihoods, commits `hello`, and selects Undo. It uses explicit simulated frame/click timestamps and never sleeps to simulate typing or contacts a prediction service. Fixture response latency is zero; this is a transport demonstration, not a realistic user study.

## Python interface

```python
from OneClick_Bridge import OneClickEngineBridge

with OneClickEngineBridge(
    engine_repo=None,              # default: sibling JS checkout
    node_executable="node",        # PATH lookup or explicit executable
    config={"rotateIndex": 5, "useClickOffset": False},
    timeout_s=10,
) as engine:
    print(engine.metadata)
    result = engine.dispatch({"type": "initialize", "time": 0.0})
    # result == {"snapshot": {...}, "effects": [...]}
    for effect in result["effects"]:
        # Host code obtains a prediction and dispatches the matching response.
        print(effect["type"])
    state = engine.get_snapshot()
```

Construction opens the protocol and creates an engine; it does **not** dispatch `initialize`. All time-dependent events must contain an explicit `time` in seconds. Pass the shared engine's event/config dictionaries unchanged; use its API documentation in the JavaScript checkout for their meanings. `get_snapshot()` does not advance state. Returned dictionaries/lists and `metadata` can be changed locally without changing the worker.

`dispatch` forwards prediction-request effects to the caller as data. A future Python prediction adapter will answer them with `character-response` and `prediction-response` events. Frames must be sent explicitly for each group (`letters` and `words`). Real time spent waiting for IPC is not automatically added to simulated time.

The process stays alive until `close()` or context-manager exit. `close()` is idempotent. Create clients **inside** each multiprocessing task, and do not pass them to child processes or share a client among concurrent callers. Use a new client for a new session; there is no automatic reset, retry, checkpoint restoration, batching, or worker pool.

## Protocol v1

Transport is UTF-8 newline-delimited JSON on stdin/stdout. stdout contains only responses, and stderr is reserved for diagnostics. One request is outstanding at a time, IDs are strictly increasing positive safe integers, and each response echoes its request ID.

Open a session:

```json
{"id":1,"op":"open","protocolVersion":1,"config":{"rotateIndex":5,"useClickOffset":false}}
```

Success has the envelope `{"id":1,"ok":true,"result":{"metadata":{...}}}`.

| Operation | Request fields beyond `id`, `op` | Result |
| --- | --- | --- |
| `open` | `protocolVersion: 1`, optional `config` object | `{metadata: {...}}` |
| `dispatch` | `event` object | Unchanged engine `{snapshot, effects}` |
| `snapshot` | none | Current engine snapshot |
| `close` | none | `{closed: true}`, then worker exits |

Errors use `{"id":2,"ok":false,"error":{"code":"ENGINE_ERROR","message":"..."}}`. Invalid protocol messages use `PROTOCOL_ERROR`; when no valid ID can be recovered the ID is null. An error ends the worker session. stdin EOF also ends it.

Only strict JSON values are accepted. Non-finite numbers (`NaN`, either infinity) are rejected, including overflow such as `1e999`. Python also rejects non-string object keys and integers outside JavaScript's safe range, ±(2^53−1), rather than silently changing them. If the engine produces a non-finite result, the worker reports an error and exits: it never serializes that value as null. A future model adapter must explicitly address non-finite scores before sending them; this milestone does not introduce a probability conversion policy.

Python limits each response line to 16 MiB and keeps the last 64 KiB of stderr. These are transport bounds, not model or experiment limits; oversized responses fail explicitly.

## Source identity

`metadata` records `protocolVersion`, `nodeVersion`, the resolved `enginePath`, `sourceFiles`, and `sourceSha256`. The worker hashes the sorted, immediate `.mjs` files in its engine directory, including itself. For each file it hashes UTF-8 `nameByteLength:name:contentByteLength:` followed by the exact file bytes. This identifies uncommitted edits as well as committed versions and requires no Git subprocess.

Keep the checkout stable while a session starts/runs. The fingerprint records sources at startup; it does not hot-reload the module or enforce a pinned revision. Future engine dependencies outside that directory must be included in the fingerprint scheme. The engine remains in its own repository; no engine source is vendored here.

## Failures and cleanup

- `BridgeConfigurationError`: missing checkout, missing/unusable Node, or unsupported Node version.
- `BridgeProtocolError`: malformed messages, wrong response IDs, incompatible handshake, oversized output, or invalid result shape.
- `BridgeRemoteError`: error returned by the worker, including an invalid engine event.
- `BridgeTimeoutError`: startup/request exceeded the configured wall-clock deadline.
- `BridgeProcessError`: worker exit/pipe failure, use after failure/close, concurrent calls, or inherited-process misuse.

All inherit from `BridgeError`. Local invalid input raises `ValueError` before sending anything and leaves the session usable. Remote/protocol/timeout/process failures terminate and reap the worker; the failed session cannot be reused. Never automatically retry a failed click: its state change may have happened before communication failed.

A background I/O thread performs both writes and reads, while another thread drains stderr. The main caller enforces request deadlines even if stdin fills or stdout never produces a full line. Failure diagnostics include the bounded stderr tail. Shutdown allows up to two seconds for a close acknowledgement and normal exit, then terminates the worker; an unresponsive termination escalates to a kill after another half second. Process cleanup adds a small bounded overhead to the request deadline. Context-manager cleanup also runs when its body raises an exception.

## Verification

The bridge tests replay all four original baseline traces (164 checkpoints), using fixtures from the selected sibling JS checkout. They compare discrete state exactly and floating-point values with absolute tolerance `1e-10`. A separate direct-engine Node run supplies full-result comparisons beyond the historical fixture projection.

Other tests cover independent sessions, Unicode, source fingerprints and alternate paths, detached state, configuration forwarding, read-only snapshots, missing dependencies, protocol errors, non-finite data, startup/request stalls, blocked writes, crashes, stderr floods, and shutdown/reaping. Fake workers are confined to temporary test directories. Integration tests fail with actionable dependency errors rather than silently skipping when the real Node/engine dependency is missing.

Run the JavaScript side from `Nomon-One-Click`:

```sh
npm test
```

No browser test rerun is required for this addition: the browser does not import the worker, and its algorithm and adapter are unchanged. Full simulator migration, real prediction backends, latency experiments, and performance tuning remain follow-up work.
