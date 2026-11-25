# TextSlinger WebSocket predictions

The opt-in `textslinger-websocket` backend uses the same prediction service as
the browser. Python still drives the shared JS engine through Node. The engine,
simulated-user policy, fixture backend, and direct local LM adapter are unchanged.

## Run locally

Requirements: Python 3.10+, Node 22+, the sibling `Nomon-One-Click` checkout,
and a running TextSlinger service. Install the optional transport in the Python
environment running the simulator:

```sh
python -m pip install -r requirements-websocket.txt
```

The client needs no TextSlinger, PyTorch, KenLM, or model files. Those stay in the
separate server process. The existing `textslinger-server` environment can also
run the client and already contains `websockets`. Fixtures and replay need no
optional transport dependency.

The tested upstream pair is [TextSlinger Server 0.4.0](https://github.com/kdv123/textslinger-server/tree/634c022)
and [TextSlinger 0.5.0](https://github.com/kdv123/textslinger/tree/3e36d70).
Use a separate Python 3.12+ server environment and follow the
[server installation guide](https://github.com/kdv123/textslinger-server/blob/634c022/docs/installation.md)
for the model extras and build instructions. The server starts independently of
the simulator.

- **CPU n-gram.** Install the `ngram` extra and build KenLM with 12-gram
  support. From the server checkout, run `python assets/download.py --models tiny`
  and then `python server.py --config configs/ngram.json`. The advertised model
  is `ngram`.
- **OPT-125M subword.** Install the `llm` extra and PyTorch. On Apple Silicon,
  use the `subword` worker from `configs/ngram-subword-mps.json`. Copy that config
  to a local file in the same directory and remove the `ngram` worker for a
  neural-only setup. On NVIDIA hardware, the upstream CUDA example is
  `configs/subword-cuda.json`. Start the chosen config with
  `python server.py --config PATH`. The advertised model is `subword`.
  The worker loads `figmtu/opt-125m-aac` and may download its weights on first startup.

The upstream examples use port `8765`. Use the port from your config, or assign
distinct ports to run both models at once. Local validation used ports `8767`
for n-gram and `8768` for OPT on MPS.

In another terminal, activate the client environment and run from `Nomon-Simulation`:

```sh
python -m OneClick_Simulation.bridge_pilot \
  --backend textslinger-websocket \
  --prediction-server http://127.0.0.1:8765 \
  --prediction-model ngram \
  --phrase "hello world" \
  --output-dir bridge_pilot_output/websocket
```

The CLI default remains `fixture`; `textslinger-ngram` still calls the local
library directly. WebSocket-specific options are:

| Option | Default | Meaning |
| --- | --- | --- |
| `--prediction-server` | `http://127.0.0.1:8766` | HTTP(S) origin, mapped to WS(S) `/api/v1/ws` |
| `--prediction-model` | `default` | Advertised model name, not worker ID |
| `--prediction-vocabulary` | omitted | Server vocabulary-profile name |
| `--prediction-timeout-s` | `15` | Wall-clock request deadline in seconds |

Do not supply local model/vocabulary paths in WebSocket mode. Its browser-matching
candidate count is fixed at 1000. Existing click, speed, fixed-character-matrix,
and simulated-latency options remain available. Nonzero exit status indicates
failure; result/trace files retain partial diagnostics. Invalid targets are
rejected before allocating either session.

## Python interface and lifecycle

```python
from OneClick_Simulation.bridge_simulated_user import BridgeSimulatedUser, ZeroNoiseClickSource
from OneClick_Simulation.websocket_predictions import WebSocketPredictionBackend

backend = WebSocketPredictionBackend("http://127.0.0.1:8766")
result = BridgeSimulatedUser().run_phrase("hello world", ZeroNoiseClickSource(), backend)
```

The phrase runner enters/exits context-managed backends and owns cleanup. Supply
a new WebSocket backend instance for each phrase. Each has one persistent
connection/session, unique IDs, increasing revisions, and one outstanding request.
Its background event loop maintains the connection between synchronous calls.
Standalone callers can use `with WebSocketPredictionBackend(...) as backend:`.

Setup/ready handshake times out after ten seconds. Requests carry a ten-second
server deadline and default to a fifteen-second client deadline. Asynchronous
send and receive are bounded even though the public `predict(effect)` is
synchronous. A timeout, disconnect, malformed JSON, mismatched identity, or
non-`ok` response fails the phrase and closes the connection/thread. Duplicate
JSON keys and non-finite numeric JSON are rejected. There is no cache, retry,
reconnection, or fallback. WSS uses normal certificate verification. Authentication
is deferred. CPU n-gram and OPT-125M on MPS both passed the word, sentence, and
deliberate-error/Undo scenarios through a real server.

## Browser parity and simulated time

Conversion matches the browser's `prediction-codec.mjs`. Complete observation
sequences are raw 27-character likelihoods; positive densities are preserved.
Insertion/deletion/outlier probabilities are zero, completion prior is 0.5,
prediction is lowercase, spelling uses the engine alphabet, and endings are spaces.
Responses use combined `log_retained_joint_mass`, stable ranking, case-insensitive
deduplication, length-based pools, and effect-specific limits. Character `"-inf"`
uses the browser's floor; impossible word scores are dropped.

The older direct adapter performs additional character normalization and can use
different probability bounds and vocabulary settings. It is not the parity oracle
for this backend. Cross-language tests use independently specified fixtures and
the browser codec. Comparable live predictions also require the same server
model, vocabulary, and search settings.

The simulated user retains `wait_for_predictions`: the next action is chosen only
after pending responses are delivered. The browser's interactive policy, which
permits further presses while replies are pending, is not introduced here.

`--character-prediction-latency-s` and `--word-prediction-latency-s` remain fixed
simulated delays. Network, computation, setup, and cleanup wall time never advance
simulated time. Clocks advance through the existing explicit frame scheduler.
Startup requests share simulated time zero even though they are serviced
sequentially in wall time.

## Diagnostics, replay, and validation

Backend metadata records URL, model/profile, client/protocol/request-profile
versions, drop count, and available `/health` data. Health lookup is bounded and
optional; lookup failures are recorded without replacing a working connection.
Reported server provenance is not independent verification of model-file hashes.

Each WebSocket prediction request additionally records `transport`: wire request,
wire response, measured `client_elapsed_s`, server timings, and status/error.
Parsing failures retain a bounded raw-message preview. These diagnostics are
separate from simulated timing. The backend receives no intended targets or
actions, only engine context/evidence. Saved traces include typed text and
observations. Credentials/query tokens in server URLs are rejected.

```sh
# Both commands work with the prediction service stopped.
python -m OneClick_Simulation.bridge_pilot \
  --replay bridge_pilot_output/websocket/trace.json
python -m OneClick_Simulation.bridge_validate \
  --audit bridge_pilot_output/websocket

# Six fixture scenarios plus actual-server word, sentence, and correction cases.
python -m OneClick_Simulation.bridge_validate \
  --include-websocket \
  --prediction-server http://127.0.0.1:8765 \
  --prediction-model ngram \
  --output-dir bridge_validation_output/websocket

# Focused model-free codec, actual-socket fixture, timing, failure, and replay tests.
python -m unittest OneClick_Simulation.test_websocket_predictions -v
```

Full auditing also checks wire requests against engine effects, decoded replies
against dispatched events, and valid measured timing fields. Existing version-1
traces remain supported. Socket tests require `websockets` and local socket access;
fixture execution and replay do not. The real-model correction scenario requires
an actual wrong commit, Undo, and eventual completion, without fixed model-specific
press counts. Artifacts are reloaded before audit. These checks establish
integration/replay consistency, not real-user performance or prediction quality.

The simulator carries its own prediction fixtures. The browser codec parity
test also runs when the optional browser WebSocket checkout is available
(`QUICKCLICK_BROWSER_REPO` overrides its path).
