# Simulated user using the shared JavaScript engine

`BridgeSimulatedUser` is an opt-in pilot that generates user actions for the browser's shared engine through `OneClick_Bridge`. It never imports or calls the duplicated Python keyboard algorithm. The existing `SimulatedUser`, study runners, experiment settings, and result formats remain unchanged.

The pilot supplies intentions and timestamps. The engine owns observations, clock placement, selection, committed text, Undo history, and delay learning. Python reads snapshots to decide what to attempt next; it cannot replace a click distribution, confirm a commit, or restore a hidden keyboard snapshot.

## Run and replay

Requirements are Python 3.10+, Node 22+, and the sibling `Nomon-One-Click` checkout containing the shared engine and worker. The fixture mode and trace replay use only the Python standard library and the bridge. The optional n-gram mode loads the existing local TextSlinger adapter and model once; it makes no network requests or downloads.

From `Nomon-Simulation`:

```sh
.venv/bin/python -m OneClick_Simulation.bridge_pilot --output-dir /tmp/quickclick-pilot
.venv/bin/python -m OneClick_Simulation.bridge_pilot --replay /tmp/quickclick-pilot/trace.json
```

The bundled fixture types `hello world` using two Space and two Enter presses, completing in 9.06 simulated seconds with the default period. This number describes a deterministic integration example, not measured user/model performance.

Optional arguments:

- `--engine-repo PATH` and `--node-executable PATH`: select the local engine checkout and Node binary.
- `--phrase TEXT` and `--fixture PATH`: supply another target and an independently specified prediction fixture.
- `--backend fixture|textslinger-ngram`: choose fixed predictions (default) or the local model.
- `--model PATH`, `--vocabulary PATH`, `--recognizer-nbest N`: explicit local n-gram inputs; both paths are required in n-gram mode.
- `--character-transition-source backend|fixed`: choose the startup matrix source (default `backend`). With `fixed`, supply `--character-transitions PATH`.
- `--character-prediction-latency-s SECONDS`, `--word-prediction-latency-s SECONDS`: finite, nonnegative simulated response delays; both default to zero.
- `--rotate-index N`: use one of the engine's existing speed indices (0–20; default 5) for the entire phrase.
- `--clicks PATH`: consume recorded CSV timing samples in file order instead of using zero noise.
- `--output-dir PATH`: write `result.json` and `trace.json` (default `bridge_pilot_output`). Files in that output directory are replaced by a later run using the same directory.

CLI exit status is zero for a completed phrase and nonzero for a failed one. Failure results retain the successful event prefix and diagnostics. Invalid target text is rejected before opening a worker.

## Python interface

```python
import json
from pathlib import Path
from OneClick_Simulation.bridge_simulated_user import (
    BridgeSimulatedUser, FixedPredictionBackend, ZeroNoiseClickSource,
)

fixture = json.loads(Path("OneClick_Simulation/bridge_fixtures/hello_world.json").read_text())
user = BridgeSimulatedUser(rotate_index=5)
result = user.run_phrase(
    target_phrase="hello world",
    click_source=ZeroNoiseClickSource(),
    prediction_backend=FixedPredictionBackend.from_dict(fixture),
)
assert result["completed"]
```

Each call owns and closes a new bridge session. Targets are lowercased, whitespace is collapsed to single spaces, and only a–z plus apostrophe are accepted inside words. The engine's committed text retains its normal trailing space. Fixtures should supply lowercase candidate text to match these normalized targets. A `BridgeSimulatedUser` instance may be reused sequentially, but not concurrently.

For scripted errors, use `SequenceClickSource([ClickSample(offset_s=...), ...])`. It returns `None` at exhaustion; it never reshuffles or loops. `ZeroNoiseClickSource` supplies repeated zero-offset, zero-dead-time samples. Both remain subject to action budgets.

Recorded CSV rows use `Click Time Relative (s)`, optional `Dead Time (s)`, and optional `Clock Period (s)`. Blank/missing dead time means zero. Explicit non-finite timing values and negative dead times are invalid. Source clock periods are recorded as metadata, not applied to the engine. This pilot uses fixed engine periods; it does not support calibration injection, arbitrary periods, or separate Space/Enter periods.

## Fixed predictions

`PredictionBackend.predict(effect)` returns raw response data for one engine effect. The backend receives committed context and click evidence, never the target phrase or intended clock.

The fixed backend uses a shared character response or 27 responses keyed by previous character, and word-response fixtures keyed by `(committed context, observation count)`. Word responses intentionally ignore the likelihood magnitudes: they test integration, not recognition quality. Fixtures may include distractors, wrong words, or no target. There is no target-aware fallback when a fixture is missing.

JSON fixture shape:

```json
{
  "characters": {
    "results": [{"token": "a", "logProb": -3.3}]
  },
  "words": [
    {"left": "", "observation_count": 0, "data": {"prefix": [], "best": []}},
    {"left": "", "observation_count": 1, "data": {
      "prefix": [{"text": "hello", "logprob": -0.1}], "best": []
    }},
    {"left": "hello ", "observation_count": 0, "data": {"prefix": [], "best": []}}
  ]
}
```

The bundled fixture includes all 27 characters. The shortened example above only illustrates the format: executable character responses must contain each of `abcdefghijklmnopqrstuvwxyz'` exactly once with a finite score. The engine still applies its existing flooring and normalization; the pilot validates input completeness before dispatch. Missing context/count entries raise a prediction failure, including requests immediately after the final word or a wrong commit. Supply all contexts a scripted error scenario may reach. Duplicate fixture keys are rejected.

Response data is computed locally, then queued for delivery at `request_time + configured_latency`. Equal arrival times preserve effect order. Missing fixture entries fail even after the final commit; no target-aware fallback is supplied.

## Local TextSlinger n-gram predictions

Use the existing simulator virtual environment with TextSlinger and KenLM installed, and an explicitly supplied local model and vocabulary. The bundled tiny model is order 12; a KenLM build supporting `KENLM_MAX_ORDER=12` is required. The loader reports missing files, dependencies, or an incompatible KenLM build rather than downloading anything. Subword and HTTP adapters are not included in this milestone.

From `Nomon-Simulation`:

```sh
.venv/bin/python -m OneClick_Simulation.bridge_pilot \
  --backend textslinger-ngram \
  --model Nomon_Text/resources/lm_char_tiny.kenlm \
  --vocabulary Nomon_Text/resources/vocab_lower_100k.txt \
  --phrase "hello world" \
  --character-prediction-latency-s 0.2 \
  --word-prediction-latency-s 0.12 \
  --output-dir /tmp/quickclick-ngram

# Replay does not load TextSlinger or require the model files.
.venv/bin/python -m OneClick_Simulation.bridge_pilot \
  --replay /tmp/quickclick-ngram/trace.json
```

Python equivalent:

```python
from OneClick_Simulation.bridge_predictions import TextSlingerNGramBackend

backend = TextSlingerNGramBackend.from_local(
    "Nomon_Text/resources/lm_char_tiny.kenlm",
    "Nomon_Text/resources/vocab_lower_100k.txt",
)
user = BridgeSimulatedUser(
    character_transition_source="backend",
    character_prediction_latency_s=0.2,
    word_prediction_latency_s=0.12,
)
result = user.run_phrase("hello world", ZeroNoiseClickSource(), backend)
```

The backend receives only engine prediction effects. It validates and reorders each observation distribution into the 27-symbol model order, preserving likelihood values, and passes observations plus committed left context to `LanguageModel.get_word_predictions`. Requested prefix/BEST limits and strict score validation are explicit. It returns `{prefix, best}`; the JavaScript engine remains responsible for its normal deduplication, ranking, word-clock placement, selection, and commit behavior. The model never receives the target phrase or intended letter.

Impossible word candidates with `-Infinity` scores are dropped before JSON serialization and counted in backend metadata. NaN and positive Infinity fail the phrase. Character and observation scores must be finite. Strict word-score checking is opt-in in the existing adapter; its legacy callers retain their prior defaults. Backend metadata records model and vocabulary paths and hashes, TextSlinger version/source/commit, search settings, and score policy. Backend drop counts are explicitly scoped to the adapter's lifetime, since one loaded model can be reused across phrases. Request records and `prediction_metadata.dropped_negative_infinity_candidates` also report per-request and per-phrase counts.

## Character transition experiment setting

`character_transition_source="backend"` routes the 27 startup `predict-characters` effects to the selected prediction backend. The TextSlinger backend calls `get_key_probs(effect["left"])` with a **single previous character**, matching the browser's startup requests. It does not build a new matrix from the evolving sentence context.

`character_transition_source="fixed"` requires `fixed_character_responses`: either a shared `{"results": [...]}` response, or a mapping from each of the 27 previous characters to a complete response. In the CLI, `--character-transitions` points to a JSON file containing that object directly. Word predictions still come from the selected backend. This lets an experiment change the word model while holding the character transitions fixed.

Every row is delivered through a normal engine `character-response` event. The engine clamps and normalizes it; Python never overwrites `transition_matrix`. The existing TextSlinger character adapter also performs its own flooring and normalization, so backend mode deliberately retains both stages. Results record the source responses, the actual effective 27×27 matrix from the engine, and its SHA-256. The matrix stays fixed throughout a phrase.

## Prediction latency experiment settings

Character and word latency settings are fixed simulated seconds per request, not measured model runtimes. All startup effects share time zero, so their waits overlap; startup takes the later of the two delays, not 27 times the character delay. Responses arriving before engine readiness do not advance uninitialized clocks. When the engine first becomes ready, the shared frame schedule starts at that event timestamp.

After every press, the pilot drains pending responses before selecting a target clock or sampling the next click. Initialized clocks continue advancing during these waits. At a timestamp shared by scheduled work, the order is letter frame, word frame, queued prediction responses in request order, then a user press. A zero-latency response caused by a press naturally follows that press. Response-induced clock placement and rephasing happen in the engine; the next intended action reads the resulting snapshot.

No sleeps, subprocess duration, or model computation time contribute to simulated time. Model computation is synchronous in wall time; simulated response delivery follows the explicit queue. Zero latency preserves the pre-integration fixed-fixture event trace exactly. Overlapping user presses while predictions are pending and variable/distributed latency remain deferred.

## Scheduling and correction

The scheduler derives the frame interval from engine period/phase-bin count and sends separate letter and word frame events. It uses integer frame indices to avoid cumulative timer drift. At a frame/press tie it sends letter frame, word frame, then the press.

For a target clock it calculates the next noon using phase and last-frame timestamp, applies the sampled offset, moves forward by whole rotations if the press would be before the minimum inter-press interval (1 ms), and adds the recorded whole-rotation component of dead time. It chooses the press timestamp once, then forwards all intervening frames. Sampled error is not canceled by aiming again after each frame. Neither sleeps nor IPC wall time advance simulated time.

Each word attempt begins with a Space press. After each response, the user looks for the target among active word clocks and selects the first match in engine order. If no match appears after all letters, it aims at the literal clock. The Undo command is excluded from word matching, so typing the word `undo` is supported.

After Enter, the user examines actual committed text:

- A valid whole-word target prefix resumes at its next missing word.
- Midword Undo discards observations, so the next attempt retypes them.
- An incorrect suffix triggers real Undo presses until the text is a valid target prefix again.
- A missed Undo can commit another wrong word. Competitors remain active and correction continues.
- A shorter valid prefix causes earlier words to be retyped, without resetting their counters.

The engine learns from wrong commits too, as the browser does. The pilot never supplies correctness labels to the engine and preserves its single-update learning rollback behavior.

Defaults are four attempts per target-word position, 30 presses per target-word position, and 200 presses per phrase. Constructor arguments `max_attempts_per_word`, `max_presses_per_word`, and `max_presses_per_phrase` override them. Word positions are zero-based. Correction work is charged to the word that triggered correction; counters persist if earlier positions are revisited. Every stop is terminal for that phrase: no forced state reset or silent continuation.

## Results and trace

`run_phrase` returns a dictionary containing:

- `target`, `final_text`, `completed`, `failure_reason`, and optional `error` diagnostics.
- `simulated_elapsed_s` (total), `startup_elapsed_s`, `typing_elapsed_s` (after all startup responses, including later waits), `prediction_wait_s` (total time spent waiting), and `typing_prediction_wait_s`. Typing time includes the response wait after the final Enter; it is not a last-keystroke-only metric.
- `prediction_metadata`: backend provenance, character source responses, effective matrix and hash, latency values, and wait policy.
- `counts` (all presses, Space, Enter, wrong commits, and intended Undo attempts), `attempts_by_word`, and `presses_by_word`.
- `engine_metadata`, `final_snapshot`, and `final_text_known`.
- `trace`: version, engine configuration/source metadata, and ordered event records.

A press counted just before a failed dispatch is an attempted press; its effect on the engine may be uncertain. A transport/dispatch error sets `final_text_known` false and preserves the last successfully observed snapshot. It does not claim to have recovered unknown worker state. Prediction/sample failures retain known engine state. Unexpected errors also close the worker and are reported as `pilot_error`.

Failure reasons include `prediction_failure`, `invalid_click_sample`, `click_source_exhausted`, `word_attempt_budget`, `word_press_budget`, `phrase_press_budget`, `target_clock_unavailable`, `engine_not_ready`, and `bridge_failure`.

Each trace record contains the exact engine event, any intended action and sample metadata, observed text/selection, emitted effects, and a SHA-256 of the complete returned snapshot. Frame events and full prediction responses are recorded too. `trace.prediction_requests` records each request effect, request/due/delivery times, status, and prediction error where applicable. New metadata fields are additive to version-1 traces; older traces still replay. Snapshot hashes avoid storing the entire matrix and clock state after every animation callback.

`replay_trace` creates a fresh bridge with the recorded configuration, requires the same engine source fingerprint, and forwards recorded events without a predictor or click source. It checks full-snapshot hashes and effects. It never retries an event recorded as failed; it verifies only that trace's successful prefix and returns `complete_trace=False`. A prediction failure can still have a fully replayable recorded prefix: `complete_trace` means all recorded engine events were verified, not that the original phrase succeeded. Consult `result.json` for the original outcome.

Pilot artifacts are intentionally separate from canonical study results. They are not suitable for combining with historical simulator metrics without an explicit migration and metric review.

## Tests and next steps

```sh
.venv/bin/python -m unittest \
  OneClick_Bridge.test_bridge \
  OneClick_Simulation.test_bridge_simulated_user \
  OneClick_Simulation.test_bridge_predictions \
  OneClick_Simulation.test_simulated_time \
  OneClick_Simulation.test_recovery \
  OneClick_Text.test_language_model -v
```

The fixture tests do not import models. Run the real local n-gram word, sentence, and deliberate wrong-commit/Undo scenarios explicitly:

```sh
QUICKCLICK_TEST_NGRAM=1 .venv/bin/python -m unittest \
  OneClick_Simulation.test_bridge_predictions.RealNGramTests -v
```

These integration tests use the bundled tiny model and vocabulary. They verify completion and replay, not a fixed number of presses across future TextSlinger versions. The pre-integration fixed-fixture baseline independently locks all 489 event records, effects, and snapshot hashes.

Also run `npm test` from the JavaScript checkout. The pilot tests cover complete phrases, early/BEST/literal selection, observation preservation, timing noise and frame ordering, wrong commits and cascading Undo misses, missing fixtures, all budgets, cleanup, and replay. A focused observation-policy test checks revisiting a shorter correct prefix; the ordinary real-engine correction scenarios exercise the actual commit and Undo implementation through Node.

Subword/HTTP adapters, overlapping input during prediction waits, dynamic context-conditioned transition matrices, multi-phrase learning sessions, the main study-runner migration, and high-throughput optimization remain separate work.

## End-to-end acceptance validation

Use the [validation guide](BRIDGE_VALIDATION.md) to run a word, sentence, mistakes, and Undo; reload saved artifacts; and independently audit metrics against a fresh-engine replay. The dedicated `bridge_validate` CLI supports fixed fixtures, optional local n-gram scenarios, and model-free auditing of saved runs.
