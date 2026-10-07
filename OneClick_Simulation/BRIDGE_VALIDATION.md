# End-to-end validation of the shared-engine simulator

This acceptance command runs the opt-in simulated user through the real Python–Node bridge, saves its artifacts, and audits them against a fresh JavaScript engine. It validates recording accuracy and reproducibility, not real-user performance. It does not change the browser algorithm or migrate study runners.

## Run the fixed scenarios

From `Nomon-Simulation`, with Python 3.10+, Node 22+, and the sibling `Nomon-One-Click` checkout:

```sh
.venv/bin/python -m OneClick_Simulation.bridge_validate \
  --output-dir /tmp/quickclick-validation
```

Fixture mode uses only the standard library and the bridge. The scenarios run in this order, each in a fresh session at rotation index 5:

| Scenario | Expected Enter outcomes | Space / Enter | Wrong commits | Intended Undo / actual Undo |
|---|---|---|---|---|
| `word` | `hello` | 1 / 1 | 0 | 0 / 0 |
| `sentence` | `hello` → `hello world` | 2 / 2 | 0 | 0 / 0 |
| `midword_undo` | empty text → `hello` | 2 / 2 | 0 | 0 / 1 |
| `incorrect_second_word` | `hello` → `hello wrong` → `hello` → `hello world` | 3 / 4 | 1 | 1 / 1 |
| `missed_undo` | `wrong` → `wrong oops` → `wrong` → empty text → `hello` | 2 / 5 | 2 | 3 / 2 |
| `incorrect_second_word_delayed` | Same correction sequence as above for the second word | 3 / 4 | 1 | 1 / 1 |

The engine retains a trailing space after committed words; the table omits it for readability. Midword Undo has one actual Undo selection despite zero intended Undo presses: the simulated user intended to select a word but missed. Missed Undo has three correction attempts but only two actual Undo selections. Both distinctions are checked.

The first five cases use zero prediction latency. The last uses 0.2 seconds for character responses and 0.1 seconds for word responses. These are simulated delays; wall time is irrelevant. Fixtures, click offsets, expected selections, counters, and text transitions are specified independently in `bridge_validation_scenarios.py`.

`--engine-repo PATH` and `--node-executable PATH` override the default checkout and Node executable. This validation command intentionally uses the pilot's default speed and budgets.

## Include the real local n-gram model

```sh
.venv/bin/python -m OneClick_Simulation.bridge_validate \
  --include-ngram \
  --model Nomon_Text/resources/lm_char_tiny.kenlm \
  --vocabulary Nomon_Text/resources/vocab_lower_100k.txt \
  --output-dir /tmp/quickclick-validation-ngram
```

This runs the six fixture scenarios plus `ngram_word`, `ngram_sentence`, and `ngram_correction`. TextSlinger is loaded once and reused across the three fresh sessions. Both model and vocabulary paths are required; there are no downloads. Use the simulator environment with its local TextSlinger installation and a KenLM build supporting the bundled order-12 model.

All model scenarios use 0.2-second character and 0.1-second word latency. The correction case deliberately samples a competing word clock's noon for one Enter press, then returns to zero-noise samples. It never mutates engine state. Acceptance requires an actual wrong commit, an intended Undo, an actual Undo selection, and eventual correct completion. It does not hard-code model-dependent press counts. Every sampled offset is saved in the trace.

## Artifacts and reports

Each scenario directory contains:

- `scenario.json`: independently specified target, settings, click-source description, and acceptance expectations.
- `result.json`: the pilot's recorded result, without its embedded trace.
- `trace.json`: all engine events, response payloads, intentions, effects, snapshot hashes, and prediction-request timing records.
- `validation.json`: pass/fail status, verified event count, independently reconstructed metrics and selections, and the first audit mismatch. Scenario expectation mismatches are also reported.

The command reloads the saved result and trace before auditing them. `summary.json` collects each scenario's status and metrics. Successful completion requires both `finished: true` and `valid: true`; an interrupted suite cannot report full success. Exit status is zero only if every requested scenario passes. Scenario failures do not prevent the remaining scenarios from producing diagnostics. Use a new output directory to retain reports across runs; rerunning with the same directory replaces files for the scenarios being run.

Within a validation report:

- `valid` means the complete successful run passed both the audit and any scenario expectations.
- `complete_trace` means every recorded event replayed consistently, all prediction requests were delivered, and frame pairs were complete. A result-metric mismatch can leave this true while `valid` is false.
- `verified_events` and `last_verified_text` identify the verified prefix when a later event fails.
- `errors` names the mismatching field, expected/actual values where available, and a zero-based event index where applicable.
- `reconstructed` contains replay-derived counters, attempts, per-word presses, timing metrics, text transitions, and Enter selections. Treat partial reconstructed values in a failed report as diagnostics, not certified results.

The auditor derives actual selections and committed text from engine snapshots, and checks the recorded observation summaries against them. It reconstructs press counts, incorrect commits, Undo intentions, attempts, and per-word allocation independently of result counters. It also checks the simulator's intention fields for consistency with the target and actual engine state.

Timing checks use independently reconstructed frame cadence and prediction-request/response matching. Startup requests overlap, so wait time is the union of request-to-response intervals, not the sum of 27 character requests. Typing time starts after startup responses and includes later prediction waits, including the response after the final Enter. Presses may not occur with pending predictions.

Discrete state, snapshot hashes, effects, and counters must match exactly. Independently recomputed times allow an absolute tolerance of `1e-10` seconds. The effective transition matrix, its hash, and source character responses are checked against replay. Engine source fingerprints must match. Backend and historical runtime metadata are checked for consistency between artifacts; replay does not establish that a particular model produced the recorded predictions. A different checkout location or Node version is allowed if the protocol and engine sources match.

## Audit an existing run without a model

```sh
.venv/bin/python -m OneClick_Simulation.bridge_validate \
  --audit /tmp/quickclick-validation-ngram/ngram_correction
```

Audit mode is read-only and prints its report. It neither loads TextSlinger nor calls a prediction service. It can also audit a current `bridge_pilot` output directory containing `result.json` and `trace.json`; `scenario.json` is optional. This full-run audit requires the prediction metadata/request records produced by the current pilot. Older minimal traces can still use the existing `bridge_pilot --replay` command.

Python API:

```python
from OneClick_Simulation.bridge_validation import validate_run

report = validate_run(result, trace, engine_repo=None, node_executable="node")
assert report["valid"], report["errors"]
```

The API audits artifacts; the CLI additionally checks the independently specified scenario expectations. Failed or incomplete runs never pass full-run validation. An event recorded as failed is not dispatched again. All workers close on success, mismatch, or exception.

## Tests

```sh
.venv/bin/python -m unittest \
  OneClick_Simulation.test_bridge_validation \
  OneClick_Bridge.test_bridge \
  OneClick_Simulation.test_bridge_simulated_user \
  OneClick_Simulation.test_bridge_predictions \
  OneClick_Simulation.test_simulated_time \
  OneClick_Simulation.test_recovery \
  OneClick_Text.test_language_model -v
```

Run `npm test` in `Nomon-One-Click` as well. The new validation tests cover the six fixed scenarios, disk serialization, intentional result/trace corruption, response ordering, pending predictions, truncated and failed traces, model-free audit, nonzero CLI exits, and worker cleanup. Run the `--include-ngram` command above for the full local-model acceptance check.
