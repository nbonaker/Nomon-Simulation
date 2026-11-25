# Recorded-click QuickClick study

The study runner reuses the historical input preparation with the shared JS engine.
By default it takes the first 10 in-vocabulary phrases in corpus order and runs users
A, B, C, D, F, and G. It uses the recorded text-entry rows and the historical
moving-block bootstrap with 20-row blocks and a deterministic seed per user. Blocks
stay within an original session. Each user starts one sampled stream and consumes
it in order across phrases. The engine starts fresh for every phrase, using its
default timing model rather than the old calibration rows.

Run from `Nomon-Simulation` with Python 3.10+, NumPy, pandas, Node 22+, and a
checkout containing the shared engine and worker. For the WebSocket backend, install
`requirements-websocket.txt` in the same Python environment and start TextSlinger.
See [WebSocket setup](WEBSOCKET.md) for the transport options. Select the model name
advertised by the server. Here is a two-user, two-phrase smoke run using the local
`subword` profile:

```sh
python -m OneClick_Simulation.bridge_study \
  --engine-repo ../Nomon-One-Click-bb-rb \
  --users A F --phrase-count 2 \
  --prediction-server http://127.0.0.1:8768 --prediction-model subword \
  --output-dir /tmp/quickclick-study-subword
```

For the local n-gram profile use port `8767` and `--prediction-model ngram`.
Omit `--users` and `--phrase-count` for the full cohort and corpus. Use a new or
empty output directory for each condition. The direct adapter is also available
with `--backend textslinger-ngram --model PATH --vocabulary PATH`.

The same cohort and corpus produce identical per-user schedules across model
conditions. Because each condition can consume a different number of clicks in a
phrase, the next phrase can start at a different offset in that schedule.
Use `--character-transitions PATH` with a fixed response JSON to hold the engine's
transition matrix constant across backends. Use the latency options to set the
simulated character and word response delays. Both default to zero.

The output includes the corpus and settings in `manifest.json`, the sampled CSV
with source row and session for each user, a `phrases.json` index with consumed
schedule ranges, and the standard `result.json` and `trace.json` for every attempted
phrase. Replay and audit use the existing pilot and validator commands against
the individual phrase directory. The full traces can take substantial disk space.

`summary.json` reports planned, attempted, unattempted, completed, and incomplete
phrases, along with total presses, Undo attempts, wrong commits, budget stops, and
failure reasons. It includes per-user counts, the pooled incomplete rate
(incomplete / attempted), and the mean of the per-user rates. An early stop due
to a bridge, prediction, or click-input failure exits nonzero and reports the
remaining phrases as unattempted. Reaching a typing budget records an incomplete
phrase and continues to the next one. The runner keeps failed phrase artifacts
regardless of how much text was entered. These results use the shared-engine
rules and schema.
