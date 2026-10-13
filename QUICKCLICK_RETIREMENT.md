# Retirement of the duplicated Python QuickClick algorithm

The supported QuickClick execution path is now Python simulated-user orchestration → persistent Node bridge → the shared browser JavaScript engine. This cleanup removes the old Python keyboard algorithm and experiment execution, without implementing any old experimental variants in JS.

## Recoverable checkpoint

- Local annotated tag: `quickclick-python-retirement-baseline`
- Simulator checkpoint: `df53ad2ef5c62264634451da46a25ca125138c66`
- Shared-engine checkpoint: `24d73c9d6e0a563cb99c1835ede742913270380f`

The tag preserves exact old implementations, settings, and reproduction commands. It has not been pushed automatically. To inspect or reproduce the old studies without changing this checkout:

```sh
git worktree add --detach ../Nomon-Simulation-python-history quickclick-python-retirement-baseline
```

Supply the historical environment and locally available data/models there. Git does not preserve ignored model files or generated outputs; these remain in their existing locations and have not been deleted. Old paper results are not interchangeable with the new JS baseline.

## Removed execution and retained functionality

| Retired component | Retained replacement or historical analysis |
|---|---|
| `OneClick_Core`, Python QuickClick keyboard, old simulated user/configuration and sanity checks | Shared JS engine, `OneClick_Bridge`, bridge simulated user and validation |
| Full-study runner | Historical `study_data` loaders, bootstrap/preflight/audit helpers and summaries; existing standalone result analyzer |
| Configuration sweeps and algorithm condition generators | Historical `config_summary` aggregation and report generation from saved settings |
| Nomon–QuickClick comparison runner | Historical `comparison` normalization, phrase selection, timing preparation, paired summaries and plots |
| Bootstrap comparison runner | Historical `bootstrap` input preparation and report helpers |
| Clock-speed sweep runner | Historical `speed` aggregation and plots using recorded periods |
| Space/Enter phase-1 and phase-2 runners | Historical `phase1`/`phase2` metrics, comparisons, validation and plots |
| Global Space/Enter sweep runner | Historical `global_sweep` aggregation, ranking and plots; no grid generation, freezing, execution or resume |
| Selected-global comparison runner | Historical `selected_comparison` held-out selection, paired metrics, reporting and plots |
| Phrase reachability runner and duplicated display filtering | Historical `phrase_data` loading; cached HTTP predictor preserved in `OneClick_Text.prediction_client` |
| Old algorithm/runner tests | Retained analysis/data tests, relocated Nomon timing/client tests, adapter tests, and shared-engine tests |

The full symbol mapping is in `User_Simulation/evaluation/historical_oneclick/retirement_map.json`. The removed-module inventory supports an import-dependency regression test. Deleted runners have no compatibility redirects to the bridge: their settings and output semantics differ.

Prediction adapters retain their public loader/prediction APIs. The 27-symbol alphabet lives in the lightweight `OneClick_Text.kconfig`; prediction-pool defaults live in `prediction_config`. Python clock-layout/selection constants are removed. The HTTP adapter is available to explicit callers but is not used by offline historical analysis or automatically connected to the bridge.

Historical input preparation requires explicit click budgets and perfect-click periods. Historical selected-comparison validators/normalizers require explicit Space/Enter periods; heatmaps use periods recorded in result tables. No JS defaults or recomputed legacy speed table are substituted. Saved reachability labels may be used for historical cohort selection; no retained tool claims they establish reachability in the current JS engine.

The separate Nomon core, keyboards, simulated users, baseline evaluations, click-profile tools, data, and model resources are preserved. Nomon-specific timing and client tests formerly embedded in mixed comparison tests now live in `test_nomon_timing_and_clients.py`.

## Current commands

From the simulator repository:

```sh
.venv/bin/python -m OneClick_Simulation.bridge_pilot
.venv/bin/python -m OneClick_Simulation.bridge_validate --output-dir /tmp/quickclick-validation
.venv/bin/python -m OneClick_Simulation.bridge_validate --include-ngram \
  --model Nomon_Text/resources/lm_char_tiny.kenlm \
  --vocabulary Nomon_Text/resources/vocab_lower_100k.txt \
  --output-dir /tmp/quickclick-ngram-validation
```

See the [historical analysis guide](User_Simulation/evaluation/historical_oneclick/README.md) for offline analysis and the [bridge validation guide](OneClick_Simulation/BRIDGE_VALIDATION.md) for simulator checks. Existing historical output schemas remain distinct from bridge-pilot results.

## Cleanup verification

Verified locally on 2026-10-07:

- The retained Python suite ran 100 tests successfully, with its environment-gated n-gram test initially skipped. That test was then run separately with `QUICKCLICK_TEST_NGRAM=1` and passed, including word, sentence, and noisy correction replay.
- All six fixed-response acceptance scenarios and three local n-gram scenarios passed, including saved-result auditing and fresh-engine replay.
- All 20 JavaScript engine/worker tests passed. The JS checkout has no changes.
- Historical tables match the captured pre-retirement fixtures within `1e-10`, preserving discrete values and ordering. Offline CLI checks cover global, phase-1, phase-2, speed, configuration, and comparison analyses; the standalone raw analyzer and retained data-preparation tests also pass.
- Import-boundary tests reject dependencies on retired modules and verify that importing historical analysis does not load simulator or prediction-model modules.
- SHA-256 checks confirmed all 40 tracked data/model/packaged-result assets in [the preservation manifest](User_Simulation/evaluation/historical_oneclick/preserved_assets.sha256.json) are unchanged. Historical output directories were not regenerated; packaged documentation only received historical-use notices.

Validation artifacts from this run are in `/private/tmp/quickclick-retirement-validation`. This is a temporary local location; rerun the documented command with a durable output directory if the artifacts need to be archived.
