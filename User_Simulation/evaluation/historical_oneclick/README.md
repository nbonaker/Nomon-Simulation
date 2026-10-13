# Offline historical QuickClick analysis

These modules preserve data preparation, metrics, tables, plotting, and reporting from the retired Python QuickClick studies. They cannot execute those studies, schedule experiment conditions, or reproduce the keyboard algorithm. Original execution is available at the `quickclick-python-retirement-baseline` tag; see the repository [retirement record](../../../QUICKCLICK_RETIREMENT.md).

The standalone `OneClick_Simulation.examples.text_simulation.analyze_results` and `User_Simulation.evaluation.evaluation_oneclick_failure_plots` commands remain available at their existing paths for historical saved outputs.

## Analyze a saved run

Supply explicit input and output directories and select the historical schema. Output must be outside the input directory so original results are not rewritten.

```sh
MPLCONFIGDIR=/tmp/quickclick-matplotlib .venv/bin/python -m User_Simulation.evaluation.historical_oneclick \
  --input-dir /path/to/saved/global-run \
  --output-dir /tmp/historical-global-analysis \
  --kind global --table screen_phrase_results.csv
```

For `global` and `phase2`, the historical reliability threshold comes from `run_config.json` or an explicit `--reliability-floor`. No current experiment defaults are assumed. `--plots` additionally creates plots from the saved rows.

| `--kind` | Inputs and behavior |
|---|---|
| `raw` | Existing `user_*.csv`/combined raw result layout accepted by the standalone analyzer; no `--table` needed |
| `config-sweep` | `--table` phrase results plus `sweep_config.csv`; re-aggregates user/config tables and generates the report; optional saved corpus, sufficiency and LM metadata are read as data |
| `comparison` | `--table` normalized system phrase results; paired/trial/comparison summaries and optional plots |
| `selected-comparison` | `--table` normalized selected-comparison rows; per-user/macro/paired/trial/curve/failure tables and report; plotting also requires `user_bootstrap_profiles.csv` |
| `speed` | `--table` normalized clock-speed rows; per-user summaries and curves; plotting uses recorded period indices/seconds/labels |
| `phase1` | `--table` normalized Space/Enter rows; combo summaries, curves, failure distribution, best-condition comparisons |
| `phase2` | `--table` phase-2 rows; enriched/trial/selection/paired summaries and historical threshold |
| `global` | `--table` explicitly selects screen or confirmation rows; per-user/global/ranked/trial/curve/failure tables |

Missing historical inputs cause an error; there is no model call or simulation fallback. Do not mix screening and confirmation rows. Period labels, failure classifications, metric denominators, ordering, and historical time horizons are retained from the original functions. These tools do not accept the new bridge-pilot JSON format.

The preserved modules also expose their normalization, data-preparation, plotting and reporting functions directly. The full mapping from old modules is recorded in `retirement_map.json`. The old runner import paths are intentionally removed. The `config_summary` functions accept a mapping of saved settings rather than an executable experiment configuration. `study_data` click budgets and perfect-click periods, and `selected_comparison` validation/normalization periods, are explicit arguments.

## Validation

`fixtures/` contains synthetic historical-shaped input rows and reference tables captured from simulator commit `df53ad2ef5c62264634451da46a25ca125138c66` before removal. They are test fixtures, not real study findings. The tests compare tables within `1e-10` for floating-point values and preserve discrete values and row/column order. Do not regenerate reference tables from the extracted implementation to conceal a regression.

```sh
MPLCONFIGDIR=/tmp/quickclick-matplotlib .venv/bin/python -m unittest discover \
  -s User_Simulation/evaluation/historical_oneclick -t . -v
.venv/bin/python -m unittest OneClick_Simulation.test_analyze_results \
  User_Simulation.evaluation.test_nomon_timing_and_clients \
  User_Simulation.evaluation.test_selection_bootstrap -v
```

The import-boundary test checks tracked and new non-ignored Python sources for imports of retired modules, including parent/relative imports and literal dynamic imports. Another test blocks simulator/model imports while importing the offline tools. Original Nomon evaluation code remains separate and supported.
