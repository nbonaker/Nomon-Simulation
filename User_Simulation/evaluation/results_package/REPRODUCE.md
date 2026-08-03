# Reproduction guide

Run commands from the repository root using the project virtual environment. Both canonical experiments use external language-model services on cache misses and therefore require explicit approval before a fresh run.

## Validate or regenerate the completed global sweep

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python -m User_Simulation.evaluation.evaluation_oneclick_global_space_enter_sweep \
  --phase all \
  --resume-run-dir User_Simulation/evaluation/outputs/oneclick_global_space_enter_sweep_2026_07_26-18_07_04
```

The resumable runner validates and skips all complete conditions, then regenerates summaries and plots.

## Validate or regenerate the held-out comparison

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python -m User_Simulation.evaluation.evaluation_nomon_oneclick_selected_global_comparison \
  --users A,B,C,D,F,G \
  --trials 5 \
  --phrase-count 20 \
  --selection-run-dir User_Simulation/evaluation/outputs/oneclick_global_space_enter_sweep_2026_07_26-18_07_04 \
  --resume-run-dir User_Simulation/evaluation/outputs/nomon_oneclick_selected_global_comparison_2026_07_28-22_55_09
```

## Relevant tests

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python -m unittest -q \
  User_Simulation.evaluation.test_oneclick_global_space_enter_sweep \
  User_Simulation.evaluation.test_nomon_oneclick_selected_global_comparison \
  User_Simulation.evaluation.test_nomon_oneclick_comparison \
  User_Simulation.evaluation.test_nomon_oneclick_bootstrap_comparison \
  OneClick_Simulation.test_recovery \
  OneClick_Simulation.test_simulated_time
```

## Canonical data lineage

1. Real user click logs are split into training and validation sessions.
2. Training sessions define regime-aware timing profiles and bootstrap schedules.
3. A pooled phrase audit identifies prediction-reachable phrases.
4. The global sweep uses one common 20-phrase tuning set to select 6.0s/6.0s.
5. The final comparison excludes those tuning phrases and deterministically selects 20 new held-out reachable phrases using seed 54321.
6. OG Nomon and OneClick receive paired absolute-second offset schedules; OneClick fixes both periods at six seconds.
7. Strict LM errors invalidate a condition rather than substituting uniform or empty predictions.
