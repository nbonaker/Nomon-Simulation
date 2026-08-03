# Experiment catalog

No existing run should be deleted or moved until references and caches are deliberately migrated. This catalog identifies which outputs should be used.

## Canonical runs

| Run directory | Status | Purpose | Use |
|---|---|---|---|
| `oneclick_global_space_enter_sweep_2026_07_26-18_07_04` | Canonical | Full six-user 6×6 global Space/Enter screen and five-cell confirmation | Source for the selected 6.0s/6.0s setting and global heatmaps |
| `nomon_oneclick_selected_global_comparison_2026_07_28-22_55_09` | Canonical | Six users × five trials × 20 new held-out phrases, OG Nomon versus selected OneClick | Source for final comparison claims, curves, failures, and paired efficiency |

## Supporting experiments

| Run directory | Status | Purpose | Current value |
|---|---|---|---|
| `oneclick_clock_speed_tradeoff_2026_07_25-23_36_05` | Supporting | Initial three-speed completion-by-time experiment | Established the cumulative-curve methodology and motivated slower clocks |
| `oneclick_space_enter_phase1_2026_07_26-12_18_22` | Supporting | Sparse independent Space/Enter screen for A and C | Showed that independent periods were worth investigating |
| `oneclick_space_enter_phase2_2026_07_26-16_58_14` | Supporting | Targeted A/C refinement | Informed the global-grid design; not a final population result |

## Smoke and validation runs

| Run directory | Status | Purpose | Use |
|---|---|---|---|
| `oneclick_global_space_enter_sweep_2026_07_26-18_03_39` | Smoke only | Two-user, three-cell, two-trial integration test | Software validation only; do not present its outcome numbers |
| `nomon_oneclick_selected_global_comparison_2026_07_28-22_53_16` | Smoke only | A/B, one-trial, three-phrase final-run smoke test | Software validation only; do not present its outcome numbers |

## Canonical runners

- `evaluation_oneclick_global_space_enter_sweep.py` — global speed selection.
- `evaluation_nomon_oneclick_selected_global_comparison.py` — final held-out system comparison.
- `evaluation_oneclick_clock_speed_tradeoff.py` — completion-by-time foundation.
- `evaluation_oneclick_phrase_audit.py` — strict phrase reachability audit.
- `synthetic_profiles.py` — historical timing profile and paired bootstrap schedules.

## Supporting or superseded runners

- `evaluation_oneclick_space_enter_phase1.py` and `phase2.py` are useful lineage but are superseded by the full global sweep for decision-making.
- `evaluation_nomon_oneclick_bootstrap_comparison.py` is the older comparison path; the selected-global held-out runner supersedes it for headline results.
- `evaluation_nomon_oneclick_comparison.py` is the earlier Gaussian comparison path; do not mix its results with the final paired-bootstrap experiment.
- `evaluation_oneclick_team_plots.py` and `evaluation_oneclick_failure_plots.py` generated earlier presentation variants; the curated figures in this package are current.
- LM audits, cohort validation, offset-standard-deviation sweeps, and selection replay remain methodological support rather than headline OneClick evidence.
