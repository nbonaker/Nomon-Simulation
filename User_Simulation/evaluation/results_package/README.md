# Nomon / OneClick simulation results package

This directory is the canonical team handoff for the completed simulation work. It contains the presentation figures, essential summary tables, experiment lineage, limitations, and reproduction commands. The original run directories remain unchanged under `../outputs/`.

## Questions answered

1. **Can state-preserving OneClick recovery prevent a mistimed Enter or Undo from forcing complete word re-entry?**  
   Yes. Snapshot restoration, protected Undo, shared switch-offset learning, and real correction success checks removed the original leaked-word and repeated-letter penalties.

2. **Is there one useful global Space/Enter clock combination?**  
   Among the tested 2.2–6.0 second grid, the best reliability-first compromise was **6.0s Space / 6.0s Enter-and-Undo**.

3. **How does that selected OneClick setting compare with OG Nomon on held-out phrases?**  
   OneClick completed phrases earlier and used far fewer clicks when both systems succeeded, but its final reliability was lower because user G remained difficult.

## Headline results

### Global OneClick speed selection

- Selected setting: **6.0s Space / 6.0s Enter-and-Undo**.
- Confirmed final completion: **87.8%**.
- Completed by 60 / 120 / 180 seconds: **35.3% / 75.0% / 85.3%**.
- Users reaching at least 80% completion: **5 of 6**.
- Worst user: G, **53%** in the selection experiment.

### Held-out OG Nomon comparison

| Metric | OG Nomon | OneClick |
|---|---:|---:|
| Final phrase completion | 97.3% | 90.8% |
| Completed by 60s | 30.0% | 50.0% |
| Completed by 120s | 70.5% | 83.8% |
| Completed by 180s | 87.0% | 88.8% |

Across the **533 phrase trials completed by both systems**, OneClick used **51.5% fewer clicks** and **13.5% less simulated clock-interaction time**. These efficiency reductions are conditioned on mutual success and must be presented beside completion rates.

OneClick completion by user was A 98%, B 99%, C 95%, D 100%, F 95%, and G 58%. G’s large absolute click-offset spread caused wrong commits, repeated Undo attempts, and click-budget exhaustion.

## Figures to show the team

1. [`figures/01_global_screen_heatmaps.pdf`](figures/01_global_screen_heatmaps.pdf) — how the global 6.0s/6.0s setting was selected.
2. [`figures/02_comparison_dashboard.pdf`](figures/02_comparison_dashboard.pdf) — headline OG Nomon versus OneClick results.
3. [`figures/03_per_user_completion_by_time.pdf`](figures/03_per_user_completion_by_time.pdf) — individual behavior and the user-G limitation.
4. [`figures/04_failure_reasons.pdf`](figures/04_failure_reasons.pdf) — remaining reliability bottlenecks.

PNG versions are included for documents and messaging; PDF versions are preferred for slides.

## What is canonical

- The full global selection run: `oneclick_global_space_enter_sweep_2026_07_26-18_07_04`.
- The held-out final comparison: `nomon_oneclick_selected_global_comparison_2026_07_28-22_55_09`.
- The figures and tables copied into this package from those two runs.

See [`EXPERIMENT_CATALOG.md`](EXPERIMENT_CATALOG.md) before using any other output directory. See [`LIMITATIONS.md`](LIMITATIONS.md) before making real-user or real-time claims.

## Project conclusion

The simulation supports OneClick as a promising efficiency-oriented interface, not yet as a universally reliable replacement for OG Nomon. Additional broad speed sweeps have diminishing value. The most useful next step is a browser-level implementation and small real-user evaluation; adaptive personalization is a strong follow-on research direction.
