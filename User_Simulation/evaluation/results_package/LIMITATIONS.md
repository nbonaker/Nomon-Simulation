# Interpretation and limitations

## What the results support

- OneClick can substantially reduce clicks on phrases that both systems complete.
- A slower global OneClick setting improves reliability across most tested users.
- Completion-by-time curves show a real modeled speed/reliability tradeoff without assigning artificial completion times to failures.
- Recovery state preservation and protected Undo prevent the earlier simulator artifacts caused by leaked wrong words and complete word re-entry.

## What the results do not establish

- **Not real writing time.** Simulated time includes clock-controlled waiting only. Reading, decision-making, visual search, UI latency, network latency, and human pauses are excluded.
- **Not a human adaptation study.** Historical click offsets were preserved in absolute seconds across tested clock periods. Real users may change their timing at a new speed.
- **Not unrestricted language entry.** The final experiments used 20 strictly prediction-reachable phrases to isolate timing and recovery from language-model coverage.
- **Not a global optimum.** The global screen tested periods from 2.2 to 6.0 seconds. The selected 6.0s/6.0s cell was the best tested reliability-first compromise, not proof that no better setting exists outside the grid.
- **Not universal reliability.** User G completed only 58% of held-out OneClick trials versus 94% with OG Nomon.
- **Not equal architecture.** OG Nomon accumulated evidence through its historical selection mechanism; OneClick made individual letter and word decisions with protected recovery.

## User G

G had the largest absolute click-offset standard deviation: about **0.509s**, or **8.5% of OneClick’s six-second rotation**. In the held-out comparison, G averaged 6.5 wrong commits and 15 Undo attempts per OneClick phrase. Of G’s 42 OneClick failures, 18 exhausted Undo recovery, 16 exhausted letter clicks, five exhausted target-Enter clicks, and three failed because the target was not displayed.

This indicates a timing-variability and recovery robustness problem rather than primarily a language-model coverage problem. It is also why aggregate OneClick efficiency must never be presented without per-user completion.

## Reporting rules

- Always show final completion, completion by time, and failure reasons before conditioned efficiency.
- Label click and time reductions as **conditioned on both systems completing the same phrase trial**.
- Keep the global selection experiment separate from the held-out system comparison.
- State that OneClick used a fixed 6.0s Space and 6.0s Enter/Undo period, while OG Nomon used historical personalized periods.
- Do not use smoke-run outcomes in presentations.
