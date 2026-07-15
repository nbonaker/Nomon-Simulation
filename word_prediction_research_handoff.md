# Nomon word-prediction research handoff

## Mission

Research and validate the long-term word-prediction behavior of Nomon's
TextSlinger integration.

The central product goal is:

> For the same typed context, TextSlinger should produce on-screen behavior that
> is very similar to legacy KenLM, remain stable across releases, and avoid
> surprising or visually noisy predictions. Improvements over the legacy
> baseline are welcome, but not at the cost of interface consistency or user
> trust.

This is not only a language-model quality project. `get_words()` is a software
interface consumed directly by the keyboard UI, so output shape, count,
determinism, ranking, probability calibration, and latency are part of the
contract.

## Repository and current state

- Repository: `/Users/jasonhong/Nomon-Simulation`
- Branch: `feature/jhong-textslinger-integration`
- The relevant changes are currently **uncommitted**.
- Preserve the existing causal-byte work in `Nomon_Text/textslinger_lm.py` and
  `Nomon_Text/probes/probe_causal_byte_input_sequence.py`.

Current working-tree files related to this investigation:

- Modified: `.gitignore`
- Modified: `Nomon_Text/textslinger_lm.py`
- Modified: `User_Simulation/simulated_user_text.py`
- New: `Nomon_Text/probes/probe_sim_screen_words.py`
- New: `Nomon_Text/probes/probe_sim_screen_words_analyze.py`
- New but from separate causal-byte work:
  `Nomon_Text/probes/probe_causal_byte_input_sequence.py`
- New explanatory artifact: `nomon_textslinger_changes.html`

The earlier investigation is documented at:

`/Users/jasonhong/nomon_word_prediction_divergence_issue.md`

Read that document first for the detailed evidence and original implementation
work items.

## How predictions reach the screen

`Nomon_Text/keyboard.py` calls:

```python
self.lm.get_words(
    left_context,
    context,
    keys_li,
    num_words_total=kconfig.num_words_total,
)
```

Every non-empty word slot returned by `get_words()` becomes a visible,
clickable word clock. There is no later quality filter.

Important configuration:

- `kconfig.N_pred = 3`: at most three word slots per next-character key.
- `kconfig.num_words_total = 17`: at most 17 word clocks globally.

Consequently, any backend implementing `get_words()` must treat these values as
hard UI-contract limits.

## Original problem

Legacy KenLM and TextSlinger generate words differently:

### Legacy KenLM

- Uses a dedicated word-level KenLM model.
- Uses `vocab_lower_100k.txt` through a trie.
- Only vocabulary words extending the exact prefix can be generated.
- Scores candidates with word-level context.
- Applies the per-key limit and global 17-word limit.

### Original TextSlinger adapter

- Loaded only the character-level KenLM model.
- Generated completed character sequences using beam search.
- Had no concept of whether a generated sequence was a real word.
- Filled up to three slots for many keys without enforcing the global cap.
- Could also admit beam hypotheses that did not preserve the exact typed
  prefix.

Observed pre-fix behavior in the full simulator:

- Average TextSlinger word clocks per refresh: **41.5**
- Peak TextSlinger word clocks: **57**
- Out-of-vocabulary displayed predictions: **33.5%**
- Refreshes containing at least one out-of-vocabulary prediction: **72%**
- Example: prefix `tod` produced `today`, `toda`, `todb`, ... `todz`.

A probability-only top-17 cap was insufficient: approximately 32.8% of the
hypothetical capped output was still out of vocabulary. Character n-gram
probability does not reliably distinguish real words from word-shaped junk.

## Current implementation

`Nomon_Text/textslinger_lm.py` now:

1. Loads `Nomon_Text/resources/vocab_lower_100k.txt` once into a set.
2. Requests a wider n-gram beam so junk hypotheses do not deplete the candidate
   pool before filtering.
3. Rejects candidates that:
   - are empty,
   - do not extend the prefix,
   - do not begin with the exact typed prefix, or
   - are absent from the vocabulary.
4. Keeps at most `N_pred` candidates in each next-character bucket.
5. Keeps only the globally highest-probability `num_words_total` bucketed
   candidates.

The key filter is:

```python
if (
    not word
    or len(word) <= prefix_len
    or not word.startswith(prefix.lower())
    or word.lower() not in self.vocabulary
):
    continue
```

The candidate pool is currently widened for the n-gram backend to:

- at least 1,024 completed hypotheses,
- 64 times the requested display count, and
- a minimum active beam width of 300.

These values were chosen empirically to reduce candidate depletion. They have
not yet been characterized comprehensively for latency, fill rate, or quality.
The causal-byte backend receives the vocabulary and prefix filter but retains
its own smaller search defaults.

Also fixed:

- `User_Simulation/simulated_user_text.py` used the misspelled column
  `"Click Time Rlative (s)"`.
- It now correctly reads `"Click Time Relative (s)"`.

## Evidence after the change

Static contexts checked:

- `"the united states of "` + `"amer"`: four current TextSlinger completions,
  all in vocabulary and prefix-consistent.
- `"i "` + `"l"`: 17 completions, all in vocabulary and prefix-consistent.
- `""` + `"th"`: 17 completions, all in vocabulary and prefix-consistent.

Full simulator probe:

- User A
- Session 1
- Real click and phrase data
- `phrase_shuffle_seed=0`

Current TextSlinger screen results:

- Average word clocks per refresh: **15.6**
- Peak word clocks: **17**
- Out-of-vocabulary predictions: **0 / 827**
- Prefix violations: **0 / 827**
- Refreshes above the 17-word cap: **0**

Legacy run from the same probe invocation:

- Average word clocks per refresh: **15.8**
- Peak word clocks: **17**
- Out-of-vocabulary predictions: **0 / 1,030**
- Prefix violations: **0 / 1,030**

One observed metric run:

| Metric | TextSlinger | Legacy KenLM |
|---|---:|---:|
| Mean click load (clicks/character) | 1.606 | 1.951 |
| Mean correction rate | 1.96% | 2.90% |
| Mean word-prediction usage | 31.54% | 19.44% |

Do **not** treat these metric differences as established improvements yet:

- This was one run.
- The simulator still has stochastic behavior beyond phrase shuffling.
- The two runs followed different downstream selection trajectories.
- The simulated user is a target-matching oracle and does not model visual
  scanning, confusion, trust, or screen-reading effort.

The result is encouraging because TextSlinger's usage advantage survived after
the count cap and vocabulary filter, but it requires deterministic,
multi-user, paired validation.

## Important limitations

### Vocabulary membership is not the same as word quality

The legacy vocabulary contains abbreviations and cruft such as `lf`, `ls`,
`thre`, `zzz`, `m'`, `wch`, and `bks`. The current filter provides parity with
legacy's event space; it does not guarantee clean modern English.

### Exact ranking parity is not currently possible

Legacy uses word-level scoring:

`P(word | previous words)`

TextSlinger n-gram uses accumulated character-level scoring:

`P(characters in word | character context)`

Even with identical candidate vocabularies, these systems will rank words and
assign clock probabilities differently.

If exact legacy output is required, investigate:

1. Re-scoring TextSlinger-generated vocabulary candidates with
   `lm_word_*.kenlm`.
2. Enumerating prefix-matching vocabulary candidates and scoring them with the
   legacy word model.
3. Retaining a dedicated legacy-parity backend.

Options 1 and 2 substantially reconstruct the legacy word predictor, so their
maintenance value must be justified.

### Candidate depletion remains possible

Beam slots can still fill with rejected candidates before enough useful words
are generated. The wider beam helped the tested contexts, but sparse prefixes
such as `amer` still returned fewer candidates than some legacy runs.

### Probability calibration needs study

Character and word probabilities compete in one jointly normalized clock
space. Filtering candidates changes the available word mass and therefore the
relative size/ranking of character and word clocks. Matching candidate strings
alone does not guarantee comparable selection dynamics.

## Research priorities

### 1. Build a deterministic golden-replay corpus

Capture a large set of real `(left_context, prefix, keys_li)` calls from legacy
simulations across users and sessions. Store:

- legacy word strings by key and rank,
- legacy word probabilities,
- legacy character probabilities,
- intended target word/character when available,
- relevant model and vocabulary versions.

Replay the same records through TextSlinger without running the interactive
selection loop.

Measure:

- exact full-screen agreement,
- top-1 word agreement,
- top-3 overlap per key,
- global candidate-set overlap/Jaccard similarity,
- rank agreement among shared candidates,
- target-word availability agreement,
- clock-probability divergence,
- number of populated keys,
- total displayed words,
- out-of-vocabulary and prefix violations.

This should become the permanent regression suite.

### 2. Make simulation comparison paired and reproducible

Identify and seed every source of randomness (`random`, NumPy, phrase ordering,
and any simulator-specific sampling). Confirm that repeating one backend with
the same seed produces identical:

- screen refresh sequence,
- selected clocks,
- typed text,
- phrase metrics, and
- output logs.

Then run both backends over:

- all available users,
- all available sessions,
- multiple seeds/trials.

Report paired per-phrase and per-user differences with distributions or
confidence intervals, not only global means.

Primary metrics:

- click load,
- correction rate,
- word-prediction usage,
- entry rate,
- error rate,
- target-word availability,
- target-word rank,
- screen clock count,
- prediction-screen churn.

### 3. Run a controlled decomposition

Compare:

1. Original adapter.
2. Global cap only.
3. Global cap + vocabulary filter.
4. Global cap + vocabulary + exact-prefix filter.
5. Current implementation with widened candidate search.

This isolates whether metric changes come from:

- candidate count,
- vocabulary filtering,
- prefix correctness,
- or candidate-pool width.

The original requested three-way decomposition has not yet been completed.

### 4. Characterize beam-search quality and latency

Sweep:

- `nbest`,
- `max_word_hypotheses`,
- `beam_search_max`,
- `beam_best`,
- maximum word length.

For each setting, measure:

- valid candidate fill rate,
- legacy candidate overlap,
- target-word recall,
- p50/p95/p99 prediction latency,
- memory use,
- worst-case prefixes and contexts.

Choose the smallest search configuration that meets an explicit fill/recall
quality bar. The current 1,024/300 settings are provisional.

### 5. Test edge cases explicitly

Include:

- empty context and empty prefix,
- one-character prefixes,
- long prefixes,
- no vocabulary matches,
- rare prefixes,
- apostrophes,
- punctuation boundaries,
- spaces,
- undo/backspace transitions,
- proper names,
- abbreviations,
- prefix case handling,
- maximum-length words,
- duplicate candidates,
- all-empty word grids,
- non-finite probabilities,
- custom `num_words_total` values.

### 6. Define the backend interface contract

Turn implicit expectations into tests and documentation:

- output grid shape equals `len(keys_li) x N_pred`,
- empty slots use `""` and `-inf`,
- predictions extend the exact prefix,
- no more than `num_words_total` non-empty slots,
- per-key predictions correspond to the next character,
- ranking within each key is descending,
- probabilities are finite or `-inf`,
- joint normalization is valid,
- calls are deterministic for fixed model/config/input,
- latency stays within an agreed UI budget.

Version or fingerprint:

- model files,
- vocabulary,
- adapter configuration,
- TextSlinger version,
- golden outputs.

This is necessary for long-term output stability.

### 7. Evaluate real-user quality

The simulator cannot model the cognitive cost of reading predictions. Plan a
small blinded comparison measuring:

- perceived clutter,
- useful-word discovery time,
- prediction trust,
- accidental selections,
- screen stability,
- preference between legacy and TextSlinger.

Consider whether the legacy vocabulary itself needs a separately versioned,
curated cleanup. Do not silently change it during parity evaluation.

### 8. Extend validation to neural backends

The causal-byte and causal-subword backends also lack a word-vocabulary fence.
Verify:

- candidate semantics,
- exact-prefix behavior,
- vocabulary filtering,
- global cap,
- candidate depletion,
- latency and memory,
- probability calibration.

Do not assume the n-gram beam settings transfer to neural backends.

## Suggested acceptance criteria

Hard requirements:

- 0 out-of-vocabulary displayed predictions.
- 0 prefix violations.
- 0 refreshes above `num_words_total`.
- Correct per-key and grid shape.
- Deterministic replay for fixed inputs and versions.
- No NaN probabilities or crashes.

Similarity requirements should be selected after measuring the replay corpus,
but should cover:

- target-word availability agreement,
- per-key top-3 overlap,
- top-word agreement,
- screen-count difference,
- probability/clock divergence.

Do not pick arbitrary similarity thresholds before observing the baseline
distribution. Report the distribution, inspect the worst disagreements, and
then choose a product-informed quality bar.

## Existing probes and commands

From `/Users/jasonhong/Nomon-Simulation`:

```bash
source venv/bin/activate

python Nomon_Text/probes/probe_sim_screen_words.py textslinger
python Nomon_Text/probes/probe_sim_screen_words.py kenlm
python Nomon_Text/probes/probe_sim_screen_words_analyze.py
```

Static exploratory comparison:

```bash
python -m Nomon_Text.probes.probe_textslinger_vs_kenlm
```

Generated screen logs:

- `Nomon_Text/probes/screen_words_textslinger.jsonl`
- `Nomon_Text/probes/screen_words_kenlm.jsonl`

These logs are ignored by Git.

## Requested research deliverables

Please return:

1. A concise explanation of the remaining sources of output divergence.
2. A deterministic replay-test design, including a proposed JSON schema.
3. Concrete similarity metrics and why each reflects user-facing behavior.
4. A plan for seeding and pairing the full simulator.
5. A beam-configuration experiment with fill-rate and latency results.
6. A recommendation among:
   - clean-but-differently-ranked TextSlinger output,
   - hybrid word-model re-scoring,
   - exact legacy parity mode.
7. Specific regression tests that should run in CI.
8. Any correctness problems found in the current uncommitted implementation.

Prioritize evidence over aggregate averages. Inspect concrete disagreement
examples and connect each metric to what a Nomon user would actually see or
select.
