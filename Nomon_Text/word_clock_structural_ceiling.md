# The word-clock structural ceiling

A known limitation of building **word** priors from a **character** language model
(the TextSlinger n-gram adapter in `textslinger_lm.py`). Not a bug — a property of
the architecture.

## The claim

> A char-LM word completion can never out-prior the single next character that starts it.

## Why

A char-LM only produces per-character conditionals. The prior for a word-completion
clock is therefore the **product of its remaining characters' probabilities**. For
prefix `"fo"` and candidate `"fox"`:

```
P("fox" completes | ...fo) = P("x" | ...fo) × P(" " | ...fox) × ...
```

Every factor is ≤ 1, so the product can never exceed its first factor, `P("x" | ...fo)`
— which is exactly the prior on the **char clock for "x"**. So a word clock is always
≤ the char clock that begins it.

Under the old **word-level** KenLM, a word like "the" was modeled as one unit and could
carry a large, sharp probability. Under the char-LM, word clocks are inherently flatter,
so the Bayesian engine needs more clicks to commit to a word → higher Click Load.

## Status (as of the validation sim, 122 phrases)

After the prefix-conditioning fix, accuracy matches/beats the old model, but Click Load
(clicks/char) sits at ~1.73–1.75 vs the ~1.50 baseline — a residual ~0.24 gap. The
ceiling is the leading hypothesis for this gap, but it has **not been measured**: we have
not yet quantified how much of the 0.24 is the ceiling vs. other causes.

## Not addressed by TextSlinger

`ConfigPredictWordsNGram` exposes only decoding knobs (`lm_scale`, insert/delete
penalties, beam width). None reweight a word's prior relative to its first character;
`lm_scale` scales char and word paths together, so it does not change the ceiling.
TextSlinger takes the char-LM factorization as given.

## If we want to close it

It is a **Nomon-side** design choice, not a TextSlinger tuning knob:

1. Blend a **word-level** prior (e.g., a word-frequency/unigram boost) into the word
   clocks, or
2. Accept the gap as the cost of the single-char-model architecture.

Either way, the honest first step is to **measure** the gap (split metrics by phrase
type; inspect high-Click-Load phrases for word clocks that are correct but lose to a
char clock) before designing a fix.
