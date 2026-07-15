"""Probe: does CausalByteLanguageModel now honor `input_sequence` the way the
Nomon adapter needs (the "Mode 1" convention already validated for NGram)?

Background: as of textslinger 8a18d38, causal_byte.predict_words() had a
"# TODO: this doesn't currently use the input_sequence" comment -- the same
bug class that broke word prediction for the NGram backend in June (see
probe_prefix_conditioning.py). Commit 8c2d8db ("Mostly tested byte model
noisy word predictions") claims to fix this. This probe checks, independent
of textslinger's own test suite, whether the fix behaves the way Nomon's
adapter (Nomon_Text/textslinger_lm.py) actually relies on:

  Mode 1 convention: left_context = text BEFORE the current word (no partial
  word in it); input_sequence = the typed prefix, one deterministic event per
  character. predict_words() should then return FULL words (prefix included),
  so the adapter's bucketing by `word[prefix_len]` is valid -- exactly like
  NGramLanguageModel already does (see test_basic_match /
  test_substitution in tests/test_ngram_predict_words.py).

Checks:
  A) Do returned "words" start with the typed prefix (full word, not a bare
     suffix)? This is the exact bug from June: if not, every word gets
     dropped by the `len(word) <= prefix_len` guard in the adapter.
  B) Equivalence: left_context=ctx, input_sequence=prefix-as-events should
     produce results consistent with left_context=ctx+prefix folded directly
     (mirrors textslinger's own test_input_sequence_equivalent_to_typed_prefix,
     but run against Nomon's char set / usage pattern instead of theirs).
  C) score_item(ctx+prefix) - score_item(ctx): the adapter's "C" constant
     used to re-condition word priors on the prefix. Must be finite.
"""

import numpy as np

from textslinger import (
    CausalByteLanguageModel,
    ConfigPredictWordsByte,
    ConfigPredictCharactersByte,
)
from textslinger.helpers import Device, Precision

BYTE_LLM = "itazap/blt-1b-hf"

with open("Nomon_Text/resources/char_set.txt") as f:
    lines = f.readlines()
CHARS = list(lines[5].rstrip("\n"))
if " " not in CHARS:
    CHARS.append(" ")

CASES = [
    ("the weather is ", "war"),   # expect warm/warning/... (full words starting "war")
    ("i went to the ", "sto"),    # expect store/stop/...
    ("", "th"),                   # no left context at all
]


def load_model():
    print(f"Loading CausalByteLanguageModel(lang_model_name={BYTE_LLM!r}) ...")
    # Device.AUTO resolves to MPS on this Mac, and MPS's caching-allocator
    # warmup chokes on this model (tries to reserve a single ~17 GiB buffer,
    # over PyTorch MPS's per-buffer allocation limit). Force CPU instead.
    return CausalByteLanguageModel(
        character_set=CHARS,
        lang_model_name=BYTE_LLM,
        device=Device.CPU,
        precision=Precision.FP32,
    )


def check_a_full_words(lm, ctx, prefix):
    input_sequence = [[(c, 0.0)] for c in prefix]
    preds = lm.predict_words(
        left_context=ctx,
        input_sequence=input_sequence,
        config=ConfigPredictWordsByte(),
        end_characters=[" ", ".", ",", "!", "?"],
        nbest=10,
        predict_lower=True,
        normalize_logprobs=False,
    ).predictions
    print(f"\n[A] ctx={ctx!r} prefix={prefix!r} -> {len(preds)} preds")
    n_full = 0
    for word, logp in preds:
        is_full = word.startswith(prefix)
        n_full += is_full
        print(f"    {'FULL ' if is_full else 'FRAG?'} {word!r:20s} logp={logp:.3f}")
    print(f"    {n_full}/{len(preds)} returned words start with the typed prefix")
    return preds


def _no_ins_del_config():
    # Mirrors textslinger's own config_no_ins_del fixture (tests/test_causal_byte.py):
    # disable the noisy-keystroke insertion/deletion penalties so a clean,
    # deterministic input_sequence should score identically to typing the
    # same text directly into left_context.
    cfg = ConfigPredictWordsByte()
    cfg.beam_best = 1.0
    cfg.beam_search_max = 4
    cfg.ins_penalty = None
    cfg.del_penalty = None
    return cfg


def check_b_equivalence(lm, ctx, prefix, config_label, config):
    if not prefix:
        return
    # Fold all but the last prefix char directly into left_context; feed the
    # last char via input_sequence. Compare against folding the whole prefix
    # into left_context with no input_sequence at all.
    with_input = lm.predict_words(
        left_context=ctx + prefix[:-1],
        input_sequence=[[(prefix[-1], 0.0)]],
        config=config,
        end_characters=[" ", ".", ",", "!", "?"],
        nbest=5,
        normalize_logprobs=False,
    ).predictions
    folded = lm.predict_words(
        left_context=ctx + prefix,
        config=config,
        end_characters=[" ", ".", ",", "!", "?"],
        nbest=5,
        normalize_logprobs=False,
    ).predictions
    reconstructed = [(prefix[-1] + w, lp) for w, lp in folded]
    print(f"\n[B:{config_label}] ctx={ctx!r} prefix={prefix!r}")
    print(f"    with_input (last char via input_sequence): {with_input}")
    print(f"    folded     (whole prefix in left_context):  {reconstructed}")
    max_gap = max(
        (abs(a[1] - b[1]) for a, b in zip(with_input, reconstructed) if a[0] == b[0]),
        default=float("nan"),
    )
    print(f"    max logp gap on matching words: {max_gap:.4f}")


def check_c_constant(lm, ctx, prefix):
    if not prefix:
        return
    c = lm.score_item(ctx + prefix) - lm.score_item(ctx)
    print(f"\n[C] ctx={ctx!r} prefix={prefix!r} -> C = {c:.4f} "
          f"({'finite, OK' if np.isfinite(c) else 'NOT FINITE - problem'})")


def main():
    lm = load_model()
    no_ins_del = _no_ins_del_config()
    for ctx, prefix in CASES:
        check_a_full_words(lm, ctx, prefix)
        check_b_equivalence(lm, ctx, prefix, "default_config", ConfigPredictWordsByte())
        check_b_equivalence(lm, ctx, prefix, "no_ins_del", no_ins_del)
        check_c_constant(lm, ctx, prefix)


if __name__ == "__main__":
    main()
