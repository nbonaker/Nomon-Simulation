"""
Probe: compare old KenLM LanguageModel.get_words() vs TextSlinger NGramLanguageModel.

Demonstrates that:
1. Character predictions are the same distribution, different log base (ratio = ln(10))
2. Word predictions differ by mechanism (trie enumeration vs beam search)
3. TextSlinger can return predictions shorter than the prefix (edge case for adapter)
4. Space-to-underscore mapping is needed (TextSlinger uses " ", keyboard uses "_")

Run with:
    cd ~/Nomon-Simulation && source venv/bin/activate
    python -m Nomon_Text.probes.probe_textslinger_vs_kenlm
"""

import os
import sys
import numpy as np
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

char_lm_path = os.path.join(BASE_DIR, "Nomon_Text/resources/lm_char_tiny.kenlm")
word_lm_path = os.path.join(BASE_DIR, "Nomon_Text/resources/lm_word_tiny.kenlm")
vocab_path = os.path.join(BASE_DIR, "Nomon_Text/resources/vocab_lower_100k.txt")
char_path = os.path.join(BASE_DIR, "Nomon_Text/resources/char_set.txt")

from Nomon_Text import kconfig
keys_li = kconfig.key_chars

from Nomon_Text.kenlm.kenlm_lm import LanguageModel
from Nomon_Text.kenlm.char_predictor import CharacterPredictor

old_lm = LanguageModel(word_lm_path, char_lm_path, vocab_path, char_path)
cp = CharacterPredictor(char_lm_path, char_path)

from textslinger import NGramLanguageModel

char_set = []
with open(char_path) as f:
    for _ in range(5):
        next(f)
    for line in f:
        for c in line.strip():
            char_set.append(c)
char_set.append(" ")

ts_lm = NGramLanguageModel(character_set=char_set, lm_path=char_lm_path, space_character="<sp>")


def run_probe(context: str, prefix: str, num_words_total: int = 17):
    """Run probe for a given context + prefix."""

    print("=" * 70)
    print(f"CONTEXT: '{context}'  PREFIX: '{prefix}'")
    print("=" * 70)

    # --- OLD API ---
    word_preds, word_probs, key_probs = old_lm.get_words(
        context, prefix, keys_li, num_words_total=num_words_total
    )

    print("\n--- OLD API (KenLM) ---")
    print(f"key_probs shape: {len(key_probs)}")
    print(f"word_preds shape: {len(word_preds)} x {len(word_preds[0])}")
    print("Top 5 key_probs:")
    for i in np.argsort(key_probs)[::-1][:5]:
        print(f"  {keys_li[i]}: {key_probs[i]:.4f}")
    print("Non-empty word_preds:")
    for i in range(len(keys_li)):
        non_empty = [
            (w, round(p, 4)) for w, p in zip(word_preds[i], word_probs[i]) if w != ""
        ]
        if non_empty:
            chars = keys_li[i] if keys_li[i] != "_" else "_(<space>)"
            print(f"  key '{chars}': {non_empty}")

    # Raw char predictor (log10)
    raw = cp.get_characters(context + prefix)
    print(f"\nRaw char predictor (log10), top 5:")
    for c, p in raw[:5]:
        print(f"  '{c}': {p:.6f}")

    # --- NEW API: Characters ---
    char_res = ts_lm.predict_characters(
        context + prefix, normalize_logprobs=False
    ).predictions
    char_d = dict(char_res)

    key_probs_ts = np.array(
        [
            max(char_d.get(" " if k == "_" else k, -float("inf")), np.log(1 / 50))
            for k in keys_li
        ]
    )

    print("\n--- NEW API: Characters (TextSlinger, log_e) ---")
    print("Top 5:")
    for c, p in char_res[:5]:
        print(f"  '{c}': {p:.4f}")

    print("\nTop 5 key_probs (TextSlinger, mapped to keys_li):")
    for i in np.argsort(key_probs_ts)[::-1][:5]:
        print(f"  {keys_li[i]}: {key_probs_ts[i]:.4f}")

    # --- Log base comparison ---
    print("\n--- LOG BASE COMPARISON ---")
    ratio_count = 0
    ratio_sum = 0
    for c, log10_p in raw:
        display_char = c if c != " " else "<space>"
        if c in char_d:
            log_e_p = char_d[c]
            ratio = log_e_p / log10_p if log10_p != 0 else float("inf")
            ratio_count += 1
            ratio_sum += ratio
            print(f"  '{display_char}': old(log10)={log10_p:.6f}, new(log_e)={log_e_p:.6f}, ratio={ratio:.4f}")
    if ratio_count > 0:
        avg_ratio = ratio_sum / ratio_count
        print(f"  Average ratio: {avg_ratio:.4f}  (ln(10) = {np.log(10):.4f})")

    # --- NEW API: Words ---
    word_res = ts_lm.predict_words(
        left_context=context + prefix,
        nbest=num_words_total,
        predict_lower=True,
    ).predictions

    print("\n--- NEW API: Words (TextSlinger beam search, log_e) ---")
    print(f"Num words returned: {len(word_res)}")
    for w, p in word_res[:10]:
        print(f"  '{w}': {p:.4f}")

    # Bucket by next char after prefix
    word_dict_ts = defaultdict(list)
    short_count = 0
    for w, p in word_res:
        if len(w) > len(prefix):
            next_char = w[len(prefix)]
            word_dict_ts[next_char].append((w, p))
        else:
            short_count += 1

    if short_count > 0:
        print(
            f"\n  NOTE: {short_count} predictions shorter than prefix '{prefix}' (filtered from bucketing)"
        )

    print("\nBucketed by next char after prefix:")
    for c in sorted(word_dict_ts.keys()):
        words = word_dict_ts[c][:3]
        print(f"  '{c}': {[(w, round(p, 4)) for w, p in words]}")

    # --- Key probs side by side ---
    print("\n--- KEY PROBS SIDE BY SIDE ---")
    print(f"{'Key':>4} {'Old(mixed)':>12} {'New(log_e)':>12} {'Diff':>10}")
    print("  (Note: old has log10 chars + log_e floor, new is all log_e)")
    top_indices = np.argsort(key_probs)[::-1][:10]
    for i in top_indices:
        diff = key_probs[i] - key_probs_ts[i]
        chars = keys_li[i] if keys_li[i] != "_" else "_"
        print(f"{chars:>4} {key_probs[i]:>12.4f} {key_probs_ts[i]:>12.4f} {diff:>10.4f}")


if __name__ == "__main__":
    # Probe 1: "the united states of " + "amer"
    run_probe("the united states of ", "amer")

    # Probe 2: shorter context "i " + "l"
    run_probe("i ", "l")

    # Probe 3: empty context + single char
    run_probe("", "a")