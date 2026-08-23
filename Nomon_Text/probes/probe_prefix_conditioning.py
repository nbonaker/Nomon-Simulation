"""Compare two fixes for the TextSlinger word-prediction prefix bug.

Fix A (suffix reconstruction): keep left_context = context+prefix, treat the
       returned token as a suffix, full word = prefix+suffix, bucket by suffix[0].
       word_prob = P(suffix | context+prefix)   -- conditioned on the prefix.

Fix B (Keith's intended input_sequence usage): left_context = context (before the
       current word), prefix passed as a deterministic input_sequence, full words
       returned, bucket by word[prefix_len].
       word_prob = P(full word | context)        -- NOT conditioned on the prefix.

We push each through the same joint normalization the adapter uses and report:
  - the full-word strings (sanity: both should be real words now)
  - total word probability mass after joint-normalization (drives word usage)
  - in how many buckets the top word beats its competing single char.
"""
import numpy as np
from textslinger import NGramLanguageModel, ConfigPredictWordsNGram, ConfigPredictCharactersNGram

KEYS = list("abcdefghijklmnopqrstuvwxyz'.#$@_")  # kconfig.key_chars (32); '_' == space
SPACE_KEY = "_"
N_PRED = 3
FLOOR = np.log(1.0 / 50.0)
END_CHARS = [" ", ".", ",", "?", "!"]  # Keith's suggestion (space + sentence punctuation)

with open("Nomon_Text/resources/char_set.txt") as f:
    lines = f.readlines()
chars = list(lines[5].rstrip("\n"))
if " " not in chars:
    chars.append(" ")
lm = NGramLanguageModel(character_set=chars, lm_path="Nomon_Text/resources/lm_char_tiny.kenlm",
                        space_character="<sp>")


def char_probs(ctx):
    preds = lm.predict_characters(ctx, config=ConfigPredictCharactersNGram(), normalize_logprobs=False).predictions
    d = {c: lp for c, lp in preds}
    out = []
    for k in KEYS:
        look = " " if k == SPACE_KEY else k
        out.append(max(d.get(look, -np.inf), FLOOR) if look in d else -np.inf)
    return np.array(out, dtype=np.float64)


def bucket_to_grid(word_dict):
    wp = np.full((len(KEYS), N_PRED), -np.inf)
    strs = [["" ] * N_PRED for _ in KEYS]
    for i, k in enumerate(KEYS):
        if k in word_dict:
            for j, (w, lp) in enumerate(sorted(word_dict[k], key=lambda x: -x[1])[:N_PRED]):
                wp[i, j] = lp
                strs[i][j] = w
    return wp, strs


def fix_a(context, prefix):
    ctx = context + prefix
    cfg = ConfigPredictWordsNGram()
    preds = lm.predict_words(left_context=ctx, config=cfg, nbest=64, predict_lower=True,
                             end_characters=END_CHARS).predictions
    wd = {}
    for suf, lp in preds:
        if not suf:
            continue
        nc = SPACE_KEY if suf[0] == " " else suf[0]
        wd.setdefault(nc, []).append((prefix + suf + " ", lp))
    return bucket_to_grid(wd)


def fix_b(context, prefix):
    cfg = ConfigPredictWordsNGram()  # default has ins_penalty_after_input=0.0 -> auto-complete
    input_sequence = [[(c, 0.0)] for c in prefix]
    preds = lm.predict_words(left_context=context, input_sequence=input_sequence, config=cfg,
                             nbest=64, predict_lower=True,
                             end_characters=END_CHARS).predictions
    pl = len(prefix)
    wd = {}
    for word, lp in preds:
        if not word or len(word) <= pl:
            continue
        nc = SPACE_KEY if word[pl] == " " else word[pl]
        wd.setdefault(nc, []).append((word + " ", lp))
    return bucket_to_grid(wd)


def joint_norm_report(name, context, prefix, grid_fn):
    kp = char_probs(context + prefix)
    wp, strs = grid_fn(context, prefix)
    z = np.logaddexp.reduce(np.hstack([kp.flatten(), wp.flatten()]))
    kp_n, wp_n = kp - z, wp - z
    word_mass = float(np.exp(wp_n[np.isfinite(wp_n)]).sum())
    # buckets where top word beats its competing single char
    wins = 0
    examples = []
    for i, k in enumerate(KEYS):
        best = wp_n[i].max()
        if np.isfinite(best):
            if best > kp_n[i]:
                wins += 1
            if strs[i][0]:
                examples.append((strs[i][0].strip(), round(float(best), 2), round(float(kp_n[i]), 2)))
    examples.sort(key=lambda x: -x[1])
    print(f"  [{name}] total word mass={word_mass:.4f}  buckets where word>char: {wins}")
    print(f"       top completions (word, word_logp_norm, competing_char_logp_norm): {examples[:5]}")


CASES = [
    ("the quick brown ", "fo"),
    ("i want to go ", "ho"),
    ("see you ", "tom"),
    ("", "th"),
    ("united states of ", "amer"),
]
for context, prefix in CASES:
    print(f"context={context!r} prefix={prefix!r}")
    joint_norm_report("Fix A suffix", context, prefix, fix_a)
    joint_norm_report("Fix B inputseq", context, prefix, fix_b)
    print()
