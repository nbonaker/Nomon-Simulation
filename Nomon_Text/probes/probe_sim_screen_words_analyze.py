#!/usr/bin/python
"""Analyze the on-screen word logs produced by probe_sim_screen_words.py.

Reports, per backend: junk rate (out-of-vocab fraction) of everything shown,
junk rate under a hypothetical global num_words_total=17 probability cap,
and how often the single most probable word on screen is junk.

Run probe_sim_screen_words.py twice first (args: "textslinger", then "kenlm")
to produce screen_words_<backend>.jsonl next to this script.
"""
import json, os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))  # Nomon_Text/probes
VOCAB = os.path.join(os.path.dirname(HERE), "resources", "vocab_lower_100k.txt")

with open(VOCAB) as f:
    vocab = set(w.strip().lower() for w in f if w.strip())

for backend in ["textslinger", "kenlm"]:
    path = os.path.join(HERE, f"screen_words_{backend}.jsonl")
    if not os.path.exists(path):
        print(f"=== {backend}: no log at {path}, skipping ===\n")
        continue
    calls = [json.loads(l) for l in open(path)]

    total_slots = junk_slots = calls_with_junk = 0
    calls_over_cap = prefix_mismatches = 0
    peak_slots = 0
    tot_17 = junk_17 = tot_top1 = junk_top1 = 0
    junk_counter = Counter()
    for c in calls:
        shown = c["shown"]
        prefix = c["prefix"].lower()
        junk = [w for w in shown if w.lower() not in vocab]
        prefix_mismatches += sum(not w.lower().startswith(prefix) for w in shown)
        total_slots += len(shown)
        peak_slots = max(peak_slots, len(shown))
        calls_over_cap += len(shown) > 17
        junk_slots += len(junk)
        junk_counter.update(w.lower() for w in junk)
        calls_with_junk += bool(junk)

        # hypothetical global cap: keep only the 17 most probable words shown
        if "probs" in c and c["probs"]:
            pairs = sorted(zip(shown, c["probs"]), key=lambda x: -x[1])
            for w, _ in pairs[:17]:
                tot_17 += 1
                junk_17 += w.lower() not in vocab
            tot_top1 += 1
            junk_top1 += pairs[0][0].lower() not in vocab

    n = len(calls)
    print(f"=== {backend} ===")
    print(f"screen refreshes: {n}, avg word-clocks per refresh: {total_slots/n:.1f}, peak: {peak_slots}")
    print(f"refreshes over 17-word cap: {calls_over_cap}/{n}")
    print(f"junk (out-of-vocab) slots: {junk_slots}/{total_slots} = {100*junk_slots/max(total_slots,1):.1f}%")
    print(f"predictions that do not extend the typed prefix: {prefix_mismatches}/{total_slots}")
    print(f"refreshes with >=1 junk word: {calls_with_junk}/{n} ({100*calls_with_junk/n:.0f}%)")
    if tot_17:
        print(f"junk under global top-17 cap: {junk_17}/{tot_17} = {100*junk_17/tot_17:.1f}%")
        print(f"most-probable word on screen is junk: {junk_top1}/{tot_top1} = {100*junk_top1/tot_top1:.1f}%")
    print(f"most common junk: {junk_counter.most_common(15)}\n")
