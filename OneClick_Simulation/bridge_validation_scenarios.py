"""Small independent expectations and fixed inputs for end-to-end acceptance."""
import copy
import json
from pathlib import Path

from .bridge_simulated_user import ClickSample


def _words(*labels):
    return {"prefix": [{"text": label, "logprob": -.1-i} for i, label in enumerate(labels)], "best": []}


def _expected(texts, ids, counts, attempts, per_word, actual_undos):
    return {"enter_texts": texts, "selection_ids": ids,
            "counts": dict(zip(("presses", "space", "enter", "wrong_commits", "undo_attempts"), counts)),
            "attempts_by_word": attempts, "presses_by_word": per_word, "actual_undo_selections": actual_undos}


def fixed_scenarios():
    shared = json.loads(Path(__file__).with_name('bridge_fixtures').joinpath('hello_world.json').read_text())
    word_map = {("", 0): _words(), ("", 1): _words("hello"), ("hello ", 0): _words()}
    sentence_map = {(item['left'], item['observation_count']): item['data'] for item in shared['words']}
    correction = copy.deepcopy(sentence_map)
    correction[("hello ", 1)] = _words("world", "wrong")
    correction[("hello wrong ", 0)] = _words()
    cascade = {("", 0): _words(), ("", 1): _words("hello", "wrong"),
               ("wrong ", 0): _words("oops"), ("wrong oops ", 0): _words(), ("hello ", 0): _words()}
    specs = [
        ("word", "hello", word_map, [0, 0], {},
         _expected(["hello "], [12], [2, 1, 1, 0, 0], [1], [2], 0)),
        ("sentence", "hello world", sentence_map, [0]*4, {},
         _expected(["hello ", "hello world "], [12, 42], [4, 2, 2, 0, 0], [1, 1], [2, 2], 0)),
        ("midword_undo", "hello", word_map, [0, -2.4, 0, 0], {},
         _expected(["", "hello "], [85, 12], [4, 2, 2, 0, 0], [2], [4], 1)),
        ("incorrect_second_word", "hello world", correction, [0, 0, 0, -.88, 0, 0, 0], {},
         _expected(["hello ", "hello wrong ", "hello ", "hello world "], [12, 51, 85, 42],
                   [7, 3, 4, 1, 1], [1, 2], [2, 5], 1)),
        ("missed_undo", "hello", cascade, [0, -.88, 1.8, 0, 0, 0, 0], {},
         _expected(["wrong ", "wrong oops ", "wrong ", "", "hello "], [51, 42, 85, 85, 12],
                   [7, 2, 5, 2, 3], [2], [7], 2)),
        ("incorrect_second_word_delayed", "hello world", correction, [0, 0, 0, -.88, 0, 0, 0],
         {"character_prediction_latency_s": .2, "word_prediction_latency_s": .1},
         _expected(["hello ", "hello wrong ", "hello ", "hello world "], [12, 51, 85, 42],
                   [7, 3, 4, 1, 1], [1, 2], [2, 5], 1)),
    ]
    return [{"name": name, "target": target, "characters": copy.deepcopy(shared['characters']),
             "word_responses": copy.deepcopy(mapping), "offsets": offsets, "options": options, "expected": expected}
            for name, target, mapping, offsets, options, expected in specs]


class MissFirstTargetEnter:
    """Real-model validation only: deliberately sample one competitor's noon.

    No engine state is changed. All later samples have zero noise. The acceptance
    check fails if no wrong commit and subsequent Undo actually occur.
    """
    def __init__(self, user, target="hello"):
        self.user, self.target, self.missed = user, target, False

    def sample(self):
        state = self.user.state
        if not self.missed and state['observations']:
            group = state['words']
            undo = group['clocks'][-1]['id']
            matching = [i for i in state['valid_word_indices'] if i != undo and group['clocks'][i]['label'].lower() == self.target]
            competitors = [i for i in state['valid_word_indices'] if i != undo and group['clocks'][i]['label'].lower() != self.target]
            if matching and competitors:
                self.missed = True
                offset = (group['phases'][matching[0]] - group['phases'][competitors[0]]) * group['period'] / group['num_divs_time']
                return ClickSample(offset)
        return ClickSample()
