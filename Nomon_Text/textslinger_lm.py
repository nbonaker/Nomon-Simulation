#!/usr/bin/python

"""Adapter between TextSlinger's language model API and Nomon's keyboard contract.

Exposes the same ``get_words()`` signature that ``keyboard.py`` expects, delegating
to TextSlinger backends internally. Also re-exports ``lognormalize_factor`` so
downstream modules (e.g. ``simulated_user_text.py``) can import it from here
instead of from ``kenlm.kenlm_lm``.
"""

import os
import numpy as np

from Nomon_Text import kconfig


def lognormalize_factor(x):
    """Return the logaddexp.reduce of the flattened input (used to jointly normalize
    a mixed probability space of character and word clocks)."""
    return np.logaddexp.reduce(x.flatten())


# Default location of the character set file relative to this module. Used when
# the caller does not pass ``character_set_path`` in ``lm_config``.
_DEFAULT_CHAR_SET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "resources", "char_set.txt"
)


def _load_character_set(character_set_path):
    """Read Nomon's ``char_set.txt`` format into a list of single-character strings.

    The file has 5 header lines (all comments) followed by a single line of
    characters. A real space ``" "`` is appended so TextSlinger can predict it.
    """
    with open(character_set_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    # Skip the 5 header lines (matches the old CharacterPredictor behavior)
    char_line = lines[5].rstrip("\n")
    chars = list(char_line)
    if " " not in chars:
        chars.append(" ")
    return chars


class TextSlingerLM:
    """Adapter between TextSlinger's LM API and Nomon's keyboard contract.

    Exposes the same ``get_words()`` signature that ``keyboard.py`` expects,
    delegating to TextSlinger backends internally.
    """

    def __init__(self, lm_config: dict):
        """Construct the LM based on a backend config dict.

        Required keys:
          - backend: "ngram" (other backends reserved for Phase 3)
          - lm_path: path to .arpa or .kenlm file (ngram)
        Optional keys:
          - character_set_path: path to char_set.txt (defaults to the bundled
            ``Nomon_Text/resources/char_set.txt``)
          - space_character: pseudo-word used by the n-gram model for spaces
            (defaults to "<sp>")
          - predict_words_config: dict of overrides for
            ``ConfigPredictWordsNGram`` (e.g. nbest scaling, beam tuning)
        """
        self.backend = lm_config.get("backend", "ngram")
        self.lm_config = lm_config

        character_set_path = lm_config.get("character_set_path", _DEFAULT_CHAR_SET_PATH)
        self.character_set = _load_character_set(character_set_path)

        if self.backend == "ngram":
            from textslinger import NGramLanguageModel, ConfigPredictWordsNGram

            space_character = lm_config.get("space_character", "<sp>")
            self.lm = NGramLanguageModel(
                character_set=self.character_set,
                lm_path=lm_config["lm_path"],
                space_character=space_character,
            )
            # Word-prediction beam search config. Defaults match TextSlinger's
            # ConfigPredictWordsNGram; callers may override via lm_config.
            pw_cfg = ConfigPredictWordsNGram()
            overrides = lm_config.get("predict_words_config", {})
            for key, value in overrides.items():
                setattr(pw_cfg, key, value)
            self._predict_words_config = pw_cfg
        else:
            raise ValueError(
                f"Unsupported backend '{self.backend}' in Phase 2; only 'ngram' is supported"
            )

        self.min_log_prob = -float("inf")

    def get_words(
        self,
        left_context,
        prefix,
        keys_li,
        num_words_total=kconfig.num_words_total,
    ):
        """Same contract as ``LanguageModel.get_words()``.

        Returns:
          word_preds: list[list[str]], shape len(keys_li) x N_pred, predicted
            words with a trailing space, ``""`` for empty slots.
          word_probs: list[list[float]], same shape, natural-log probs,
            ``-inf`` for empty slots.
          key_probs: ndarray, shape len(keys_li), natural-log probs per char.

        Character scores and word scores are *jointly* normalized so that
        ``logaddexp.reduce(hstack([key_probs, word_probs])) ≈ 0``.
        """
        # TextSlinger expects real spaces. Nomon uses kconfig.space_char ('_').
        # for reference, check textslinger/tests/test_ngram_predict_words.py on various paramaters
        ts_left_context = (left_context).replace(kconfig.space_char, " ") # for word predictions / _get_word_probs()
        ts_input_sequence = [[(c, 0.0)] for c in prefix]  if prefix else None
        ts_end_characters = [" ", "!", "?", ".", ","]

        ts_full_context = ts_left_context + prefix # for char predictions / _get_char_probs()

        # --- Character path -------------------------------------------------
        key_probs = self._get_char_probs(ts_full_context, keys_li)

        # --- Word path ------------------------------------------------------
        word_preds, word_probs = self._get_word_preds(
            ts_left_context, ts_input_sequence, prefix, keys_li, num_words_total, ts_end_characters
        )

        # --- Joint normalization -------------------------------------------
        key_probs = np.asarray(key_probs, dtype=np.float64)
        word_probs = np.asarray(word_probs, dtype=np.float64)

        # condition word priors on the typed prefix: P(word|context) - logP(prefix|context) = P(suffix| context +prefix)
        constant = self.lm.score_item(ts_full_context) - self.lm.score_item(ts_left_context)

        # subtract this extra constant
        word_probs = word_probs - constant

        normalize_factor = lognormalize_factor(
            np.hstack([key_probs.flatten(), word_probs.flatten()])
        )
        key_probs = key_probs - normalize_factor
        word_probs = word_probs - normalize_factor

        # Blank out word predictions whose probability dropped to -inf
        word_preds = np.where(word_probs != -float("inf"), word_preds, "")

        return word_preds.tolist(), word_probs.tolist(), key_probs

    def _get_char_probs(self, ts_context, keys_li):
        """Return ndarray of natural-log character probs in ``keys_li`` order.

        Unknown keys (not in the character set / not space) get ``-inf``.
        Known keys are floored at ``log(1/50)`` to match the old behavior.
        """
        from textslinger import ConfigPredictCharactersNGram

        preds = self.lm.predict_characters(
            ts_context,
            config=ConfigPredictCharactersNGram(),
            unnormalized_logprobs=True,
        )
        # TextSlinger returns predictions in descending log-prob order over its
        # character_set (which includes " "). Build a lookup.
        char_to_logp = {char: logp for char, logp in preds}

        floor = np.log(1.0 / 50.0)
        key_probs = []
        for key in keys_li:
            # Map the keyboard's space pseudo-char to a real space for lookup.
            lookup = " " if key == kconfig.space_char else key
            if lookup in char_to_logp:
                key_probs.append(max(char_to_logp[lookup], floor))
            else:
                key_probs.append(-float("inf"))
        return np.array(key_probs, dtype=np.float64)

    def _get_word_preds(self, ts_left_context, ts_input_sequence, prefix, keys_li, num_words_total, ts_end_characters):
        """Return (word_preds, word_probs) grid of shape len(keys_li) x N_pred.

        Words are bucketed by the character that immediately follows the prefix.
        Empty slots are filled with ``""`` and ``-inf``.
        """
        n_pred = kconfig.N_pred

        # Ask TextSlinger for more candidates than num_words_total so that after
        # filtering short/empty predictions we still have enough to fill buckets.
        request_n = max(num_words_total * 4, 64)

        preds = self.lm.predict_words(
            left_context=ts_left_context,
            end_characters=ts_end_characters,
            input_sequence=ts_input_sequence,
            config=self._predict_words_config,
            nbest=request_n,
            predict_lower=True,
            return_log_probs=True,
        )

        prefix_len = len(prefix)
        # Bucket completed words by their next-char (the char at position prefix_len).
        # Words equal to or shorter than the prefix are dropped (cannot extend it).
        word_dict = {}
        for word, logp in preds:
            if not word or len(word) <= prefix_len:
                continue
            next_char = word[prefix_len]
            # Normalize the space pseudo-char back to kconfig.space_char for bucketing.
            bucket_key = kconfig.space_char if next_char == " " else next_char
            word_dict.setdefault(bucket_key, []).append((word, logp))

        word_preds = []
        word_probs = []
        for key in keys_li:
            key_word_preds = [""] * n_pred
            key_word_probs = [-float("inf")] * n_pred
            if key in word_dict:
                # Already in descending log-prob order from TextSlinger; take top N_pred.
                top = word_dict[key][:n_pred]
                for i, (word, logp) in enumerate(top):
                    key_word_preds[i] = word + " "
                    key_word_probs[i] = float(logp)
            word_preds.append(key_word_preds)
            word_probs.append(key_word_probs)

        return word_preds, word_probs
