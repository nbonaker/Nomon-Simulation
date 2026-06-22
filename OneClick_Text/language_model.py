from __future__ import division
import math
import requests
from OneClick_Text import kconfig
from OneClick_Core import config


# The character API is served by the nomontomcat LM server. It returns
# {"results": [{"token": <char>, "logProb": <float>}, ...]} where the space token is a
# literal " "; only the 27 key_chars are kept, so non-letter tokens are ignored. The
# word-level decoder (/rec/distrib) is served by imagineville.
CHAR_PREDICT_URL = "https://nomontomcat.csail.mit.edu/LM/character/predict"
WORD_API_URL = "https://api.imagineville.org/rec/distrib"

LOG_CLAMP_MIN = math.log(0.01)


def _log_add_exp(log_probs):
    """log(sum(exp(x))) for a list of log values — numerically stable."""
    if not log_probs:
        return float('-inf')
    max_lp = max(log_probs)
    return max_lp + math.log(sum(math.exp(lp - max_lp) for lp in log_probs))


class LanguageModel:
    """
    HTTP client for the nomontomcat character API and the imagineville word (/rec/distrib) API.

    - get_key_probs(context): P(next char | full left context) over the 27 key chars,
      used to place the letter clocks once per word (mirrors the JS char_base query in
      oneclick/lm.js, which conditions on the full left context+prefix).
    - get_word_predictions(left, observations): POST the observation matrix and return
      the API's prefix completions and BEST decodings.
    """

    def __init__(self):
        self.key_chars = kconfig.key_chars

    def get_key_probs(self, context):
        """
        Return a 27-length list of log-probs for each key char given the full left
        context. Falls back to uniform if context is empty or the API call fails.
        Normalization (log-softmax over the 27 chars) matches oneclick/lm.js format_chars.
        """
        uniform = [-math.log(len(self.key_chars))] * len(self.key_chars)
        if not context:
            return uniform

        try:
            resp = requests.get(
                CHAR_PREDICT_URL,
                params={"left": context, "num": 64},
                timeout=10,
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
        except Exception as e:
            print(f"[LM] char predict failed for context '{context}': {e}")
            return uniform

        raw = {}
        for item in results:
            token = item.get("token", "")
            if token in self.key_chars:
                raw[token] = max(float(item["logProb"]), LOG_CLAMP_MIN)
        for c in self.key_chars:
            raw.setdefault(c, LOG_CLAMP_MIN)

        log_z = _log_add_exp(list(raw.values()))
        if log_z == float('-inf'):
            return uniform
        return [raw[c] - log_z for c in self.key_chars]

    def get_word_predictions(self, left_context, observations):
        """
        POST the observation matrix to the word API.

        Returns (prefix, best): lists of word-prediction dicts as returned by the API
        (each carries "text" and "logprob"). Callers read logprob via keyboard._logprob,
        which tolerates the API's lowercase 'logprob' (cf. oneclick/lm.js:136,159).
        """
        body = {
            "left": left_context,
            "numBest": config.num_best_fetch,
            "numPrefix": config.num_prefix_fetch,
            "distribs": observations,
            "config": "nomon",
        }
        try:
            resp = requests.post(WORD_API_URL, json=body, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"[LM] word predict failed: {e}")
            return [], []

        return data.get("prefix", []), data.get("best", [])
