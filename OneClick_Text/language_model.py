from __future__ import division
import hashlib
import json
import math
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
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

    def __init__(self, config_dict=None):
        config_dict = config_dict or {}
        self.key_chars = kconfig.key_chars
        self.strict_errors = bool(config_dict.get("strict_errors", False))
        self.char_predict_url = config_dict.get("char_predict_url", CHAR_PREDICT_URL)
        self.word_api_url = config_dict.get("word_api_url", WORD_API_URL)
        self.char_timeout_s = float(config_dict.get("char_timeout_s", 10.0))
        self.word_timeout_s = float(config_dict.get("word_timeout_s", 15.0))
        cache_dir = config_dict.get("cache_dir")
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json", "User-Agent": "Nomon-Simulation/1"})
        retry = Retry(
            total=3,
            connect=3,
            read=3,
            status=3,
            backoff_factor=0.25,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET", "POST"}),
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.request_count = 0
        self.cache_hit_count = 0

    def _cache_path(self, kind, payload):
        if self.cache_dir is None:
            return None
        cache_key = json.dumps([kind, payload], separators=(",", ":"), sort_keys=True)
        digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def _read_cache(self, cache_path):
        if cache_path is not None and cache_path.exists():
            self.cache_hit_count += 1
            return json.loads(cache_path.read_text(encoding="utf-8"))
        return None

    def _write_cache(self, cache_path, payload):
        if cache_path is None:
            return
        temporary_path = cache_path.with_suffix(".tmp")
        temporary_path.write_text(
            json.dumps(payload, separators=(",", ":"), sort_keys=True),
            encoding="utf-8",
        )
        temporary_path.replace(cache_path)

    def get_key_probs(self, context):
        """
        Return a 27-length list of log-probs for each key char given the full left
        context. Falls back to uniform if context is empty or the API call fails.
        Normalization (log-softmax over the 27 chars) matches oneclick/lm.js format_chars.
        """
        uniform = [-math.log(len(self.key_chars))] * len(self.key_chars)
        if not context:
            return uniform

        params = {"left": context, "num": 64}
        cache_path = self._cache_path("char_predict", [self.char_predict_url, params])
        payload = self._read_cache(cache_path)
        try:
            if payload is None:
                resp = self.session.get(
                    self.char_predict_url,
                    params=params,
                    timeout=self.char_timeout_s,
                )
                resp.raise_for_status()
                payload = resp.json()
                self.request_count += 1
                self._write_cache(cache_path, payload)
            results = payload.get("results", [])
        except Exception as e:
            if self.strict_errors:
                raise RuntimeError(
                    f"OneClick character language-model request failed for context {context!r}"
                ) from e
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
        cache_path = self._cache_path("word_predict", [self.word_api_url, body])
        payload = self._read_cache(cache_path)
        try:
            if payload is None:
                resp = self.session.post(self.word_api_url, json=body, timeout=self.word_timeout_s)
                resp.raise_for_status()
                payload = resp.json()
                self.request_count += 1
                self._write_cache(cache_path, payload)
        except Exception as e:
            if self.strict_errors:
                raise RuntimeError("OneClick word language-model request failed") from e
            print(f"[LM] word predict failed: {e}")
            return [], []

        return payload.get("prefix", []), payload.get("best", [])
