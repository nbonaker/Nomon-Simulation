"""Historical Nomon language-model API adapter.

The NomonWeb client used Imagineville's word and character prediction
endpoints. This adapter exposes the same ``get_words`` contract as the local
KenLM implementation so the Python simulator can use that historical backend.
"""

from __future__ import annotations

import hashlib
import json
import math
import ssl
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from Nomon_Text import kconfig


DEFAULT_WORD_PREDICT_URL = "https://api.imagineville.org/word/predict"
DEFAULT_CHAR_PREDICT_URL = "https://api.imagineville.org/character/predict"


def _log_add_exp(values: list[float]) -> float:
    if not values:
        return -float("inf")
    return float(np.logaddexp.reduce(np.asarray(values, dtype=np.float64)))


def _item_text(item: dict) -> str:
    # Imagineville currently returns ``text``; newer Nomon docs show ``token``.
    return str(item.get("text", item.get("token", "")))


def _item_log_prob(item: dict) -> float:
    return float(item.get("logProb", item.get("logprob", -float("inf"))))


class ImaginevilleLM:
    """Adapt the historical HTTP API to Nomon's keyboard LM interface."""

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.word_predict_url = config.get("word_predict_url", DEFAULT_WORD_PREDICT_URL)
        self.char_predict_url = config.get("char_predict_url", DEFAULT_CHAR_PREDICT_URL)
        self.timeout_s = float(config.get("timeout_s", 20.0))
        self.num_results = int(config.get("num_results", 25))
        self.num_words_total = int(config.get("num_words_total", kconfig.num_words_total))
        self.n_pred = int(config.get("N_pred", kconfig.N_pred))
        self.prob_thres = float(config.get("prob_thres", kconfig.prob_thres))
        ca_file = config.get("ca_file")
        if ca_file is None and ssl.get_default_verify_paths().cafile is None:
            system_ca_file = Path("/etc/ssl/cert.pem")
            ca_file = str(system_ca_file) if system_ca_file.exists() else None
        self.ca_file = ca_file if ca_file is not None else True
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json", "User-Agent": "Nomon-Simulation/1"})
        retry = Retry(
            total=3,
            connect=3,
            read=3,
            status=3,
            backoff_factor=0.25,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        cache_dir = config.get("cache_dir")
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.request_count = 0
        self.cache_hit_count = 0

    def get_words(
        self,
        left_context,
        prefix,
        keys_li,
        num_words_total=None,
    ):
        if num_words_total is None:
            num_words_total = self.num_words_total
        left_context = str(left_context).replace(kconfig.space_char, " ")
        prefix = str(prefix).replace(kconfig.space_char, " ")

        word_results = self._get_results(
            self.word_predict_url,
            {"left": left_context, "prefix": prefix, "num": self.num_results},
        )
        char_results = self._get_results(
            self.char_predict_url,
            {"left": left_context + prefix, "num": self.num_results},
        )

        word_preds, word_probs = self._format_words(
            word_results,
            prefix,
            keys_li,
            num_words_total,
        )
        key_probs = self._format_characters(char_results, keys_li)
        return word_preds, word_probs, key_probs

    def _get_results(self, base_url: str, params: dict) -> list[dict]:
        payload = self._get_json(base_url, params)
        results = payload.get("results")
        if not isinstance(results, list):
            raise RuntimeError(f"Language-model API response has no results list: {base_url}")
        return results

    def _get_json(self, base_url: str, params: dict) -> dict:
        cache_path = self._cache_path(base_url, params)
        if cache_path is not None and cache_path.exists():
            self.cache_hit_count += 1
            with cache_path.open("r", encoding="utf-8") as cache_file:
                return json.load(cache_file)

        url = f"{base_url}?{urlencode(params)}"
        try:
            response = self.session.get(
                base_url,
                params=params,
                timeout=self.timeout_s,
                verify=self.ca_file,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            raise RuntimeError(f"Language-model API request failed: {url}") from exc

        self.request_count += 1
        if cache_path is not None:
            temporary_path = cache_path.with_suffix(".tmp")
            with temporary_path.open("w", encoding="utf-8") as cache_file:
                json.dump(payload, cache_file, separators=(",", ":"), sort_keys=True)
            temporary_path.replace(cache_path)
        return payload

    def _cache_path(self, base_url: str, params: dict) -> Path | None:
        if self.cache_dir is None:
            return None
        cache_key = json.dumps([base_url, params], separators=(",", ":"), sort_keys=True)
        digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def _format_words(
        self,
        results: list[dict],
        prefix: str,
        keys_li: list[str],
        num_words_total: int,
    ) -> tuple[list[list[str]], list[list[float]]]:
        n_pred = self.n_pred
        threshold = -float("inf") if self.prob_thres <= 0 else math.log(self.prob_thres)
        word_preds = [[""] * n_pred for _ in keys_li]
        word_probs = [[-float("inf")] * n_pred for _ in keys_li]
        admitted: list[tuple[int, int, str, float]] = []

        # Match NomonWeb's historical ordering: fill per-next-character buckets
        # in keyboard order, with both per-key and global limits.
        for key_index, key in enumerate(keys_li[: len(kconfig.main_chars)]):
            for item in results:
                word = _item_text(item)
                log_prob = _item_log_prob(item)
                if len(word) <= len(prefix) or word[len(prefix)] != key:
                    continue
                if log_prob <= threshold:
                    continue
                slot = sum(1 for admitted_item in admitted if admitted_item[0] == key_index)
                if slot >= n_pred or len(admitted) >= num_words_total:
                    continue
                admitted.append((key_index, slot, word + " ", log_prob))

        normalizer = _log_add_exp([item[3] for item in admitted])
        for key_index, slot, word, log_prob in admitted:
            word_preds[key_index][slot] = word
            word_probs[key_index][slot] = log_prob - normalizer

        return word_preds, word_probs

    def _format_characters(self, results: list[dict], keys_li: list[str]) -> np.ndarray:
        floor = math.log(0.01)
        char_probs: dict[str, float] = {}
        normalized_values = []
        for item in results:
            token = _item_text(item)
            token = kconfig.space_char if token in {" ", "<sp>"} else token
            log_prob = max(_item_log_prob(item), floor)
            char_probs[token] = log_prob
            normalized_values.append(log_prob)

        normalizer = _log_add_exp(normalized_values)
        return np.asarray(
            [char_probs.get(key, -float("inf")) - normalizer for key in keys_li],
            dtype=np.float64,
        )
