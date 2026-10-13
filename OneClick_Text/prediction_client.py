"""Cached word-service adapter; contains no keyboard reachability or selection logic."""
from __future__ import annotations

import hashlib

import json

import math

import threading

from pathlib import Path

from typing import Any, Protocol

import requests

from OneClick_Text import prediction_config as oneclick_config

from OneClick_Text import kconfig

WORD_API_URL = "https://api.imagineville.org/rec/distrib"

class WordPredictionClient(Protocol):
    def get_word_predictions(
        self,
        left_context: str,
        observations: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]: ...

def perfect_observation(character: str) -> dict[str, Any]:
    target_probability = 0.99
    other_probability = (1.0 - target_probability) / (len(kconfig.key_chars) - 1)
    return {
        "distrib": [
            {
                "text": key,
                "logProb": math.log(target_probability if key == character else other_probability),
            }
            for key in kconfig.key_chars
        ]
    }

class CachedOneClickWordClient:
    def __init__(
        self,
        cache_dir: Path,
        word_api_url: str = WORD_API_URL,
        timeout_s: float = 20.0,
    ):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.word_api_url = word_api_url
        self.timeout_s = timeout_s
        self.request_count = 0
        self.cache_hit_count = 0
        self._counter_lock = threading.Lock()

    def get_word_predictions(
        self,
        left_context: str,
        observations: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        body = {
            "left": left_context,
            "numBest": oneclick_config.num_best_fetch,
            "numPrefix": oneclick_config.num_prefix_fetch,
            "distribs": observations,
            "config": "nomon",
        }
        cache_path = self._cache_path(body)
        if cache_path.exists():
            with self._counter_lock:
                self.cache_hit_count += 1
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            response = requests.post(
                self.word_api_url,
                json=body,
                timeout=self.timeout_s,
                headers={"Accept": "application/json", "User-Agent": "Nomon-Simulation/1"},
            )
            response.raise_for_status()
            payload = response.json()
            cache_path.write_text(
                json.dumps(payload, separators=(",", ":"), sort_keys=True),
                encoding="utf-8",
            )
            with self._counter_lock:
                self.request_count += 1

        prefix = payload.get("prefix", [])
        best = payload.get("best", [])
        if not isinstance(prefix, list) or not isinstance(best, list):
            raise RuntimeError("OneClick word API response must contain prefix and best lists")
        return prefix, best

    def _cache_path(self, body: dict[str, Any]) -> Path:
        cache_key = json.dumps([self.word_api_url, body], separators=(",", ":"), sort_keys=True)
        digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"
