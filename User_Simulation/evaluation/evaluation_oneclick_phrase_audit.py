"""Audit whether existing real phrases are reachable through OneClick.

Run from the repository root:

    python -m User_Simulation.evaluation.evaluation_oneclick_phrase_audit
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import pandas as pd
import requests

from OneClick_Core import config as oneclick_config
from OneClick_Text import kconfig
from OneClick_Text.language_model import WORD_API_URL
from User_Simulation.evaluation.evaluation_baseline import TEXT_DATA_ROOT, write_json


PHRASE_STATUS_PREDICTION = "prediction_reachable"
PHRASE_STATUS_FALLBACK = "fallback_only"
PHRASE_STATUS_UNREACHABLE = "unreachable"
PHRASE_STATUS_ERROR = "audit_error"


class WordPredictionClient(Protocol):
    def get_word_predictions(
        self,
        left_context: str,
        observations: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]: ...


def _logprob(item: dict[str, Any]) -> float:
    return float(item.get("logprob", item.get("logProb", -99.0)))


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


def active_api_words(
    prefix_results: list[dict[str, Any]],
    best_results: list[dict[str, Any]],
    observation_count: int,
) -> tuple[list[str], list[str]]:
    """Apply the same display filtering as OneClick Keyboard.update_word_list."""

    words_by_letter: dict[str, list[str]] = {}
    prefix_words = []
    seen_prefix = set()
    for item in sorted(prefix_results, key=_logprob, reverse=True):
        text = str(item.get("text", ""))
        if not text or len(text) <= observation_count:
            continue
        normalized = text.lower()
        if normalized in seen_prefix:
            continue
        next_character = text[observation_count].lower()
        if next_character not in kconfig.key_chars:
            continue
        bucket = words_by_letter.setdefault(next_character, [])
        if len(bucket) >= kconfig.n_pred:
            continue
        bucket.append(text)
        prefix_words.append(text)
        seen_prefix.add(normalized)

    best_words = []
    seen_best = set()
    for item in sorted(best_results, key=_logprob, reverse=True):
        text = str(item.get("text", ""))
        normalized = text.lower()
        if not text or len(text) != observation_count or normalized in seen_best:
            continue
        best_words.append(text)
        seen_best.add(normalized)
        if len(best_words) >= kconfig.n_best:
            break
    return prefix_words, best_words


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


def audit_word(
    target_word: str,
    left_context: str,
    client: WordPredictionClient,
) -> dict[str, Any]:
    target_normalized = target_word.lower()
    observations: list[dict[str, Any]] = []
    observed_characters = []
    unsupported_characters = []

    for prefix_length, raw_character in enumerate(target_normalized, start=1):
        observed_character = raw_character if raw_character in kconfig.key_chars else "'"
        if raw_character not in kconfig.key_chars:
            unsupported_characters.append(raw_character)
        observed_characters.append(observed_character)
        observations.append(perfect_observation(observed_character))
        try:
            prefix_results, best_results = client.get_word_predictions(left_context, observations)
        except Exception as exc:
            return {
                "word_status": PHRASE_STATUS_ERROR,
                "prediction_reachable": False,
                "literal_fallback_reachable": False,
                "first_reachable_prefix_length": None,
                "first_reachable_source": None,
                "api_prefixes_queried": prefix_length,
                "unsupported_characters": "".join(dict.fromkeys(unsupported_characters)),
                "audit_error": str(exc),
            }

        prefix_words, best_words = active_api_words(
            prefix_results,
            best_results,
            prefix_length,
        )
        prefix_matches = {word.lower() for word in prefix_words}
        best_matches = {word.lower() for word in best_words}
        if target_normalized in best_matches or target_normalized in prefix_matches:
            source = "best" if target_normalized in best_matches else "prefix"
            return {
                "word_status": PHRASE_STATUS_PREDICTION,
                "prediction_reachable": True,
                "literal_fallback_reachable": "".join(observed_characters) == target_normalized,
                "first_reachable_prefix_length": prefix_length,
                "first_reachable_source": source,
                "api_prefixes_queried": prefix_length,
                "unsupported_characters": "".join(dict.fromkeys(unsupported_characters)),
                "audit_error": None,
            }

    literal_fallback_reachable = "".join(observed_characters) == target_normalized
    return {
        "word_status": (
            PHRASE_STATUS_FALLBACK if literal_fallback_reachable else PHRASE_STATUS_UNREACHABLE
        ),
        "prediction_reachable": False,
        "literal_fallback_reachable": literal_fallback_reachable,
        "first_reachable_prefix_length": len(target_normalized) if literal_fallback_reachable else None,
        "first_reachable_source": "argmax" if literal_fallback_reachable else None,
        "api_prefixes_queried": len(target_normalized),
        "unsupported_characters": "".join(dict.fromkeys(unsupported_characters)),
        "audit_error": None,
    }


def load_real_phrase_candidates(data_root: Path = TEXT_DATA_ROOT) -> pd.DataFrame:
    rows = []
    for phrase_path in sorted(data_root.glob("user_*_text_phrase_data_clean.csv")):
        user_id = phrase_path.name.removeprefix("user_").removesuffix("_text_phrase_data_clean.csv")
        phrase_df = pd.read_csv(phrase_path)
        for phrase_text in phrase_df["Phrase Text"].dropna().astype(str):
            rows.append({"Phrase Text": phrase_text.strip(), "Source User ID": user_id})
    candidates = pd.DataFrame(rows)
    grouped = (
        candidates.groupby("Phrase Text", sort=True)["Source User ID"]
        .agg(lambda values: ",".join(sorted(set(values))))
        .reset_index()
    )
    grouped.insert(0, "Phrase ID", [f"real_{index:03d}" for index in range(1, len(grouped) + 1)])
    return grouped


def audit_phrase(
    phrase_record: dict[str, Any],
    client: WordPredictionClient,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    phrase_text = str(phrase_record["Phrase Text"])
    target_words = phrase_text.split()
    word_rows = []
    committed_words = []
    for word_index, target_word in enumerate(target_words, start=1):
        left_context = " ".join(committed_words)
        if left_context:
            left_context += " "
        result = audit_word(target_word, left_context, client)
        reachable_prefix_length = result["first_reachable_prefix_length"]
        characters_saved = (
            len(target_word) - int(reachable_prefix_length)
            if reachable_prefix_length is not None
            else None
        )
        word_rows.append(
            {
                "phrase_id": phrase_record["Phrase ID"],
                "phrase_text": phrase_text,
                "source_user_ids": phrase_record["Source User ID"],
                "word_index": word_index,
                "target_word": target_word,
                "target_word_length": len(target_word),
                "left_context": left_context,
                "characters_saved_at_first_reachability": characters_saved,
                **result,
            }
        )
        committed_words.append(target_word)

    statuses = [row["word_status"] for row in word_rows]
    if PHRASE_STATUS_ERROR in statuses:
        phrase_status = PHRASE_STATUS_ERROR
    elif PHRASE_STATUS_UNREACHABLE in statuses:
        phrase_status = PHRASE_STATUS_UNREACHABLE
    elif PHRASE_STATUS_FALLBACK in statuses:
        phrase_status = PHRASE_STATUS_FALLBACK
    else:
        phrase_status = PHRASE_STATUS_PREDICTION

    phrase_row = {
        "phrase_id": phrase_record["Phrase ID"],
        "phrase_text": phrase_text,
        "source_user_ids": phrase_record["Source User ID"],
        "num_words": len(word_rows),
        "phrase_status": phrase_status,
        "all_words_prediction_reachable": all(
            row["prediction_reachable"] for row in word_rows
        ),
        "all_words_overall_reachable": all(
            row["prediction_reachable"] or row["literal_fallback_reachable"]
            for row in word_rows
        ),
        "prediction_reachable_word_count": sum(
            row["prediction_reachable"] for row in word_rows
        ),
        "target_character_count": sum(row["target_word_length"] for row in word_rows),
        "characters_saved_at_first_reachability": sum(
            row["characters_saved_at_first_reachability"] or 0 for row in word_rows
        ),
        "full_word_required_count": sum(
            row["characters_saved_at_first_reachability"] == 0 for row in word_rows
        ),
        "fallback_only_word_count": statuses.count(PHRASE_STATUS_FALLBACK),
        "unreachable_word_count": statuses.count(PHRASE_STATUS_UNREACHABLE),
        "audit_error_word_count": statuses.count(PHRASE_STATUS_ERROR),
        "fallback_only_words": ",".join(
            row["target_word"] for row in word_rows if row["word_status"] == PHRASE_STATUS_FALLBACK
        ),
        "unreachable_words": ",".join(
            row["target_word"] for row in word_rows if row["word_status"] == PHRASE_STATUS_UNREACHABLE
        ),
    }
    return phrase_row, word_rows


def build_output_dir(base_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    output_dir = base_dir / f"oneclick_phrase_audit_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def run_audit(
    phrase_df: pd.DataFrame,
    client: WordPredictionClient,
    workers: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = phrase_df.to_dict("records")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(lambda record: audit_phrase(record, client), records))
    phrase_rows = [result[0] for result in results]
    word_rows = [word_row for result in results for word_row in result[1]]
    return pd.DataFrame(phrase_rows), pd.DataFrame(word_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / ".cache" / "oneclick_phrase_audit",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-phrases", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    phrase_df = load_real_phrase_candidates()
    if args.max_phrases is not None:
        phrase_df = phrase_df.head(args.max_phrases).copy()

    client = CachedOneClickWordClient(args.cache_dir)
    phrase_results_df, word_results_df = run_audit(phrase_df, client, workers=args.workers)
    output_dir = build_output_dir(args.output_dir)
    phrase_results_df.to_csv(output_dir / "phrase_reachability.csv", index=False)
    word_results_df.to_csv(output_dir / "word_reachability.csv", index=False)
    status_counts = phrase_results_df["phrase_status"].value_counts().to_dict()
    prediction_source_counts = word_results_df["first_reachable_source"].value_counts().to_dict()
    summary = {
        "phrase_count": int(len(phrase_results_df)),
        "word_count": int(len(word_results_df)),
        "unique_source_phrase_count": int(len(phrase_df)),
        "phrase_status_counts": {key: int(value) for key, value in status_counts.items()},
        "prediction_source_counts": {
            key: int(value) for key, value in prediction_source_counts.items()
        },
        "prediction_reachable_phrase_rate": float(
            phrase_results_df["all_words_prediction_reachable"].mean()
        ),
        "overall_reachable_phrase_rate": float(
            phrase_results_df["all_words_overall_reachable"].mean()
        ),
        "mean_first_reachable_prefix_length": float(
            word_results_df["first_reachable_prefix_length"].mean()
        ),
        "mean_characters_saved_at_first_reachability": float(
            word_results_df["characters_saved_at_first_reachability"].mean()
        ),
        "full_word_required_count": int(
            (word_results_df["characters_saved_at_first_reachability"] == 0).sum()
        ),
        "api_request_count": int(client.request_count),
        "api_cache_hit_count": int(client.cache_hit_count),
        "cache_dir": str(args.cache_dir.resolve()),
        "audit_assumption": "perfect intended-letter observations with OneClick display limits",
    }
    write_json(output_dir / "audit_summary.json", summary)

    print(f"Saved OneClick phrase audit to: {output_dir}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
