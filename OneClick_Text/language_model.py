"""Local TextSlinger adapter for the OneClick/QuickClick simulator."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path
import subprocess
from typing import Any, Sequence

# The OneClick backend is local-only. Set these before importing TextSlinger,
# because Transformers reads the offline flags during module initialization.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from OneClick_Core import config
from OneClick_Text import kconfig

from textslinger import InputChannelConfig, InputEvent, WordList
from textslinger.causal_subword import (
    CausalSubwordLanguageModel,
    ConfigPredictCharactersSubword,
    ConfigPredictWordsSubword,
)
from textslinger.ngram import (
    ConfigPredictCharactersNGram,
    ConfigPredictWordsNGram,
    NGramLanguageModel,
)
from textslinger.helpers import Device, ModelQuantization, Precision


LOG_CLAMP_MIN = math.log(0.01)
CAUSAL_SUBWORD_BACKEND = "causal_subword"
NGRAM_BACKEND = "ngram"
SUPPORTED_BACKENDS = (CAUSAL_SUBWORD_BACKEND, NGRAM_BACKEND)
DEFAULT_RECOGNIZER_NBEST = 1000
DEFAULT_CHARACTER_RESULT_CACHE_SIZE = 4096
DEFAULT_WORD_RESULT_CACHE_SIZE = 20000
WORD_SPELLING_CHARACTERS = tuple(kconfig.key_chars)
NGRAM_MODEL_ALPHABET = tuple("abcdefghijklmnopqrstuvwxyz '.,?!")

DEFAULT_SUBWORD_WORD_SEARCH = {
    "max_active_hypotheses": 60,
    "beam_best": 8.0,
    "max_terminal_token_paths": 12800,
    "max_omitted_prefix_paths": 65536,
    "max_omitted_joint_prefix_paths": 64,
    "text_state_remaining_mass_tolerance": 1e-3,
}
DEFAULT_NGRAM_WORD_SEARCH = {
    "max_active_hypotheses": 100,
    "beam_best": 8.0,
    "remaining_mass_tolerance": 1e-3,
    "max_character_depth": 64,
}


def _log_add_exp(log_probs: Sequence[float]) -> float:
    """Return log(sum(exp(x))) without losing precision."""
    if not log_probs:
        return float("-inf")
    maximum = max(log_probs)
    if maximum == float("-inf"):
        return maximum
    return maximum + math.log(sum(math.exp(value - maximum) for value in log_probs))


def _prediction_log_mass(prediction: Any) -> float:
    probability = prediction.probability
    value = probability.log_lower_bound
    if value is None:
        value = probability.log_retained_joint_mass
    value = float(value)
    if math.isnan(value) or value == float("inf"):
        raise ValueError("TextSlinger returned an invalid prediction score")
    return value


def validate_model_directory(model_path: str | Path) -> Path:
    """Validate the minimum files needed for an offline Hugging Face load."""
    path = Path(model_path).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"language-model directory does not exist: {path}")

    missing = []
    if not (path / "config.json").is_file():
        missing.append("config.json")
    if not any(
        (path / name).is_file()
        for name in ("tokenizer.json", "tokenizer_config.json")
    ):
        missing.append("tokenizer.json or tokenizer_config.json")
    weight_files = (
        list(path.glob("*.safetensors"))
        + list(path.glob("pytorch_model*.bin"))
        + list(path.glob("*.safetensors.index.json"))
        + list(path.glob("pytorch_model*.bin.index.json"))
    )
    if not weight_files:
        missing.append("model weights")
    if missing:
        raise ValueError(
            f"language-model directory {path} is incomplete; missing: "
            + ", ".join(missing)
        )
    return path


def validate_ngram_model_file(model_path: str | Path) -> Path:
    """Return a resolved local KenLM/ARPA path or fail before output creation."""
    path = Path(model_path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"n-gram model file does not exist: {path}")
    return path


def validate_vocabulary_file(vocabulary_path: str | Path) -> Path:
    """Return a resolved TextSlinger word-list path."""
    path = Path(vocabulary_path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"vocabulary file does not exist: {path}")
    return path


class LanguageModel:
    """Expose TextSlinger predictions through the existing OneClick contract."""

    def __init__(
        self,
        model: Any,
        *,
        backend: str = CAUSAL_SUBWORD_BACKEND,
        word_list: WordList | None = None,
        recognizer_nbest: int = DEFAULT_RECOGNIZER_NBEST,
        word_search: dict[str, Any] | None = None,
        character_cache_size: int = DEFAULT_CHARACTER_RESULT_CACHE_SIZE,
        word_cache_size: int = DEFAULT_WORD_RESULT_CACHE_SIZE,
    ):
        if model is None:
            raise ValueError("a loaded TextSlinger model is required")
        if backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"unsupported TextSlinger backend {backend!r}; "
                f"expected one of {SUPPORTED_BACKENDS}"
            )
        if recognizer_nbest < config.num_prefix_fetch + config.num_best_fetch:
            raise ValueError(
                "recognizer_nbest must be large enough for both prediction pools"
            )
        self.model = model
        self.backend = backend
        self.word_list = word_list
        self.key_chars = tuple(kconfig.key_chars)
        self.recognizer_nbest = int(recognizer_nbest)
        self.character_cache_size = max(0, int(character_cache_size))
        self.word_cache_size = max(0, int(word_cache_size))
        self._character_cache = OrderedDict()
        self._word_cache = OrderedDict()
        self.character_cache_hits = 0
        self.character_cache_misses = 0
        self.word_cache_hits = 0
        self.word_cache_misses = 0
        if backend == NGRAM_BACKEND:
            search_values = {**DEFAULT_NGRAM_WORD_SEARCH, **(word_search or {})}
            self.word_search_config = ConfigPredictWordsNGram(**search_values)
            self.character_search_config = ConfigPredictCharactersNGram()
        else:
            search_values = {
                **DEFAULT_SUBWORD_WORD_SEARCH,
                **(word_search or {}),
            }
            self.word_search_config = ConfigPredictWordsSubword(**search_values)
            self.character_search_config = ConfigPredictCharactersSubword()
        self.word_search_values = search_values
        self.input_channel = InputChannelConfig(
            channel_insertion_probability=0.0,
            channel_deletion_probability=0.0,
            uniform_mixture_probability=0.0,
            complete_endpoint_probability=0.5,
        )

    @staticmethod
    def _cache_get(cache: OrderedDict, key: Any) -> Any:
        try:
            value = cache.pop(key)
        except KeyError:
            return None
        cache[key] = value
        return value

    @staticmethod
    def _cache_put(
        cache: OrderedDict,
        key: Any,
        value: Any,
        max_size: int,
    ) -> None:
        if max_size <= 0:
            return
        cache.pop(key, None)
        cache[key] = value
        if len(cache) > max_size:
            cache.popitem(last=False)

    def result_cache_stats(self) -> dict[str, dict[str, int]]:
        """Return per-adapter result-cache occupancy and hit counters."""
        return {
            "character": {
                "max_entries": self.character_cache_size,
                "entries": len(self._character_cache),
                "hits": self.character_cache_hits,
                "misses": self.character_cache_misses,
            },
            "word": {
                "max_entries": self.word_cache_size,
                "entries": len(self._word_cache),
                "hits": self.word_cache_hits,
                "misses": self.word_cache_misses,
            },
        }

    def get_key_probs(self, context: str) -> list[float]:
        """Return normalized next-character log masses in QuickClick key order."""
        uniform = [-math.log(len(self.key_chars))] * len(self.key_chars)
        if not context:
            return uniform

        cached = self._cache_get(self._character_cache, context)
        if cached is not None:
            self.character_cache_hits += 1
            return list(cached)
        self.character_cache_misses += 1

        result = self.model.predict_characters(
            left_context=context,
            output_characters=self.key_chars,
            config=self.character_search_config,
            predict_lower=True,
        )
        raw = {}
        for prediction in result.predictions:
            character = getattr(prediction, "character", None)
            if character in self.key_chars and character not in raw:
                raw[character] = max(
                    _prediction_log_mass(prediction), LOG_CLAMP_MIN
                )
        values = [raw.get(character, LOG_CLAMP_MIN) for character in self.key_chars]
        log_normalizer = _log_add_exp(values)
        if log_normalizer == float("-inf"):
            return uniform
        normalized = tuple(value - log_normalizer for value in values)
        self._cache_put(
            self._character_cache,
            context,
            normalized,
            self.character_cache_size,
        )
        return list(normalized)

    @staticmethod
    def _input_events(
        observations: Sequence[Sequence[float]],
    ) -> tuple[InputEvent, ...]:
        events = []
        for row_index, row in enumerate(observations):
            if len(row) != len(kconfig.key_chars):
                raise ValueError(
                    f"observation row {row_index} has {len(row)} alternatives; "
                    f"expected {len(kconfig.key_chars)}"
                )
            scores = tuple(float(value) for value in row)
            if any(
                math.isnan(value) or value == float("inf") for value in scores
            ):
                raise ValueError(
                    f"observation row {row_index} contains an invalid score"
                )
            events.append(
                InputEvent(
                    alternatives=tuple(zip(kconfig.key_chars, scores)),
                    # Insertions are disabled, so this required density is inert.
                    channel_insertion_log_density=0.0,
                )
            )
        return tuple(events)

    def get_word_predictions(
        self,
        left_context: str,
        observations: Sequence[Sequence[float]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Return autocomplete and exact-length words from one mixed search."""
        events = self._input_events(observations)
        observation_key = tuple(
            tuple(score for _, score in event.alternatives) for event in events
        )
        cache_key = (left_context, observation_key)
        cached = self._cache_get(self._word_cache, cache_key)
        if cached is not None:
            self.word_cache_hits += 1
            cached_prefix, cached_best = cached
            return (
                [
                    {"text": text, "logprob": logprob}
                    for text, logprob in cached_prefix
                ],
                [
                    {"text": text, "logprob": logprob}
                    for text, logprob in cached_best
                ],
            )
        self.word_cache_misses += 1

        result = self.model.predict_words(
            left_context=left_context,
            input_sequence=events,
            input_channel=self.input_channel,
            config=self.word_search_config,
            word_spelling_characters=WORD_SPELLING_CHARACTERS,
            word_list=self.word_list,
            nbest=self.recognizer_nbest,
            predict_lower=True,
        )

        event_count = len(events)
        prefix = []
        best = []
        seen = set()
        for prediction in result.predictions:
            word = getattr(prediction, "word", None)
            if not isinstance(word, str) or not word:
                continue
            normalized = word.lower()
            if normalized in seen:
                continue
            try:
                score = _prediction_log_mass(prediction)
            except (AttributeError, TypeError, ValueError):
                continue
            if len(word) > event_count and len(prefix) < config.num_prefix_fetch:
                prefix.append({"text": word, "logprob": score})
                seen.add(normalized)
            elif len(word) == event_count and len(best) < config.num_best_fetch:
                best.append({"text": word, "logprob": score})
                seen.add(normalized)
            if (
                len(prefix) == config.num_prefix_fetch
                and len(best) == config.num_best_fetch
            ):
                break
        cached_result = (
            tuple((item["text"], item["logprob"]) for item in prefix),
            tuple((item["text"], item["logprob"]) for item in best),
        )
        self._cache_put(
            self._word_cache,
            cache_key,
            cached_result,
            self.word_cache_size,
        )
        return prefix, best


def load_local_language_model(
    model_path: str | Path,
    *,
    backend: str = CAUSAL_SUBWORD_BACKEND,
    vocabulary_path: str | Path | None = None,
    device: str = "mps",
    precision: str = "fp32",
    recognizer_nbest: int = DEFAULT_RECOGNIZER_NBEST,
) -> LanguageModel:
    """Load one local TextSlinger backend and wrap it for QuickClick."""
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"unsupported TextSlinger backend {backend!r}; "
            f"expected one of {SUPPORTED_BACKENDS}"
        )
    if recognizer_nbest < config.num_prefix_fetch + config.num_best_fetch:
        raise ValueError(
            "recognizer_nbest must be large enough for both prediction pools"
        )

    if backend == NGRAM_BACKEND:
        path = validate_ngram_model_file(model_path)
        if vocabulary_path is None:
            raise ValueError("vocabulary_path is required for the n-gram backend")
        resolved_vocabulary_path = validate_vocabulary_file(vocabulary_path)
        word_list = WordList.from_file(str(resolved_vocabulary_path))
        try:
            model = NGramLanguageModel(
                lm_path=str(path),
                model_alphabet=NGRAM_MODEL_ALPHABET,
                space_character="<sp>",
            )
        except (OSError, RuntimeError) as error:
            if "KENLM_MAX_ORDER" in str(error) or "order 12" in str(error):
                raise RuntimeError(
                    "the bundled character n-gram is order 12, but the "
                    "installed KenLM build does not support that order; "
                    "rebuild KenLM with KENLM_MAX_ORDER=12"
                ) from error
            raise
        adapter = LanguageModel(
            model,
            backend=backend,
            word_list=word_list,
            recognizer_nbest=recognizer_nbest,
        )
        adapter.model_path = path
        adapter.vocabulary_path = resolved_vocabulary_path
        adapter.requested_device = "cpu"
        adapter.resolved_device = "cpu"
        adapter.precision = None
        adapter.quantization = None
        return adapter

    path = validate_model_directory(model_path)
    requested_device = Device(device)
    requested_precision = Precision(precision)
    import torch

    if requested_device == Device.MPS and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable in this PyTorch environment")
    if requested_device == Device.CUDA and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this PyTorch environment")
    model = CausalSubwordLanguageModel(
        lang_model_name=str(path),
        device=requested_device,
        precision=requested_precision,
        quantization=ModelQuantization.NONE,
    )
    word_list = None
    resolved_vocabulary_path = None
    if vocabulary_path is not None:
        resolved_vocabulary_path = validate_vocabulary_file(vocabulary_path)
        word_list = WordList.from_file(str(resolved_vocabulary_path))
    adapter = LanguageModel(
        model,
        backend=backend,
        word_list=word_list,
        recognizer_nbest=recognizer_nbest,
    )
    adapter.model_path = path
    adapter.vocabulary_path = resolved_vocabulary_path
    adapter.requested_device = requested_device.value
    adapter.resolved_device = str(getattr(model, "device", requested_device.value))
    adapter.precision = requested_precision.value
    adapter.quantization = "none"
    return adapter


def _sha256_file(path: Path | None) -> str | None:
    """Return a stable artifact hash without loading the complete file at once."""
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit_for(path: Path) -> str | None:
    for candidate in (path, *path.parents):
        if (candidate / ".git").exists():
            try:
                return subprocess.run(
                    ["git", "-C", str(candidate), "rev-parse", "HEAD"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
            except (OSError, subprocess.CalledProcessError):
                return None
    return None


def language_model_metadata(language_model: LanguageModel) -> dict[str, Any]:
    """Build the reproducibility record written beside study outputs."""
    try:
        import textslinger

        source_path = Path(textslinger.__file__).resolve().parent
    except (ImportError, TypeError):
        source_path = None
    try:
        version = importlib.metadata.version("textslinger")
    except importlib.metadata.PackageNotFoundError:
        version = None

    raw_model_path = getattr(language_model, "model_path", None)
    model_path = Path(raw_model_path).resolve() if raw_model_path else None
    model_config = {}
    config_path = (
        model_path / "config.json"
        if model_path is not None and model_path.is_dir()
        else None
    )
    if config_path and config_path.is_file():
        model_config = json.loads(config_path.read_text(encoding="utf-8"))
    raw_vocabulary_path = getattr(language_model, "vocabulary_path", None)
    vocabulary_path = (
        Path(raw_vocabulary_path).resolve() if raw_vocabulary_path else None
    )
    model_family = getattr(
        language_model,
        "backend",
        CAUSAL_SUBWORD_BACKEND,
    )
    return {
        "backend": "textslinger",
        "model_family": model_family,
        "model_source": (
            "local_file" if model_path and model_path.is_file() else "local_directory"
        ),
        "network_access": "disabled",
        "textslinger_version": version,
        "textslinger_source_path": str(source_path) if source_path else None,
        "textslinger_git_commit": (
            _git_commit_for(source_path) if source_path else None
        ),
        "model_path": str(model_path) if model_path else None,
        "model_sha256": _sha256_file(model_path),
        "vocabulary_path": (
            str(vocabulary_path) if vocabulary_path is not None else None
        ),
        "vocabulary_sha256": _sha256_file(vocabulary_path),
        "model_alphabet": (
            list(NGRAM_MODEL_ALPHABET) if model_family == NGRAM_BACKEND else None
        ),
        "model_name_or_path": model_config.get("_name_or_path"),
        "hugging_face_revision": model_config.get("_commit_hash"),
        "requested_device": getattr(language_model, "requested_device", None),
        "resolved_device": getattr(language_model, "resolved_device", None),
        "precision": getattr(language_model, "precision", None),
        "quantization": getattr(language_model, "quantization", None),
        "recognizer_nbest": getattr(language_model, "recognizer_nbest", None),
        "character_search": (
            asdict(language_model.character_search_config)
            if hasattr(language_model, "character_search_config")
            else None
        ),
        "word_search": dict(getattr(language_model, "word_search_values", {})),
        "model_cache": {
            "min_prefix": getattr(
                getattr(language_model, "model", None), "cache_min_prefix", None
            ),
            "max_entries": getattr(
                getattr(language_model, "model", None), "cache_max_entries", None
            ),
            "max_bytes": getattr(
                getattr(language_model, "model", None), "cache_max_bytes", None
            ),
        },
        "result_cache": (
            language_model.result_cache_stats()
            if hasattr(language_model, "result_cache_stats")
            else {
                "character": {
                    "max_entries": 0,
                    "entries": 0,
                    "hits": 0,
                    "misses": 0,
                },
                "word": {
                    "max_entries": 0,
                    "entries": 0,
                    "hits": 0,
                    "misses": 0,
                },
            }
        ),
        "input_channel": {
            "channel_insertion_probability": 0.0,
            "channel_deletion_probability": 0.0,
            "uniform_mixture_probability": 0.0,
            "complete_endpoint_probability": 0.5,
        },
        "word_search_policy": "mixed",
        "prefix_limit": config.num_prefix_fetch,
        "best_limit": config.num_best_fetch,
    }
