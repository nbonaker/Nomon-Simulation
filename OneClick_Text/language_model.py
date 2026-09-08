"""Local TextSlinger adapter for the OneClick/QuickClick simulator."""

from __future__ import annotations

import importlib.metadata
import json
import math
import os
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

from textslinger import InputChannelConfig, InputEvent
from textslinger.causal_subword import (
    CausalSubwordLanguageModel,
    ConfigPredictCharactersSubword,
    ConfigPredictWordsSubword,
)
from textslinger.helpers import Device, ModelQuantization, Precision


LOG_CLAMP_MIN = math.log(0.01)
DEFAULT_RECOGNIZER_NBEST = 1000
WORD_SPELLING_CHARACTERS = tuple(kconfig.key_chars)

DEFAULT_WORD_SEARCH = {
    "max_active_hypotheses": 60,
    "beam_best": 8.0,
    "max_terminal_token_paths": 12800,
    "max_omitted_prefix_paths": 65536,
    "max_omitted_joint_prefix_paths": 64,
    "text_state_remaining_mass_tolerance": 1e-3,
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


class LanguageModel:
    """Expose TextSlinger predictions through the existing OneClick contract."""

    def __init__(
        self,
        model: Any,
        *,
        recognizer_nbest: int = DEFAULT_RECOGNIZER_NBEST,
        word_search: dict[str, Any] | None = None,
    ):
        if model is None:
            raise ValueError("a loaded TextSlinger model is required")
        if recognizer_nbest < config.num_prefix_fetch + config.num_best_fetch:
            raise ValueError(
                "recognizer_nbest must be large enough for both prediction pools"
            )
        self.model = model
        self.key_chars = tuple(kconfig.key_chars)
        self.recognizer_nbest = int(recognizer_nbest)
        search_values = {**DEFAULT_WORD_SEARCH, **(word_search or {})}
        self.word_search_values = search_values
        self.word_search_config = ConfigPredictWordsSubword(**search_values)
        self.character_search_config = ConfigPredictCharactersSubword()
        self.input_channel = InputChannelConfig(
            channel_insertion_probability=0.0,
            channel_deletion_probability=0.0,
            uniform_mixture_probability=0.0,
            complete_endpoint_probability=0.5,
        )

    def get_key_probs(self, context: str) -> list[float]:
        """Return normalized next-character log masses in QuickClick key order."""
        uniform = [-math.log(len(self.key_chars))] * len(self.key_chars)
        if not context:
            return uniform

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
        return [value - log_normalizer for value in values]

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
        result = self.model.predict_words(
            left_context=left_context,
            input_sequence=events,
            input_channel=self.input_channel,
            config=self.word_search_config,
            word_spelling_characters=WORD_SPELLING_CHARACTERS,
            word_list=None,
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
        return prefix, best


def load_local_language_model(
    model_path: str | Path,
    *,
    device: str = "mps",
    precision: str = "fp32",
    recognizer_nbest: int = DEFAULT_RECOGNIZER_NBEST,
) -> LanguageModel:
    """Load one local TextSlinger model and wrap it for OneClick."""
    path = validate_model_directory(model_path)
    requested_device = Device(device)
    requested_precision = Precision(precision)
    if recognizer_nbest < config.num_prefix_fetch + config.num_best_fetch:
        raise ValueError(
            "recognizer_nbest must be large enough for both prediction pools"
        )
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
    adapter = LanguageModel(model, recognizer_nbest=recognizer_nbest)
    adapter.model_path = path
    adapter.requested_device = requested_device.value
    adapter.resolved_device = str(getattr(model, "device", requested_device.value))
    adapter.precision = requested_precision.value
    return adapter


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
    config_path = model_path / "config.json" if model_path else None
    if config_path and config_path.is_file():
        model_config = json.loads(config_path.read_text(encoding="utf-8"))
    return {
        "backend": "textslinger",
        "model_source": "local_directory",
        "network_access": "disabled",
        "textslinger_version": version,
        "textslinger_source_path": str(source_path) if source_path else None,
        "textslinger_git_commit": (
            _git_commit_for(source_path) if source_path else None
        ),
        "model_path": str(model_path) if model_path else None,
        "model_name_or_path": model_config.get("_name_or_path"),
        "hugging_face_revision": model_config.get("_commit_hash"),
        "requested_device": getattr(language_model, "requested_device", None),
        "resolved_device": getattr(language_model, "resolved_device", None),
        "precision": getattr(language_model, "precision", None),
        "quantization": "none",
        "recognizer_nbest": getattr(language_model, "recognizer_nbest", None),
        "character_search": asdict(
            getattr(
                language_model,
                "character_search_config",
                ConfigPredictCharactersSubword(),
            )
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
