"""Local adapters for shared-engine effects; no model import until load."""
from __future__ import annotations
import copy
import math
from pathlib import Path

ALPHABET = "abcdefghijklmnopqrstuvwxyz'"


def _finite_number(value, label):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return float(value)


def validate_character_response(data):
    """One finite score per engine symbol; normalization stays in JS."""
    if not isinstance(data, dict) or not isinstance(data.get('results'), list):
        raise ValueError('Character response requires a results list')
    seen = set()
    for item in data['results']:
        token = item.get('token')
        if not isinstance(token, str) or len(token) != 1 or token not in ALPHABET or token in seen:
            raise ValueError('Character response must contain each of the 27 symbols exactly once')
        seen.add(token)
        _finite_number(item['logProb'], 'Character score')
    if seen != set(ALPHABET):
        raise ValueError('Character response must contain each of the 27 symbols exactly once')


class FixedCharacterTransitions:
    """A shared response, or 27 responses keyed by previous character."""
    def __init__(self, responses):
        if not isinstance(responses, dict):
            raise ValueError('Fixed transitions require a shared response or 27 character-keyed responses')
        if 'results' in responses:
            validate_character_response(responses)
            self.responses = {c: copy.deepcopy(responses) for c in ALPHABET}
        else:
            if set(responses) != set(ALPHABET):
                raise ValueError('Fixed transition rows must match the 27-symbol alphabet')
            for row in responses.values():
                validate_character_response(row)
            self.responses = copy.deepcopy(responses)

    def predict(self, effect):
        if effect['type'] != 'predict-characters' or effect['left'] not in self.responses:
            raise ValueError('Expected a startup single-character transition request')
        return copy.deepcopy(self.responses[effect['left']])


class TextSlingerNGramBackend:
    """Translate engine requests to the local n-gram LanguageModel API.

    Use from_local() in production; injection keeps contract tests model-free.
    A loaded adapter can be reused sequentially across phrases. It receives no
    target text or intended actions.
    """
    def __init__(self, language_model, metadata=None):
        if tuple(language_model.key_chars) != tuple(ALPHABET):
            raise ValueError("Language model alphabet must match the engine's 27-symbol order")
        self.language_model = language_model
        self._metadata = copy.deepcopy(metadata or {'backend': 'textslinger', 'model_family': 'ngram'})
        self.dropped_negative_infinity = 0

    @classmethod
    def from_local(cls, model_path, vocabulary_path, *, recognizer_nbest=1000):
        for label, path in (('Model', model_path), ('Vocabulary', vocabulary_path)):
            if path is None or not Path(path).expanduser().is_file():
                raise ValueError(f'{label} file does not exist: {path}')
        try:
            from OneClick_Text.language_model import load_local_language_model, language_model_metadata
        except ImportError as exc:
            raise RuntimeError('TextSlinger n-gram dependencies are unavailable; use the configured simulator .venv and local TextSlinger installation') from exc
        model = load_local_language_model(model_path, backend='ngram', vocabulary_path=vocabulary_path,
                                          recognizer_nbest=recognizer_nbest)
        return cls(model, language_model_metadata(model))

    def metadata(self):
        return {**copy.deepcopy(self._metadata),
                'dropped_negative_infinity_candidates': self.dropped_negative_infinity,
                'candidate_count_scope': 'adapter_lifetime',
                'nonfinite_word_policy': 'drop_negative_infinity_fail_nan_positive_infinity', 'strict_scores': True}

    def predict(self, effect):
        if not isinstance(effect.get('left'), str):
            raise ValueError('Prediction context must be text')
        if effect['type'] == 'predict-characters':
            if len(effect['left']) != 1 or effect['left'] not in ALPHABET:
                raise ValueError('Expected a single-character transition context')
            scores = self.language_model.get_key_probs(effect['left'])
            if len(scores) != len(ALPHABET):
                raise ValueError('Character prediction must contain 27 scores')
            response = {'results': [{'token': c, 'logProb': _finite_number(score, 'Character score')}
                                     for c, score in zip(ALPHABET, scores)]}
            validate_character_response(response)
            return response
        if effect['type'] != 'predict-words':
            raise ValueError(f"Unsupported prediction effect: {effect['type']}")
        observations = []
        for index, row in enumerate(effect['distribs']):
            scores = {}
            for item in row['distrib']:
                char = item['text']
                if not isinstance(char, str) or len(char) != 1 or char not in ALPHABET or char in scores:
                    raise ValueError(f'Observation {index} must contain each engine symbol exactly once')
                scores[char] = _finite_number(item['logProb'], 'Observation score')
            if set(scores) != set(ALPHABET):
                raise ValueError(f'Observation {index} must contain 27 symbols')
            observations.append([scores[char] for char in ALPHABET])
        limits = [effect['numPrefix'], effect['numBest']]
        if any(type(limit) is not int or limit < 0 for limit in limits):
            raise ValueError('Prediction limits must be nonnegative integers')
        prefix, best = self.language_model.get_word_predictions(
            effect['left'], observations, prefix_limit=limits[0], best_limit=limits[1], strict_scores=True)
        result = {}
        for category, candidates, limit in zip(('prefix', 'best'), (prefix, best), limits):
            clean = []
            for item in candidates:
                score = item['logprob']
                if score == -math.inf:
                    self.dropped_negative_infinity += 1
                    continue
                score = _finite_number(score, 'Word score')
                if not isinstance(item['text'], str) or not item['text']:
                    raise ValueError('Word prediction must contain nonempty text')
                clean.append({'text': item['text'], 'logprob': score})
            result[category] = clean[:limit]
        return result
