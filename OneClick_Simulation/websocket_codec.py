"""DOM/network-free counterpart of the browser's prediction-codec.mjs."""
from __future__ import annotations

import json
import math

ALPHABET = "abcdefghijklmnopqrstuvwxyz'"
CHARACTER_FLOOR = math.log(0.01)


def strict_loads(text):
    def invalid(value):
        raise ValueError(f"Non-JSON number: {value}")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    result = json.loads(text, parse_constant=invalid, object_pairs_hook=unique)

    def finite(value):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("JSON numeric overflow")
        if isinstance(value, dict):
            for item in value.values():
                finite(item)
        elif isinstance(value, list):
            for item in value:
                finite(item)

    finite(result)
    return result


def _score(value):
    if value == "-inf":
        return -math.inf
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Score must be a finite number or "-inf"')
    return value


def prediction_request(effect, *, session_id, request_id, revision, model="default", vocabulary_profile=None):
    if not isinstance(effect.get("left"), str):
        raise ValueError("Prediction context must be text")
    request = dict(version=1, session_id=session_id, request_id=request_id, revision=revision,
                   model=model, replaceable=False, deadline_ms=10000, left_context=effect["left"],
                   predict_lower=True, word_spelling_characters=list(ALPHABET), word_ending_characters=[" "],
                   nbest=1000, input_channel=dict(channel_insertion_probability=0.0,
                       channel_deletion_probability=0.0, uniform_mixture_probability=0.0, complete_prior=0.5))
    if vocabulary_profile is not None:
        request["vocabulary_profile"] = vocabulary_profile
    if effect.get("type") == "predict-characters":
        index = effect.get("index")
        if type(index) is not int or not 0 <= index < len(ALPHABET) or ALPHABET[index] != effect["left"]:
            raise ValueError("Invalid character transition effect")
        return dict(request, operation="characters", lane=f"characters-{index}", output_characters=list(ALPHABET))
    if effect.get("type") != "predict-words":
        raise ValueError("Unsupported prediction effect")
    if any(type(effect.get(k)) is not int or effect[k] < 0 for k in ("numPrefix", "numBest")):
        raise ValueError("Prediction pool limits must be nonnegative integers")
    if not isinstance(effect.get("distribs"), list):
        raise ValueError("Observation sequence must be an array")
    events = []
    for row in effect["distribs"]:
        entries = row.get("distrib")
        if not isinstance(entries, list) or len(entries) != len(ALPHABET):
            raise ValueError("Observation requires 27 symbols")
        values = {}
        for item in entries:
            char, score = item.get("text"), item.get("logProb")
            if not isinstance(char, str) or len(char) != 1 or char not in ALPHABET or char in values:
                raise ValueError("Invalid or duplicate observation symbol")
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                raise ValueError("Observation score must be finite")
            values[char] = score
        events.append(dict(alternatives=[dict(character=c, log_likelihood=values[c]) for c in ALPHABET],
                           channel_insertion_log_density=0.0))
    return dict(request, operation="words", lane="words", input_events=events)


def prediction_response(effect, request, response):
    if (not isinstance(response, dict) or type(response.get("version")) is not int
            or type(response.get("revision")) is not int or response.get("version") != 1
            or any(response.get(k) != request[k] for k in ("request_id", "session_id", "revision", "operation", "model"))):
        raise ValueError("Mismatched TextSlinger response identity or protocol version")
    if response.get("status") != "ok":
        raise ValueError(f"TextSlinger {response.get('status')}: {response.get('error')}")
    if effect["type"] == "predict-characters":
        predictions = response["data"]["characters"]["predictions"]
        if not isinstance(predictions, list) or len(predictions) != len(ALPHABET):
            raise ValueError("Character response requires 27 symbols")
        results, seen = [], set()
        for item in predictions:
            char = item.get("character")
            if not isinstance(char, str) or len(char) != 1 or char not in ALPHABET or char in seen:
                raise ValueError("Invalid or duplicate character response symbol")
            seen.add(char)
            score = _score(item["probability"]["log_retained_joint_mass"])
            results.append(dict(token=char, logProb=CHARACTER_FLOOR if score == -math.inf else score))
        return dict(type="character-response", index=effect["index"], data=dict(results=results))
    predictions = response["data"]["words"]["predictions"]
    if not isinstance(predictions, list):
        raise ValueError("Word response requires a predictions array")
    ranked = []
    for item in predictions:
        if not isinstance(item.get("word"), str) or not item["word"]:
            raise ValueError("Word prediction requires nonempty text")
        score = _score(item["probability"]["log_retained_joint_mass"])
        if score != -math.inf:
            ranked.append(dict(text=item["word"], logprob=score))
    ranked.sort(key=lambda item: -item["logprob"])
    data, seen = dict(prefix=[], best=[]), set()
    n = len(effect["distribs"])
    for item in ranked:
        key, length = item["text"].lower(), len(item["text"])
        category = "prefix" if length > n else "best" if length == n else None
        if key in seen or category is None:
            continue
        limit = effect["numPrefix"] if category == "prefix" else effect["numBest"]
        if len(data[category]) < limit:
            data[category].append(item)
            seen.add(key)
    return dict(type="prediction-response", data=data)
