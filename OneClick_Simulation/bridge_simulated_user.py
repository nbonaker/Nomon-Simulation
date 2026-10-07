"""Opt-in shared-engine pilot. No imports from the legacy Python keyboard."""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import heapq
import json
import math
from typing import Protocol

from OneClick_Bridge import OneClickEngineBridge, BridgeError

ALPHABET = "abcdefghijklmnopqrstuvwxyz'"


class PredictionBackend(Protocol):
    def predict(self, effect: dict) -> dict:
        """Return raw service response data; no target phrase is supplied."""


class MissingPredictionFixture(ValueError):
    pass


class FixedPredictionBackend:
    """Explicit fixtures, keyed by (committed context, observation count)."""

    def __init__(self, character_response, word_responses):
        from .bridge_predictions import FixedCharacterTransitions
        self.characters = FixedCharacterTransitions(character_response)
        self.character_response = copy.deepcopy(character_response)
        self.word_responses = copy.deepcopy(word_responses)
        # Validate and fingerprint fixtures before a worker is allocated. Keeping
        # the digest avoids making error reporting depend on reserializing inputs.
        self._fixture_sha256 = snapshot_digest({
            "characters": self.character_response,
            "words": [[left, count, data] for (left, count), data in sorted(self.word_responses.items())]})

    @classmethod
    def from_dict(cls, fixture):
        words = {}
        for item in fixture["words"]:
            key = (item["left"], item["observation_count"])
            if key in words:
                raise ValueError(f"Duplicate prediction fixture: {key!r}")
            words[key] = item["data"]
        return cls(fixture["characters"], words)

    def metadata(self):
        return {"backend": "fixed", "fixture_sha256": self._fixture_sha256}

    def predict(self, effect):
        if effect["type"] == "predict-characters":
            return self.characters.predict(effect)
        if effect["type"] == "predict-words":
            key = (effect["left"], len(effect["distribs"]))
            if key not in self.word_responses:
                raise MissingPredictionFixture(f"No word fixture for context/count {key!r}")
            return copy.deepcopy(self.word_responses[key])
        raise ValueError(f"Unsupported prediction effect: {effect['type']}")


@dataclass(frozen=True)
class ClickSample:
    offset_s: float = 0.0
    dead_time_s: float = 0.0
    source_period_s: float | None = None

    def validate(self):
        if not math.isfinite(self.offset_s) or not math.isfinite(self.dead_time_s):
            raise ValueError("Click offset and dead time must be finite")
        if self.dead_time_s < 0:
            raise ValueError("Dead time must not be negative")
        if self.source_period_s is not None and (not math.isfinite(self.source_period_s) or self.source_period_s <= 0):
            raise ValueError("Recorded source period must be positive and finite")
        return self


class ZeroNoiseClickSource:
    def sample(self):
        return ClickSample()


class SequenceClickSource:
    """Consume ClickSample objects or recorded CSV-style rows exactly once, in order."""

    def __init__(self, samples):
        self._samples = iter(samples)

    def sample(self):
        item = next(self._samples, None)
        if item is None or isinstance(item, ClickSample):
            return item
        # Preserve recorded missing dead time as zero; explicit infinities/negative
        # times still fail. Recorded period is provenance, never engine configuration.
        dead = item.get("Dead Time (s)")
        dead = 0.0 if dead is None or dead == "" else float(dead)
        period = item.get("Clock Period (s)")
        period = None if period is None or period == "" else float(period)
        return ClickSample(float(item["Click Time Relative (s)"]), dead, period)


def normalize_target(phrase):
    if not isinstance(phrase, str):
        raise ValueError("Target phrase must be text")
    normalized = " ".join(phrase.lower().split())
    if not normalized or any(char not in ALPHABET + " " for char in normalized):
        raise ValueError("Target must contain only words made of a-z and apostrophe")
    return normalized


def snapshot_digest(snapshot):
    encoded = json.dumps(snapshot, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class PilotStop(Exception):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


class FrameScheduler:
    """Ideal animation callbacks, explicit simulated time, and one-time press aiming."""

    def __init__(self, initial_snapshot, dispatch, start_time=0.0):
        self.origin = start_time
        self.now = start_time
        self.frame_index = 0
        self.dispatch = dispatch
        letters, words = initial_snapshot["letters"], initial_snapshot["words"]
        self.interval = letters["period"] / letters["num_divs_time"]
        word_interval = words["period"] / words["num_divs_time"]
        if not math.isclose(self.interval, word_interval, abs_tol=1e-12, rel_tol=0):
            raise ValueError("Pilot requires a shared letter/word animation interval")

    @staticmethod
    def press_time(now, group, clock_id, sample):
        sample.validate()
        period = group["period"]
        noon = group["latest_time"] + (0.5 - group["phases"][clock_id] / group["num_divs_time"]) * period
        if noon < now:
            noon += math.ceil((now - noon) / period) * period
        press = noon + sample.offset_s
        earliest = now + 0.001
        if press < earliest:
            press += math.ceil((earliest - press) / period) * period
        # Match the legacy fixed-period whole-rotation component, without changing speed.
        dead_rotations = (sample.dead_time_s // period) * period
        press += dead_rotations
        if not math.isfinite(press) or press < earliest - 1e-12:
            raise ValueError("Calculated press time is invalid")
        return press, dead_rotations

    def advance_to(self, target):
        # Normalize only roundoff at exact frame boundaries. The press is not
        # re-aimed after frames; sampled error must survive into the engine.
        boundary = self.origin + round((target - self.origin) / self.interval) * self.interval
        if abs(target - boundary) < 1e-12:
            target = boundary
        if target < self.now:
            raise ValueError("Simulated time cannot move backwards")
        while self.origin + (self.frame_index + 1) * self.interval <= target:
            self.frame_index += 1
            self.now = self.origin + self.frame_index * self.interval
            self.dispatch({"type": "frame", "group": "letters", "time": self.now})
            self.dispatch({"type": "frame", "group": "words", "time": self.now})
        self.now = target


class BridgeSimulatedUser:
    """One fresh engine per phrase, actual browser selections and correction policy."""

    def __init__(self, engine_repo=None, node_executable="node", rotate_index=5,
                 max_attempts_per_word=4, max_presses_per_word=30, max_presses_per_phrase=200,
                 timeout_s=10, character_transition_source="backend",
                 fixed_character_responses=None, character_prediction_latency_s=0.0,
                 word_prediction_latency_s=0.0):
        for name, limit in (("max_attempts_per_word", max_attempts_per_word),
                            ("max_presses_per_word", max_presses_per_word),
                            ("max_presses_per_phrase", max_presses_per_phrase)):
            if type(limit) is not int or limit < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(rotate_index) is not int or not 0 <= rotate_index <= 20:
            raise ValueError("rotate_index must be an integer from 0 to 20")
        if character_transition_source not in ("backend", "fixed"):
            raise ValueError("character_transition_source must be backend or fixed")
        for latency in (character_prediction_latency_s, word_prediction_latency_s):
            if isinstance(latency, bool) or not isinstance(latency, (float, int)) or not math.isfinite(latency) or latency < 0:
                raise ValueError("Prediction latency must be finite and nonnegative")
        self.character_source = character_transition_source
        self.fixed_characters = None
        if character_transition_source == "fixed":
            from .bridge_predictions import FixedCharacterTransitions
            self.fixed_characters = FixedCharacterTransitions(fixed_character_responses)
        elif fixed_character_responses is not None:
            raise ValueError("fixed_character_responses requires character_transition_source='fixed'")
        self.latencies = {"predict-characters": float(character_prediction_latency_s),
                          "predict-words": float(word_prediction_latency_s)}
        self.bridge_options = {"engine_repo": engine_repo, "node_executable": node_executable, "timeout_s": timeout_s}
        self.config = {"rotateIndex": rotate_index}
        self.max_attempts = max_attempts_per_word
        self.max_word_presses = max_presses_per_word
        self.max_phrase_presses = max_presses_per_phrase

    def run_phrase(self, target_phrase, click_source, prediction_backend):
        # Invalid targets are rejected before allocating a worker.
        self.target = normalize_target(target_phrase)
        self.words = self.target.split()
        self.prefixes = [""] + [" ".join(self.words[:i]) + " " for i in range(1, len(self.words) + 1)]
        self.source, self.backend = click_source, prediction_backend
        self.state = None
        self.scheduler = None
        self.now = 0.0
        self.pending = []
        self.requests = []
        self.character_responses = {}
        self.startup_elapsed = None
        self.prediction_wait = 0.0
        self.typing_prediction_wait = 0.0
        self.current_word = 0
        self.attempts = [0] * len(self.words)
        self.presses_by_word = [0] * len(self.words)
        self.counts = {"presses": 0, "space": 0, "enter": 0, "wrong_commits": 0, "undo_attempts": 0}
        self.trace = {"version": 1, "engine_config": copy.deepcopy(self.config), "engine_metadata": None, "events": []}
        self.uncertain = False
        self.failure_detail = None
        failure = None
        error = None
        try:
            with OneClickEngineBridge(config=self.config, **self.bridge_options) as self.bridge:
                self.trace["engine_metadata"] = self.bridge.metadata
                self._dispatch({"type": "initialize", "time": 0.0})
                self._drain_predictions()
                self.startup_elapsed = self.now
                if not self.state["ready"]:
                    raise PilotStop("engine_not_ready")
                self._type_phrase()
        except PilotStop as exc:
            failure = exc.reason
        except BridgeError as exc:
            failure = "bridge_failure"
            error = {"type": type(exc).__name__, "message": str(exc)}
        except Exception as exc:
            # Context manager has already reaped the worker. Retain successful
            # prefix events and diagnostics rather than reset or retry the engine.
            failure = "pilot_error"
            error = {"type": type(exc).__name__, "message": str(exc)}
        final_text = self.state["typed"] if self.state else ""
        matrix = copy.deepcopy(self.state["transition_matrix"]) if self.state else None
        prediction_metadata = {
            "backend": copy.deepcopy(self.backend.metadata()) if hasattr(self.backend, "metadata") else {"backend": type(self.backend).__name__},
            "character_transition_source": self.character_source,
            "character_responses": copy.deepcopy(self.character_responses),
            "effective_transition_matrix": matrix,
            "effective_transition_matrix_sha256": snapshot_digest(matrix) if matrix is not None else None,
            "character_prediction_latency_s": self.latencies["predict-characters"],
            "word_prediction_latency_s": self.latencies["predict-words"],
            "wait_policy": "wait_for_predictions",
            "dropped_negative_infinity_candidates": sum(r.get("dropped_negative_infinity_candidates", 0) for r in self.requests),
        }
        self.trace["prediction_metadata"] = copy.deepcopy(prediction_metadata)
        self.trace["prediction_requests"] = self.requests
        return {
            "target": self.target, "final_text": final_text,
            "completed": failure is None and final_text == self.prefixes[-1],
            "failure_reason": failure, "error": error or getattr(self, "failure_detail", None),
            "simulated_elapsed_s": self.now,
            "startup_elapsed_s": self.startup_elapsed if self.startup_elapsed is not None else self.now,
            "typing_elapsed_s": self.now - self.startup_elapsed if self.startup_elapsed is not None else 0.0,
            "prediction_wait_s": self.prediction_wait, "typing_prediction_wait_s": self.typing_prediction_wait,
            "prediction_metadata": prediction_metadata, "counts": dict(self.counts),
            "attempts_by_word": list(self.attempts), "presses_by_word": list(self.presses_by_word),
            "engine_metadata": self.trace["engine_metadata"], "final_text_known": self.state is not None and not self.uncertain,
            "final_snapshot": copy.deepcopy(self.state), "trace": self.trace,
        }

    def _dispatch(self, event, intent=None):
        self.now = event.get("time", self.now)
        record = {"event": copy.deepcopy(event), "intent": copy.deepcopy(intent), "status": "pending"}
        self.trace["events"].append(record)
        try:
            result = self.bridge.dispatch(event)
        except Exception as exc:
            record.update(status="error", error={"type": type(exc).__name__, "message": str(exc)})
            self.uncertain = True  # Transport failure may follow a state-changing event.
            raise
        self.state = result["snapshot"]
        record.update(status="ok", snapshot_sha256=snapshot_digest(self.state), effects=copy.deepcopy(result["effects"]),
                      observed={"typed": self.state["typed"], "observation_count": len(self.state["observations"]),
                                "selection": self.state["last_selection"] if event["type"] == "enter" else None})
        if self.state["ready"] and self.scheduler is None:
            self.scheduler = FrameScheduler(self.state, self._dispatch, start_time=self.now)
        if event["type"] == "enter":
            if self.state["last_selection"]["id"] != self._undo_id() and self._prefix_position() is None:
                self.counts["wrong_commits"] += 1
        for effect in result["effects"]:
            self._queue_prediction(effect)
        return result

    def _queue_prediction(self, effect):
        request = {"id": len(self.requests), "effect": copy.deepcopy(effect),
                   "requested_time": self.now, "due_time": self.now + self.latencies[effect["type"]],
                   "status": "pending"}
        self.requests.append(request)
        try:
            if not math.isfinite(request["due_time"]):
                raise ValueError("Prediction due time must be finite")
            backend = self.fixed_characters if effect["type"] == "predict-characters" and self.fixed_characters else self.backend
            dropped_before = getattr(backend, "dropped_negative_infinity", 0)
            data = backend.predict(copy.deepcopy(effect))
            request["dropped_negative_infinity_candidates"] = getattr(backend, "dropped_negative_infinity", 0) - dropped_before
            # Diagnose unusable backend payloads before recording an engine event.
            json.dumps(data, allow_nan=False)
            if effect["type"] == "predict-characters":
                from .bridge_predictions import validate_character_response
                validate_character_response(data)
                response = {"type": "character-response", "index": effect["index"], "data": copy.deepcopy(data)}
                self.character_responses[effect["left"]] = copy.deepcopy(data)
            elif effect["type"] == "predict-words":
                response = {"type": "prediction-response", "data": copy.deepcopy(data)}
            else:
                raise ValueError(f"Unknown effect: {effect['type']}")
            heapq.heappush(self.pending, (request["due_time"], request["id"], response))
        except Exception as exc:
            request.update(status="error", error={"type": type(exc).__name__, "message": str(exc)})
            self.failure_detail = {**request["error"], "effect": copy.deepcopy(effect)}
            raise PilotStop("prediction_failure") from exc

    def _drain_predictions(self):
        # The entire batch shares its request timestamp. Equal arrivals retain
        # effect order; initialized clocks run before a response at a frame tie.
        while self.pending:
            due, request_id, response = heapq.heappop(self.pending)
            before = self.now
            if self.scheduler is not None:
                self.scheduler.advance_to(due)
                due = self.scheduler.now
            self.now = due
            elapsed = self.now - before
            self.prediction_wait += elapsed
            if self.startup_elapsed is not None:
                self.typing_prediction_wait += elapsed
            response["time"] = self.now
            request = self.requests[request_id]
            try:
                self._dispatch(response)
            except Exception:
                request["status"] = "dispatch_error"
                raise
            request.update(status="delivered", delivered_time=self.now)

    def _prefix_position(self):
        try:
            return self.prefixes.index(self.state["typed"])
        except ValueError:
            return None

    def _undo_id(self):
        # The last logical slot is the command clock. A prediction may itself be
        # the word "Undo", so labels alone must not identify the command.
        return self.state["words"]["clocks"][-1]["id"]

    def _matching_word(self, target):
        clocks = self.state["words"]["clocks"]
        return next((i for i in self.state["valid_word_indices"]
                     if i != self._undo_id() and clocks[i]["label"].lower() == target), None)

    def _press(self, group, clock_id, action, target):
        if self.counts["presses"] >= self.max_phrase_presses:
            raise PilotStop("phrase_press_budget")
        if self.presses_by_word[self.current_word] >= self.max_word_presses:
            raise PilotStop("word_press_budget")
        try:
            sample = self.source.sample()
            if sample is not None:
                sample.validate()
        except Exception as exc:
            self.failure_detail = {"type": type(exc).__name__, "message": str(exc)}
            raise PilotStop("invalid_click_sample") from exc
        if sample is None:
            raise PilotStop("click_source_exhausted")
        if not any(clock["id"] == clock_id and clock["active"] for clock in self.state[group]["clocks"]):
            raise PilotStop("target_clock_unavailable")
        time, dead_rotations = self.scheduler.press_time(self.scheduler.now, self.state[group], clock_id, sample)
        self.scheduler.advance_to(time)
        kind = "space" if group == "letters" else "enter"
        self.counts["presses"] += 1
        self.counts[kind] += 1
        self.presses_by_word[self.current_word] += 1
        if action == "undo":
            self.counts["undo_attempts"] += 1
        intent = {"action": action, "target": target, "clock_id": clock_id, "word_position": self.current_word,
                  "offset_s": sample.offset_s, "dead_time_s": sample.dead_time_s,
                  "dead_rotations_s": dead_rotations, "source_period_s": sample.source_period_s}
        self._dispatch({"type": kind, "time": self.scheduler.now}, intent)
        self._drain_predictions()

    def _type_phrase(self):
        while True:
            position = self._prefix_position()
            if position == len(self.words):
                return
            if position is None:
                # Keep charging cascading correction to the original attempted word.
                self._press("words", self._undo_id(), "undo", "Undo")
                continue
            self.current_word = position
            if self.attempts[position] >= self.max_attempts:
                raise PilotStop("word_attempt_budget")
            self.attempts[position] += 1
            target = self.words[position]
            selected = None
            for letter in target:
                clock_id = next(clock["id"] for clock in self.state["letters"]["clocks"] if clock["label"] == letter)
                self._press("letters", clock_id, "letter", letter)
                selected = self._matching_word(target)
                if selected is not None:
                    break
            if selected is None:
                # The literal immediately precedes Undo in the engine's stable IDs.
                # Resolve it through active clocks; never synthesize its text.
                selected = self._undo_id() - 1
            self._press("words", selected, "target_enter", target)
            # Enter (including midword Undo) clears observations. Re-evaluate actual
            # text, then start a fresh attempt if needed; never restore a snapshot.


def replay_trace(trace, engine_repo=None, node_executable="node"):
    """Verify successful prefix events; a failed/uncertain dispatch is never retried."""
    if trace.get("version") != 1 or not trace.get("engine_metadata"):
        raise ValueError("Trace needs version 1 and engine source metadata")
    with OneClickEngineBridge(engine_repo=engine_repo, node_executable=node_executable,
                              config=trace["engine_config"]) as bridge:
        if bridge.metadata["sourceSha256"] != trace["engine_metadata"]["sourceSha256"]:
            raise ValueError("Trace engine fingerprint differs from the current checkout")
        count = 0
        for record in trace["events"]:
            if record["status"] != "ok":
                return {"verified_events": count, "complete_trace": False, "snapshot": bridge.get_snapshot()}
            result = bridge.dispatch(record["event"])
            if snapshot_digest(result["snapshot"]) != record["snapshot_sha256"] or result["effects"] != record["effects"]:
                raise AssertionError(f"Replay mismatch at event {count}: {record['event']['type']}")
            count += 1
        return {"verified_events": count, "complete_trace": True, "snapshot": bridge.get_snapshot()}
