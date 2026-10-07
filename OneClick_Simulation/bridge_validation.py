"""Audit saved pilot artifacts against a fresh shared engine, without a predictor."""
from __future__ import annotations

import heapq
import json
import math

from OneClick_Bridge import OneClickEngineBridge
from .bridge_simulated_user import ALPHABET, normalize_target, snapshot_digest

TOLERANCE = 1e-10


class _Mismatch(Exception):
    def __init__(self, field, expected, actual, event_index=None):
        self.detail = {"field": field, "expected": expected, "actual": actual, "event_index": event_index}
        super().__init__(field)


def _equal(field, actual, expected, index=None):
    if actual != expected or snapshot_digest(actual) != snapshot_digest(expected):
        raise _Mismatch(field, expected, actual, index)


def _time(field, actual, expected, index=None):
    if (isinstance(actual, bool) or not isinstance(actual, (int, float)) or not math.isfinite(actual)
            or not math.isclose(actual, expected, rel_tol=0, abs_tol=TOLERANCE)):
        raise _Mismatch(field, expected, actual, index)


def _duration(intervals):
    """Union of response-wait intervals: startup requests overlap."""
    total, end = 0.0, 0.0
    for start, stop in sorted(intervals):
        total += max(0.0, stop - max(start, end))
        end = max(end, stop)
    return total


def validate_run(result, trace, engine_repo=None, node_executable="node"):
    """Return a JSON report; valid means a complete successful run passed its audit.

    Failed runs retain verified prefix diagnostics, but cannot pass this full-run
    validator. Intent is checked for policy consistency; actual selections come
    only from replay. Metadata consistency is not proof of model provenance.
    """
    report = {"version": 1, "valid": False, "complete_trace": False,
              "verified_events": 0, "errors": [], "reconstructed": {}}
    index = None
    try:
        json.dumps([result, trace], allow_nan=False)
        _equal("trace.version", trace["version"], 1)
        target = normalize_target(result["target"])
        _equal("target", result["target"], target)
        words = target.split()
        prefixes = [""] + [" ".join(words[:i]) + " " for i in range(1, len(words) + 1)]
        _equal("engine_metadata", result["engine_metadata"], trace["engine_metadata"])
        meta = trace["prediction_metadata"]
        _equal("prediction_metadata", result["prediction_metadata"], meta)
        _equal("prediction_metadata.wait_policy", meta["wait_policy"], "wait_for_predictions")
        latencies = {"predict-characters": meta["character_prediction_latency_s"],
                     "predict-words": meta["word_prediction_latency_s"]}
        for kind, value in latencies.items():
            _time(f"latency.{kind}", value, value)
            if value < 0:
                raise _Mismatch(f"latency.{kind}", "nonnegative", value)
        if not trace["events"]:
            raise _Mismatch("events", "nonempty complete trace", [])
        requests = trace["prediction_requests"]
        counts = dict(presses=0, space=0, enter=0, wrong_commits=0, undo_attempts=0)
        per_word, attempts = [0] * len(words), [0] * len(words)
        report["reconstructed"] = {"counts": counts, "presses_by_word": per_word, "attempts_by_word": attempts}
        transitions, selections, actual_undos = [], [], 0
        pending, generated, intervals, typing_intervals = [], 0, [], []
        characters = {}
        startup = None
        previous_time = 0.0
        current_word = 0
        frame_origin, frame_step, frame_number, frame_group = None, None, 1, "letters"
        with OneClickEngineBridge(engine_repo=engine_repo, node_executable=node_executable,
                                  config=trace["engine_config"]) as engine:
            for field in ("sourceSha256", "protocolVersion", "sourceFiles"):
                _equal(f"engine_metadata.{field}", trace["engine_metadata"][field], engine.metadata[field])
            state = engine.get_snapshot()
            for index, record in enumerate(trace["events"]):
                _equal("event.status", record["status"], "ok", index)
                event = record["event"]
                kind, time = event["type"], event["time"]
                _time("event.time", time, time, index)
                if time < previous_time:
                    raise _Mismatch("event.time", f">= {previous_time}", time, index)
                previous_time = time
                if index == 0:
                    _equal("initial_event", event, {"type": "initialize", "time": 0.0}, index)
                elif kind not in ("frame", "character-response", "prediction-response", "space", "enter"):
                    raise _Mismatch("event.type", "pilot frame/response/press", kind, index)

                # Audit callback cadence independently of the simulator scheduler.
                if frame_origin is not None:
                    next_frame = frame_origin + frame_number * frame_step
                    if kind == "frame":
                        _equal("frame.group", event["group"], frame_group, index)
                        _time("frame.time", time, next_frame, index)
                        if frame_group == "letters":
                            frame_group = "words"
                        else:
                            frame_group, frame_number = "letters", frame_number + 1
                    elif time >= next_frame - 1e-12:
                        raise _Mismatch("frame.order", f"{frame_group} frame at {next_frame} before this event", kind, index)
                elif kind == "frame":
                    raise _Mismatch("frame.readiness", "no frames before readiness", event, index)

                if pending and time > pending[0][0] + TOLERANCE:
                    raise _Mismatch("response.deadline", pending[0][0], time, index)
                if kind in ("space", "enter"):
                    _equal("press.pending_predictions", len(pending), 0, index)
                    if startup is None or not state["ready"]:
                        raise _Mismatch("press.readiness", "initialized with startup responses delivered", state["ready"], index)
                    intent = record["intent"]
                    before = state["typed"]
                    correction = before not in prefixes
                    if not correction:
                        current_word = prefixes.index(before)
                    if current_word >= len(words):
                        raise _Mismatch("press.after_completion", "no additional presses", kind, index)
                    _equal("intent.word_position", intent["word_position"], current_word, index)
                    group = state["letters" if kind == "space" else "words"]
                    active = {clock["id"]: clock for clock in group["clocks"] if clock["active"]}
                    if intent["clock_id"] not in active:
                        raise _Mismatch("intent.clock_id", "active clock", intent["clock_id"], index)
                    undo_id = state["words"]["clocks"][-1]["id"]
                    if correction:
                        _equal("correction.press", kind, "enter", index)
                        _equal("intent.action", intent["action"], "undo", index)
                        _equal("intent.clock_id", intent["clock_id"], undo_id, index)
                        _equal("intent.target", intent["target"], "Undo", index)
                        counts["undo_attempts"] += 1
                    elif kind == "space":
                        _equal("intent.action", intent["action"], "letter", index)
                        letter = words[current_word][len(state["observations"])]
                        _equal("intent.target", intent["target"], letter, index)
                        _equal("intent.clock_label", active[intent["clock_id"]]["label"], letter, index)
                        if not state["observations"]:
                            attempts[current_word] += 1
                    else:
                        _equal("intent.action", intent["action"], "target_enter", index)
                        _equal("intent.target", intent["target"], words[current_word], index)
                        matching = [i for i in state["valid_word_indices"] if i != undo_id
                                    and state["words"]["clocks"][i]["label"].lower() == words[current_word]]
                        _equal("intent.clock_id", intent["clock_id"], matching[0] if matching else undo_id - 1, index)
                    counts["presses"] += 1
                    counts[kind] += 1
                    per_word[current_word] += 1
                elif record["intent"] is not None:
                    raise _Mismatch("intent", None, record["intent"], index)

                if kind in ("character-response", "prediction-response"):
                    if not pending:
                        raise _Mismatch("response.request", "pending prediction", None, index)
                    due, request_id, effect, requested = heapq.heappop(pending)
                    _time("response.time", time, due, index)
                    expected_kind = "character-response" if effect["type"] == "predict-characters" else "prediction-response"
                    _equal("response.type", kind, expected_kind, index)
                    request = requests[request_id]
                    _equal("request.status", request["status"], "delivered", index)
                    _time("request.delivered_time", request["delivered_time"], time, index)
                    if kind == "character-response":
                        _equal("response.index", event["index"], effect["index"], index)
                        characters[ALPHABET[event["index"]]] = event["data"]
                    intervals.append((requested, time))
                    if startup is not None:
                        typing_intervals.append((requested, time))

                before_state = state
                replayed = engine.dispatch(event)
                state = replayed["snapshot"]
                _equal("snapshot_sha256", record["snapshot_sha256"], snapshot_digest(state), index)
                _equal("effects", record["effects"], replayed["effects"], index)
                observed = {"typed": state["typed"], "observation_count": len(state["observations"]),
                            "selection": state["last_selection"] if kind == "enter" else None}
                _equal("observed", record["observed"], observed, index)
                if not before_state["ready"] and state["ready"]:
                    frame_origin = time
                    frame_step = state["letters"]["period"] / state["letters"]["num_divs_time"]
                if kind == "enter":
                    winner = state["last_selection"]["id"]
                    undo_id = before_state["words"]["clocks"][-1]["id"]
                    actual_undos += int(winner == undo_id)
                    counts["wrong_commits"] += int(winner != undo_id and state["typed"] not in prefixes)
                    selections.append({"event_index": index, "id": winner,
                                       "label": before_state["words"]["clocks"][winner]["label"],
                                       "undo": winner == undo_id, "typed": state["typed"],
                                       "observations_before": len(before_state["observations"]),
                                       "observations_after": len(state["observations"])})
                if state["typed"] != before_state["typed"]:
                    transitions.append({"event_index": index, "before": before_state["typed"], "after": state["typed"]})
                for effect in replayed["effects"]:
                    if generated >= len(requests):
                        raise _Mismatch("prediction_requests", "entry for emitted effect", "missing", index)
                    request = requests[generated]
                    due = time + latencies[effect["type"]]
                    _equal("request.id", request["id"], generated, index)
                    _equal("request.effect", request["effect"], effect, index)
                    _time("request.requested_time", request["requested_time"], time, index)
                    _time("request.due_time", request["due_time"], due, index)
                    heapq.heappush(pending, (due, generated, effect, time))
                    generated += 1
                if startup is None and index > 0 and not pending:
                    startup = time
                report["verified_events"] += 1
                report["last_verified_text"] = state["typed"]
            _equal("pending_predictions", len(pending), 0, index)
            _equal("prediction_requests.length", len(requests), generated, index)
            if frame_group != "letters":
                raise _Mismatch("frames", "complete letter/word frame pair", frame_group, index)
            if startup is None:
                raise _Mismatch("startup", "all startup responses delivered", None, index)
            report["complete_trace"] = True
            rebuilt = {"counts": counts, "attempts_by_word": attempts, "presses_by_word": per_word,
                       "final_text": state["typed"], "simulated_elapsed_s": previous_time,
                       "startup_elapsed_s": startup, "typing_elapsed_s": previous_time - startup,
                       "prediction_wait_s": _duration(intervals), "typing_prediction_wait_s": _duration(typing_intervals),
                       "actual_undo_selections": actual_undos, "text_transitions": transitions, "selections": selections}
            report["reconstructed"] = rebuilt
            for field in ("counts", "attempts_by_word", "presses_by_word", "final_text"):
                _equal(field, result[field], rebuilt[field])
            for field in ("simulated_elapsed_s", "startup_elapsed_s", "typing_elapsed_s", "prediction_wait_s", "typing_prediction_wait_s"):
                _time(field, result[field], rebuilt[field])
            _equal("final_snapshot", result["final_snapshot"], state)
            _equal("character_responses", meta["character_responses"], characters)
            _equal("effective_transition_matrix", meta["effective_transition_matrix"], state["transition_matrix"])
            _equal("effective_transition_matrix_sha256", meta["effective_transition_matrix_sha256"], snapshot_digest(state["transition_matrix"]))
            _equal("final_text_known", result["final_text_known"], True)
            _equal("failure_reason", result["failure_reason"], None)
            _equal("error", result["error"], None)
            _equal("completion.text", state["typed"], prefixes[-1])
            _equal("completed", result["completed"], True)
            report.update(valid=True, complete_trace=True)
    except _Mismatch as exc:
        report["errors"].append(exc.detail)
    except Exception as exc:
        report["errors"].append({"field": "validation", "event_index": index,
                                 "type": type(exc).__name__, "message": str(exc)})
    return report
