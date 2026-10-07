"""Run with: python -m OneClick_Bridge.demo [--engine-repo PATH]. No network/LM."""
from __future__ import annotations

import argparse
import json

from . import OneClickEngineBridge

ALPHABET = "abcdefghijklmnopqrstuvwxyz'"


def run_demo(engine_repo=None, node_executable="node", emit=print):
    with OneClickEngineBridge(engine_repo=engine_repo, node_executable=node_executable) as engine:
        emit("Engine: " + json.dumps(engine.metadata, ensure_ascii=False))
        now = 0.0
        frame_index = 0

        def answer_requests(effects):
            for effect in effects:
                if effect["type"] == "predict-characters":
                    engine.dispatch({"type": "character-response", "index": effect["index"], "time": now,
                                     "data": {"results": [{"token": token, "logProb": -3.3} for token in ALPHABET]}})
                elif effect["type"] == "predict-words":
                    # Fixed fixture backend. This is deliberately not a real recognizer.
                    data = {"prefix": [{"text": "hello", "logprob": -0.1}], "best": []}
                    if len(effect["distribs"]) == 1:
                        data["best"] = [{"text": "h", "logprob": -0.2}]
                    engine.dispatch({"type": "prediction-response", "time": now, "data": data})
                else:
                    raise AssertionError(f"Unexpected effect: {effect}")

        def aim(group_name, clock_id, offset=0.0):
            nonlocal now, frame_index
            # Step the simulated animation schedule until this clock's next target
            # time lies before the next frame. Do not sleep or consult wall time.
            earliest = now + 0.001  # Keep separate simulated presses distinct.
            for _ in range(1000):
                group = engine.get_snapshot()[group_name]
                target = (group["latest_time"] + group["period"] / 2
                          - group["phases"][clock_id] * group["period"] / group["num_divs_time"] + offset)
                next_frame = (frame_index + 1) * 0.04
                if max(now, earliest) <= target < next_frame:
                    now = target
                    return
                frame_index += 1
                now = frame_index * 0.04
                engine.dispatch({"type": "frame", "group": "letters", "time": now})
                engine.dispatch({"type": "frame", "group": "words", "time": now})
            raise AssertionError("Could not aim at the selected clock")

        answer_requests(engine.dispatch({"type": "initialize", "time": now})["effects"])
        assert engine.get_snapshot()["ready"]
        aim("letters", ALPHABET.index("h"), offset=0.05)
        click = engine.dispatch({"type": "space", "time": now})
        row = click["snapshot"]["formatted_observations"][0]["distrib"]
        assert len(row) == 27
        strongest = sorted(row, key=lambda item: item["logProb"], reverse=True)[:3]
        emit(f"Space at {now:.3f}s: 27 likelihoods; strongest = {strongest}")
        answer_requests(click["effects"])
        clocks = [clock for clock in engine.get_snapshot()["words"]["clocks"] if clock["active"]]
        emit("Word clocks: " + json.dumps(clocks, ensure_ascii=False))
        word_id = next(clock["id"] for clock in clocks if clock["label"] == "hello")
        aim("words", word_id)
        commit = engine.dispatch({"type": "enter", "time": now})
        assert commit["snapshot"]["typed"] == "hello "
        emit(f"Enter at {now:.3f}s: committed {commit['snapshot']['typed']!r}")
        answer_requests(commit["effects"])
        undo_id = next(clock["id"] for clock in engine.get_snapshot()["words"]["clocks"] if clock["label"] == "Undo")
        aim("words", undo_id)
        undo = engine.dispatch({"type": "enter", "time": now})
        assert undo["snapshot"]["typed"] == ""
        assert undo["snapshot"]["delay_model"]["n_samples"] == 0
        answer_requests(undo["effects"])
        emit(f"Undo at {now:.3f}s: text restored to ''; timing update rolled back.")
        return {"typed": engine.get_snapshot()["typed"], "metadata": engine.metadata, "simulated_time": now}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-repo")
    parser.add_argument("--node-executable", default="node")
    args = parser.parse_args()
    run_demo(args.engine_repo, args.node_executable)


if __name__ == "__main__":
    main()
