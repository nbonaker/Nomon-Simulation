"""Run the recorded-click QuickClick baseline through the shared JS engine."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

from User_Simulation.evaluation.historical_oneclick import study_data

from .bridge_simulated_user import BridgeSimulatedUser, SequenceClickSource, normalize_target
from .websocket_predictions import add_websocket_arguments, backend_from_arguments


BUDGET_FAILURES = {"word_attempt_budget", "word_press_budget", "phrase_press_budget"}


def write_json(path, value):
    pending = path.with_suffix(".tmp")
    with pending.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    pending.replace(path)


def summarize(rows, planned):
    failures = Counter(row["failure_reason"] for row in rows if not row["completed"])
    attempted = len(rows)
    incomplete = sum(failures.values())
    return {
        "planned": planned,
        "attempted": attempted,
        "unattempted": planned - attempted,
        "completed": attempted - incomplete,
        "incomplete": incomplete,
        "incomplete_rate": incomplete / attempted if attempted else None,
        "budget_stops": sum(failures[reason] for reason in BUDGET_FAILURES),
        "failure_reasons": dict(failures),
        "counts": {
            field: sum(row["counts"][field] for row in rows)
            for field in ("presses", "space", "enter", "wrong_commits", "undo_attempts")
        },
    }


def run_study(output_dir, user, backend_factory, *, users, phrase_count, backend_settings):
    """Use one deterministic click stream per user and one engine per phrase."""
    users = tuple(users)
    if not users or len(set(users)) != len(users) or set(users) - set(study_data.REAL_USERS):
        raise ValueError("Select unique users from A, B, C, D, F, G")
    corpus = study_data.load_fixed_iv_phrases(phrase_count)
    phrases = [
        {"id": str(phrase_id), "text": normalize_target(phrase)}
        for phrase_id, phrase in zip(corpus["Comparison Phrase ID"], corpus["Phrase Text"])
    ]
    required = sum(
        min(user.max_phrase_presses, len(phrase["text"].split()) * user.max_word_presses)
        for phrase in phrases
    )
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError(f"Choose an empty output directory: {root}")

    schedules = {}
    for user_id in users:
        original = study_data.load_real_user_clicks(user_id)
        seed = study_data.bootstrap_seed_for_user(user_id)
        sampled = study_data.block_bootstrap_click_stream(
            original, required_clicks=required, seed=seed
        )
        # The retained helper prepends calibration rows for the old keyboard.
        clicks = sampled.loc[sampled["Session Num"].notna()]
        schedule_path = root / f"user_{user_id}_click_schedule.csv"
        clicks.to_csv(schedule_path, index=False)
        schedules[user_id] = {
            "file": schedule_path.name,
            "sha256": hashlib.sha256(schedule_path.read_bytes()).hexdigest(),
            "seed": seed,
            "original_rows": int(original["Session Num"].notna().sum()),
            "scheduled_rows": len(clicks),
        }
    write_json(root / "manifest.json", {
        "version": 1,
        "users": users,
        "phrases": phrases,
        "sampling": {"method": "within_session_moving_block_bootstrap",
                     "block_size": study_data.BOOTSTRAP_BLOCK_SIZE, "users": schedules},
        "engine_config": user.config,
        "simulation_settings": {
            "max_attempts_per_word": user.max_attempts,
            "max_presses_per_word": user.max_word_presses,
            "max_presses_per_phrase": user.max_phrase_presses,
            "character_transition_source": user.character_source,
            "prediction_latencies_s": user.latencies,
        },
        "backend": backend_settings,
    })

    rows = []
    stopped = None
    for user_id in users:
        cursor = 0
        with (root / schedules[user_id]["file"]).open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            source = SequenceClickSource(reader)
            for index, phrase in enumerate(phrases, 1):
                result = user.run_phrase(phrase["text"], source, backend_factory())
                trace = result.pop("trace")
                directory = Path(f"user_{user_id}", f"phrase_{index:03d}")
                (root / directory).mkdir(parents=True)
                write_json(root / directory / "result.json", result)
                write_json(root / directory / "trace.json", trace)
                # Generated schedules have one physical line per sample plus the header.
                end = max(0, reader.line_num - 1)
                rows.append({
                    "user": user_id, "phrase_id": phrase["id"], "artifacts": directory.as_posix(),
                    "schedule_start": cursor, "schedule_end_exclusive": end,
                    **{key: result[key] for key in (
                        "target", "final_text", "completed", "failure_reason",
                        "counts", "simulated_elapsed_s", "typing_elapsed_s",
                    )},
                })
                cursor = end
                write_json(root / "phrases.json", rows)
                outcome = "completed" if result["completed"] else result["failure_reason"]
                print(f"User {user_id}, phrase {index}/{len(phrases)}: {outcome}", flush=True)
                if not result["completed"] and result["failure_reason"] not in BUDGET_FAILURES:
                    stopped = {"user": user_id, "phrase_id": phrase["id"],
                               "reason": result["failure_reason"], "artifacts": directory.as_posix()}
                    break
        if stopped:
            break

    totals = summarize(rows, len(users) * len(phrases))
    by_user = {uid: summarize([row for row in rows if row["user"] == uid], len(phrases))
               for uid in users}
    rates = [item["incomplete_rate"] for item in by_user.values()
             if item["incomplete_rate"] is not None]
    summary = {
        **totals, "by_user": by_user, "stopped": stopped,
        "mean_user_incomplete_rate": sum(rates) / len(rates) if rates else None,
    }
    write_json(root / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--users", nargs="+", choices=study_data.REAL_USERS,
                        default=study_data.REAL_USERS)
    parser.add_argument("--phrase-count", type=int, default=study_data.FIXED_IV_PHRASE_COUNT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--engine-repo")
    parser.add_argument("--node-executable", default="node")
    parser.add_argument("--rotate-index", type=int, default=5)
    parser.add_argument("--character-prediction-latency-s", type=float, default=0.0)
    parser.add_argument("--word-prediction-latency-s", type=float, default=0.0)
    parser.add_argument("--character-transitions", type=Path,
                        help="Hold the character transition responses fixed across conditions")
    parser.add_argument("--backend", choices=("textslinger-websocket", "textslinger-ngram"),
                        default="textslinger-websocket")
    parser.add_argument("--model", type=Path, help="Local KenLM model for textslinger-ngram")
    parser.add_argument("--vocabulary", type=Path, help="Local word list for textslinger-ngram")
    add_websocket_arguments(parser)
    args = parser.parse_args(argv)
    try:
        if args.backend == "textslinger-ngram":
            if not args.model or not args.vocabulary:
                parser.error("textslinger-ngram requires --model and --vocabulary")
            if any(value is not None for value in (args.prediction_server, args.prediction_model,
                                                   args.prediction_vocabulary, args.prediction_timeout_s)):
                parser.error("WebSocket prediction options require textslinger-websocket")
            from .bridge_predictions import TextSlingerNGramBackend
            backend = TextSlingerNGramBackend.from_local(args.model, args.vocabulary)
            def backend_factory():
                return backend
            backend_settings = backend.metadata()
        else:
            if args.model or args.vocabulary:
                parser.error("Use --prediction-model and --prediction-vocabulary for WebSocket")
            def backend_factory():
                return backend_from_arguments(args)
            backend_settings = backend_factory().metadata()
        fixed = json.loads(args.character_transitions.read_text()) if args.character_transitions else None
        user = BridgeSimulatedUser(
            engine_repo=args.engine_repo, node_executable=args.node_executable,
            rotate_index=args.rotate_index,
            character_transition_source="fixed" if args.character_transitions else "backend",
            fixed_character_responses=fixed,
            character_prediction_latency_s=args.character_prediction_latency_s,
            word_prediction_latency_s=args.word_prediction_latency_s,
        )
        summary = run_study(args.output_dir, user, backend_factory, users=args.users,
                            phrase_count=args.phrase_count, backend_settings=backend_settings)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"Study setup or execution failed: {exc}\n")
    print(f"{summary['completed']}/{summary['attempted']} attempted phrases completed. "
          f"Summary: {args.output_dir / 'summary.json'}")
    return int(summary["stopped"] is not None)


if __name__ == "__main__":
    raise SystemExit(main())
