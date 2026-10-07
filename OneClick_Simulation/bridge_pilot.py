"""Run/replay the shared-engine pilot with fixtures or a local TextSlinger n-gram."""
import argparse
import csv
import json
from pathlib import Path

from .bridge_simulated_user import (BridgeSimulatedUser, FixedPredictionBackend,
                                   ZeroNoiseClickSource, SequenceClickSource, replay_trace)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phrase', default='hello world')
    parser.add_argument('--fixture', type=Path, default=Path(__file__).with_name('bridge_fixtures') / 'hello_world.json')
    parser.add_argument('--backend', choices=('fixture', 'textslinger-ngram'), default='fixture')
    parser.add_argument('--model', type=Path, help='Local KenLM/ARPA model; required for textslinger-ngram')
    parser.add_argument('--vocabulary', type=Path, help='Local word list; required for textslinger-ngram')
    parser.add_argument('--recognizer-nbest', type=int, default=1000)
    parser.add_argument('--character-transition-source', choices=('backend', 'fixed'), default='backend')
    parser.add_argument('--character-transitions', type=Path, help='Shared response or character-keyed responses JSON; required for fixed')
    parser.add_argument('--character-prediction-latency-s', type=float, default=0.0)
    parser.add_argument('--word-prediction-latency-s', type=float, default=0.0)
    parser.add_argument('--clicks', type=Path, help='Recorded CSV; omitted means repeatable zero noise')
    parser.add_argument('--engine-repo')
    parser.add_argument('--node-executable', default='node')
    parser.add_argument('--rotate-index', type=int, default=5)
    parser.add_argument('--output-dir', type=Path, default=Path('bridge_pilot_output'))
    parser.add_argument('--replay', type=Path, help='Verify a saved trace instead of running a new phrase')
    args = parser.parse_args(argv)
    if args.replay:
        report = replay_trace(json.loads(args.replay.read_text()), args.engine_repo, args.node_executable)
        print(json.dumps({'verified_events': report['verified_events'], 'complete_trace': report['complete_trace'],
                          'typed': report['snapshot']['typed']}, indent=2))
        return 0 if report['complete_trace'] else 1
    if args.backend == 'textslinger-ngram' and (args.model is None or args.vocabulary is None):
        parser.error('--backend textslinger-ngram requires --model and --vocabulary')
    if (args.character_transition_source == 'fixed') != (args.character_transitions is not None):
        parser.error('--character-transition-source fixed requires --character-transitions (and vice versa)')
    fixed_characters = json.loads(args.character_transitions.read_text()) if args.character_transitions else None
    user = BridgeSimulatedUser(engine_repo=args.engine_repo, node_executable=args.node_executable,
                               rotate_index=args.rotate_index, character_transition_source=args.character_transition_source,
                               fixed_character_responses=fixed_characters,
                               character_prediction_latency_s=args.character_prediction_latency_s,
                               word_prediction_latency_s=args.word_prediction_latency_s)
    if args.backend == 'textslinger-ngram':
        from .bridge_predictions import TextSlingerNGramBackend
        backend = TextSlingerNGramBackend.from_local(args.model, args.vocabulary, recognizer_nbest=args.recognizer_nbest)
    else:
        backend = FixedPredictionBackend.from_dict(json.loads(args.fixture.read_text()))
    if args.clicks:
        with args.clicks.open(newline='') as stream:
            source = SequenceClickSource(list(csv.DictReader(stream)))
    else:
        source = ZeroNoiseClickSource()
    result = user.run_phrase(args.phrase, source, backend)
    trace = result.pop('trace')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'result.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    (args.output_dir / 'trace.json').write_text(json.dumps(trace, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps({key: result[key] for key in ('target', 'final_text', 'completed', 'failure_reason', 'simulated_elapsed_s', 'counts')}, indent=2))
    print(f"Result and replay trace: {args.output_dir.resolve()}")
    return 0 if result['completed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
