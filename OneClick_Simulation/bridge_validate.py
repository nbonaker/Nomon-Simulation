"""Run and audit small shared-engine acceptance scenarios, or audit saved files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .bridge_simulated_user import (BridgeSimulatedUser, ClickSample, FixedPredictionBackend,
                                    SequenceClickSource, ZeroNoiseClickSource)
from .bridge_validation import validate_run
from .bridge_validation_scenarios import fixed_scenarios, MissFirstTargetEnter


def _write(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def check_scenario(report, expected):
    """Check independently specified expectations, never derived from run output."""
    if not report['valid']:
        return report
    rebuilt = report['reconstructed']
    actual = {**rebuilt, 'enter_texts': [s['typed'] for s in rebuilt['selections']],
              'selection_ids': [s['id'] for s in rebuilt['selections']]}
    for field, value in expected.items():
        if field == 'requires_correction':
            correct = rebuilt['counts']['wrong_commits'] > 0 and rebuilt['counts']['undo_attempts'] > 0 and rebuilt['actual_undo_selections'] > 0
            if correct == value:
                continue
            found = correct
        else:
            found = actual.get(field)
            if found == value:
                continue
        report['valid'] = False
        report['errors'].append({'field': f'scenario.{field}', 'expected': value, 'actual': found, 'event_index': None})
    # Midword Undo must discard real observations, not just leave text unchanged.
    for selection in rebuilt['selections']:
        if selection['undo'] and selection['observations_before'] and selection['observations_after'] != 0:
            report['valid'] = False
            report['errors'].append({'field': 'scenario.midword_undo', 'expected': 0,
                                     'actual': selection['observations_after'], 'event_index': selection['event_index']})
    return report


def audit_directory(directory, engine_repo=None, node_executable='node'):
    """Audit disk artifacts. This path never loads a prediction model."""
    directory = Path(directory)
    try:
        result = json.loads((directory / 'result.json').read_text(encoding='utf-8'))
        trace = json.loads((directory / 'trace.json').read_text(encoding='utf-8'))
        report = validate_run(result, trace, engine_repo, node_executable)
        scenario = directory / 'scenario.json'
        if scenario.exists():
            specification = json.loads(scenario.read_text(encoding='utf-8'))
            if result['target'] != specification['target']:
                report['valid'] = False
                report['errors'].append({'field': 'scenario.target', 'expected': specification['target'],
                                         'actual': result['target'], 'event_index': None})
            check_scenario(report, specification['expected'])
    except Exception as exc:
        report = {'version': 1, 'valid': False, 'complete_trace': False, 'verified_events': 0, 'reconstructed': {},
                  'errors': [{'field': 'artifacts', 'type': type(exc).__name__, 'message': str(exc), 'event_index': None}]}
    return report


def _save_and_audit(directory, name, target, expected, options, source_description, run, engine_repo, node):
    directory.mkdir(parents=True, exist_ok=True)
    _write(directory / 'scenario.json', {'name': name, 'target': target, 'expected': expected,
                                        'options': options, 'click_source': source_description})
    result = run()
    trace = result.pop('trace')
    _write(directory / 'result.json', result)
    _write(directory / 'trace.json', trace)
    report = audit_directory(directory, engine_repo, node)
    _write(directory / 'validation.json', report)
    return {'scenario': name, 'valid': report['valid'], 'completed': result['completed'],
            'final_text': result['final_text'], 'counts': result['counts'],
            'simulated_elapsed_s': result['simulated_elapsed_s'], 'verified_events': report['verified_events'],
            'complete_trace': report['complete_trace'], 'errors': report['errors']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('bridge_validation_output'))
    parser.add_argument('--engine-repo')
    parser.add_argument('--node-executable', default='node')
    parser.add_argument('--audit', type=Path, help='Audit an existing scenario directory, without running predictions')
    parser.add_argument('--include-ngram', action='store_true', help='Also run local model word, sentence, and correction scenarios')
    parser.add_argument('--model', type=Path)
    parser.add_argument('--vocabulary', type=Path)
    args = parser.parse_args(argv)
    if args.audit:
        report = audit_directory(args.audit, args.engine_repo, args.node_executable)
        # An audit is read-only; print the report without replacing source artifacts.
        print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
        return 0 if report['valid'] else 1
    if args.include_ngram and (args.model is None or args.vocabulary is None):
        parser.error('--include-ngram requires --model and --vocabulary')
    if not args.include_ngram and (args.model is not None or args.vocabulary is not None):
        parser.error('--model and --vocabulary require --include-ngram')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    bridge_options = {'engine_repo': args.engine_repo, 'node_executable': args.node_executable}

    def execute(name, target, expected, options, source_description, run):
        directory = args.output_dir / name
        try:
            summary = _save_and_audit(directory, name, target, expected, options, source_description,
                                      run, args.engine_repo, args.node_executable)
        except Exception as exc:
            summary = {'scenario': name, 'valid': False,
                       'errors': [{'field': 'scenario.execution', 'type': type(exc).__name__, 'message': str(exc), 'event_index': None}]}
            directory.mkdir(parents=True, exist_ok=True)
            _write(directory / 'validation.json', summary)
        summaries.append(summary)
        print(f"{'PASS' if summary['valid'] else 'FAIL'} {name}: "
              f"text={summary.get('final_text')!r}, counts={summary.get('counts')}, "
              f"time={summary.get('simulated_elapsed_s')}, events={summary.get('verified_events')}")
        for error in summary['errors']:
            print('  ' + json.dumps(error, ensure_ascii=False))
        # Keep completed scenario diagnostics if a later scenario is interrupted.
        _write(args.output_dir / 'summary.json', {'valid': False, 'passed_so_far': all(s['valid'] for s in summaries),
                                                'finished': False, 'scenarios': summaries})

    for spec in fixed_scenarios():
        def run(spec=spec):
            user = BridgeSimulatedUser(**bridge_options, **spec['options'])
            backend = FixedPredictionBackend(spec['characters'], spec['word_responses'])
            return user.run_phrase(spec['target'], SequenceClickSource([ClickSample(x) for x in spec['offsets']]), backend)
        execute(spec['name'], spec['target'], spec['expected'], spec['options'], {'offsets_s': spec['offsets']}, run)
    if args.include_ngram:
        try:
            from .bridge_predictions import TextSlingerNGramBackend
            backend = TextSlingerNGramBackend.from_local(args.model, args.vocabulary)
        except Exception as exc:
            summaries.append({'scenario': 'ngram_setup', 'valid': False,
                              'errors': [{'field': 'model', 'type': type(exc).__name__, 'message': str(exc), 'event_index': None}]})
            print(f'FAIL ngram_setup: {exc}')
        else:
            options = {'character_prediction_latency_s': .2, 'word_prediction_latency_s': .1}
            for name, target, correction in [('ngram_word', 'hello', False), ('ngram_sentence', 'hello world', False),
                                              ('ngram_correction', 'hello', True)]:
                def run(target=target, correction=correction):
                    user = BridgeSimulatedUser(**bridge_options, **options)
                    source = MissFirstTargetEnter(user) if correction else ZeroNoiseClickSource()
                    return user.run_phrase(target, source, backend)
                execute(name, target, {'requires_correction': True} if correction else {}, options,
                        {'policy': 'miss_first_target_enter' if correction else 'zero_noise'}, run)
    valid = all(summary['valid'] for summary in summaries)
    _write(args.output_dir / 'summary.json', {'valid': valid, 'finished': True, 'scenarios': summaries})
    print(f"Validation artifacts: {args.output_dir.resolve()}")
    return 0 if valid else 1


if __name__ == '__main__':
    raise SystemExit(main())
