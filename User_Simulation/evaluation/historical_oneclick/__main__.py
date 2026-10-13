"""Analyze explicitly supplied historical CSVs, without models or simulation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def analyze(input_dir, output_dir, *, kind, table=None, reliability_floor=None, plots=False):
    source, output = Path(input_dir).resolve(), Path(output_dir).resolve()
    if not source.is_dir():
        raise ValueError(f'Saved-run directory does not exist: {source}')
    if output == source or source in output.parents or output in source.parents:
        raise ValueError('Use a separate output directory, outside the saved-run directory')
    output.mkdir(parents=True, exist_ok=True)
    config_path = source / 'run_config.json'
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    if kind in ('global', 'phase2'):
        if reliability_floor is None:
            reliability_floor = config.get('reliability_floor')
        if reliability_floor is None or not 0 <= float(reliability_floor) <= 1:
            raise ValueError('Supply --reliability-floor or a saved run_config.json with that value')
        reliability_floor = float(reliability_floor)
    tables = {}
    if kind == 'raw':
        from OneClick_Simulation.examples.text_simulation.analyze_results import analyze_results
        users, aggregate, failures = analyze_results(source, output)
        tables = {'metrics_per_user': users, 'metrics_aggregate': aggregate, 'failure_reasons': failures}
        table_path = None
    else:
        if table is None:
            raise ValueError('--table must name a saved CSV inside --input-dir')
        table_path = (source / table).resolve()
        if source not in table_path.parents:
            raise ValueError('--table must be inside the saved-run directory')
        frame = pd.read_csv(table_path)
        if kind == 'global':
            from . import global_sweep as g, phase1
            users, summary = g.build_global_summary(frame, reliability_floor)
            tables = {'per_user_summary': users, 'global_summary': summary,
                      'ranked_summary': g.rank_global_summary(summary),
                      'curve_points': g.build_global_curve_points(frame),
                      'failure_distribution': phase1.build_failure_distribution(frame)}
            trials, global_trials = g.build_global_trial_summary(frame)
            tables.update(trial_summary=trials, global_trial_summary=global_trials)
            if plots:
                g.create_confirmation_plots(output, frame, users, summary, tables['curve_points'], global_trials, reliability_floor)
        elif kind in ('phase1', 'phase2'):
            from . import phase1, phase2
            summary = phase1.build_combo_summary(frame)
            curves = phase1.build_curve_points(frame)
            tables = {'per_user_summary': summary, 'curve_points': curves,
                      'failure_distribution': phase1.build_failure_distribution(frame)}
            if kind == 'phase1':
                tables['best_comparison'] = phase1.build_best_comparison(summary)
                if plots:
                    indices = sorted(set(frame.space_period_index) | set(frame.enter_period_index))
                    phase1.create_plots(output, summary, curves, list(frame.user_id.unique()), indices)
            else:
                summary, trials = phase2.enrich_candidate_summary(frame, reliability_floor)
                selection = phase2.build_selection_summary(summary, reliability_floor)
                tables.update(per_user_summary=summary, trial_summary=trials, selection_summary=selection,
                              paired_comparisons=phase2.build_paired_comparisons(frame, summary, selection))
                if plots:
                    phase2.create_plots(output, frame, summary, trials, curves, selection)
        elif kind == 'speed':
            from . import speed
            summary, curves = speed.build_speed_summary(frame), speed.build_curve_points(frame)
            tables = {'per_user_summary': summary, 'curve_points': curves}
            if plots:
                from matplotlib import colormaps
                periods = summary[['clock_period_index', 'clock_period_s', 'clock_period_label']].drop_duplicates().sort_values('clock_period_index').to_dict('records')
                for i, period in enumerate(periods):
                    period['color'] = colormaps['viridis'](i / max(1, len(periods)-1))
                speed.create_plots(output, curves, list(frame.user_id.unique()), periods)
        elif kind == 'comparison':
            from . import comparison as c
            paired = c.build_phrase_trial_comparison(frame)
            summary = c.build_comparison_summary(paired)
            tables = {'paired_phrase_results': paired, 'trial_summary': c.build_trial_summary(frame), 'comparison_summary': summary}
            if plots: c.plot_summary(summary, output)
        elif kind == 'selected-comparison':
            from . import selected_comparison as c
            users = c.summarize_system_results(frame)
            paired = c.build_paired_results(frame)
            tables = {'per_user_summary': users, 'macro_summary': c.build_macro_summary(users),
                      'paired_results': paired, 'paired_summary': c.build_paired_summary(paired),
                      'trial_summary': c.build_trial_summary(frame), 'curve_points': c.build_curve_points(frame),
                      'failure_distribution': c.build_failure_distribution(frame)}
            # Plotting this family also requires a saved timing-profile table.
            if plots:
                profiles = pd.read_csv(source / 'user_bootstrap_profiles.csv')
                c.create_plots(users, tables['macro_summary'], tables['curve_points'], tables['paired_summary'],
                               tables['failure_distribution'], profiles, output)
            c.write_report(output, tables['macro_summary'], tables['paired_summary'])
        elif kind == 'config-sweep':
            from . import config_summary as c
            manifest = pd.read_csv(source / 'sweep_config.csv')
            users, summaries = [], []
            for row in manifest.to_dict('records'):
                subset = frame.loc[frame['config_id'] == row['config_id']]
                settings = {column: row[column] for column in c.CONFIG_COLUMNS if column != 'config_id'}
                user_table = c.summarize_by_user_config(subset, row['config_id'], settings)
                users.append(user_table)
                summaries.append(c.summarize_by_config(user_table))
            tables = {'summary_by_user_config': pd.concat(users, ignore_index=True),
                      'summary_by_config': pd.concat(summaries, ignore_index=True)}
            # The report consumes copies in the analysis directory, never rewrites originals.
            frame.to_csv(output / 'phrase_results.csv', index=False)
            manifest.to_csv(output / 'sweep_config.csv', index=False)
            for filename in ('fixed_iv_phrase_corpus.csv', 'click_stream_sufficiency.csv'):
                if (source / filename).is_file():
                    (output / filename).write_bytes((source / filename).read_bytes())
        else:
            raise ValueError(f'Unknown analysis kind: {kind}')
    for name, frame in tables.items():
        frame.to_csv(output / f'{name}.csv', index=False)
    if kind == 'config-sweep':
        from .config_summary import write_sweep_report
        metadata_path = source / 'lm_config.json'
        write_sweep_report(output, json.loads(metadata_path.read_text()) if metadata_path.is_file() else {})
    provenance = {'source': str(source), 'kind': kind, 'input_table': str(table_path) if table_path else None,
                  'input_sha256': hashlib.sha256(table_path.read_bytes()).hexdigest() if table_path else None,
                  'reliability_floor': reliability_floor, 'historical_semantics': True,
                  'simulation_executed': False, 'tables': list(tables)}
    (output / 'analysis_metadata.json').write_text(json.dumps(provenance, indent=2)+'\n')
    return tables


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--kind', choices=('raw','config-sweep','comparison','selected-comparison','speed','phase1','phase2','global'), required=True)
    parser.add_argument('--table', help='Saved CSV filename relative to input directory; required except for raw')
    parser.add_argument('--reliability-floor', type=float, help='Historical threshold; otherwise read run_config.json')
    parser.add_argument('--plots', action='store_true')
    args = parser.parse_args(argv)
    try:
        tables = analyze(args.input_dir, args.output_dir, kind=args.kind, table=args.table,
                         reliability_floor=args.reliability_floor, plots=args.plots)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        parser.error(str(exc))
    print(f'Wrote {len(tables)} historical analysis tables to {args.output_dir.resolve()}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
