"""Historical parity fixtures and a dependency boundary for retired QuickClick code."""
import ast
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pandas as pd

from . import global_sweep as g, phase1, phase2
from .__main__ import analyze

FIXTURES = Path(__file__).with_name('fixtures')
ROOT = Path(__file__).resolve().parents[3]


def retired_imports(source, module, retired):
    found = []
    tree = ast.parse(source)
    for node in ast.walk(tree):
        candidates = []
        if isinstance(node, ast.Import):
            candidates = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            parent = node.module or ''
            if node.level:
                parent = importlib.util.resolve_name('.'*node.level + parent, module.rsplit('.',1)[0])
            candidates = [parent] + [parent+'.'+a.name for a in node.names]
        elif isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant):
            function = node.func.id if isinstance(node.func,ast.Name) else node.func.attr if isinstance(node.func,ast.Attribute) else ''
            if function in ('import_module','__import__') and isinstance(node.args[0].value,str):
                candidates = [node.args[0].value]
        for candidate in candidates:
            if candidate == 'OneClick_Core' or candidate.startswith('OneClick_Core.') or any(candidate == x or candidate.startswith(x+'.') for x in retired):
                found.append((node.lineno,candidate))
    return found


class RetirementTests(unittest.TestCase):
    def test_no_retained_source_imports_retired_modules(self):
        retired = json.loads(Path(__file__).with_name('removed_modules.json').read_text())
        paths = subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=ROOT).decode().split('\0')
        failures = []
        for name in paths:
            path = ROOT/name
            if path.suffix == '.py' and path.is_file():
                failures.extend((name,line,module) for line,module in retired_imports(path.read_text(), name[:-3].replace('/','.'), retired))
        self.assertEqual(failures, [])
        for module in retired:
            self.assertFalse((ROOT/Path(*module.split('.')).with_suffix('.py')).exists())

    def test_guard_catches_parent_relative_and_dynamic_imports(self):
        old=['OneClick_Simulation.simulated_user']
        for source in ('from OneClick_Simulation import simulated_user', 'from . import simulated_user',
                       'import OneClick_Core.config', 'importlib.import_module("OneClick_Simulation.simulated_user")'):
            self.assertTrue(retired_imports(source,'OneClick_Simulation.example',old))

    def test_analysis_matches_pre_retirement_tables(self):
        frame = pd.read_csv(FIXTURES/'global_input.csv')
        users, summary = g.build_global_summary(frame,.5)
        trials,global_trials = g.build_global_trial_summary(frame)
        actual={'global_per_user':users,'global_summary':summary,'global_ranked':g.rank_global_summary(summary),
                'global_curves':g.build_global_curve_points(frame),'global_trials_0':trials,'global_trials_1':global_trials,
                'phase1_summary':phase1.build_combo_summary(frame),'phase1_failures':phase1.build_failure_distribution(frame),
                'phase2_trials':phase2.build_trial_summary(frame)}
        for name, table in actual.items():
            with self.subTest(name=name):
                expected = pd.read_csv(FIXTURES/(name+'.csv'))
                pd.testing.assert_frame_equal(table.reset_index(drop=True),expected,check_dtype=False,rtol=0,atol=1e-10)

    def test_analysis_cli_preserves_inputs_and_writes_tables_and_plots(self):
        before = {p.name:p.read_bytes() for p in FIXTURES.iterdir()}
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'analysis'
            tables=analyze(FIXTURES,output,kind='global',table='global_input.csv',reliability_floor=.5,plots=True)
            self.assertEqual(len(tables),7)
            pd.testing.assert_frame_equal(pd.read_csv(output/'global_summary.csv'),pd.read_csv(FIXTURES/'global_summary.csv'),rtol=0,atol=1e-10)
            self.assertTrue(list(output.rglob('*.png')))
            self.assertFalse(json.loads((output/'analysis_metadata.json').read_text())['simulation_executed'])
            with self.assertRaisesRegex(ValueError,'reliability'):
                analyze(FIXTURES,Path(tmp)/'missing',kind='global',table='global_input.csv')
            with self.assertRaisesRegex(ValueError,'separate output'):
                analyze(FIXTURES,FIXTURES,kind='raw')
        self.assertEqual(before,{p.name:p.read_bytes() for p in FIXTURES.iterdir()})

    def test_offline_imports_do_not_load_simulators_or_prediction_models(self):
        code = '''
import importlib, importlib.abc, pathlib, sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.startswith(('OneClick_Bridge', 'OneClick_Core', 'OneClick_Text.language_model', 'textslinger', 'User_Simulation.simulated_user', 'Nomon_Text')):
   raise AssertionError('Unexpected simulation/model import: '+fullname)
sys.meta_path.insert(0,Block())
package='User_Simulation.evaluation.historical_oneclick'
for name in ('comparison','bootstrap','speed','phase1','phase2','global_sweep','selected_comparison','study_data','config_summary','phrase_data','__main__'):
 importlib.import_module(package+'.'+name)
'''
        subprocess.run([sys.executable,'-c',code],cwd=ROOT,check=True)

    def test_raw_summary_matches_pre_retirement_table(self):
        from OneClick_Simulation.examples.text_simulation.analyze_results import summarize_users
        actual = summarize_users(pd.read_csv(FIXTURES/'raw_input.csv'))
        pd.testing.assert_frame_equal(actual, pd.read_csv(FIXTURES/'raw_summary.csv'),
                                      check_dtype=False, rtol=0, atol=1e-10)

    def test_phase_speed_and_saved_configuration_entry_points(self):
        from . import config_summary as c
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/'source'
            source.mkdir()
            frame = pd.read_csv(FIXTURES/'global_input.csv')
            frame['clock_period_index'] = frame.space_period_index
            frame['clock_period_s'] = frame.space_period_s
            frame['clock_period_label'] = frame.space_period_s.astype(str)+'s'
            frame.to_csv(source/'speed.csv', index=False)
            raw = pd.read_csv(FIXTURES/'raw_input.csv')
            raw['config_id'] = 'saved'
            raw.to_csv(source/'raw.csv', index=False)
            settings = dict(zip(c.CONFIG_COLUMNS, ['saved', 'baseline', 'default', 6.0,
                                                   False, 'default', 'default', 'default', 1.0]))
            pd.DataFrame([settings]).to_csv(source/'sweep_config.csv', index=False)
            for kind, directory, table in [('phase1', FIXTURES, 'global_input.csv'),
                                           ('phase2', FIXTURES, 'global_input.csv'),
                                           ('speed', source, 'speed.csv'),
                                           ('config-sweep', source, 'raw.csv')]:
                with self.subTest(kind=kind):
                    result = analyze(directory, root/kind, kind=kind, table=table,
                                     reliability_floor=.5, plots=True)
                    self.assertTrue(result)
            users = pd.read_csv(root/'config-sweep'/'summary_by_user_config.csv')
            self.assertEqual(users.user_id.tolist(), ['A', 'B'])
            self.assertEqual(users['Clicks per Character'].tolist(), [.625, 1.25])
            self.assertEqual(users['Completion Rate'].tolist(), [1., 1.])
            self.assertEqual(users['Correction Rate'].tolist(), [.5, .5])
            self.assertTrue((root/'config-sweep'/'sweep_report.md').exists())

    def test_comparison_entry_points_keep_paired_outcomes(self):
        base = pd.read_csv(FIXTURES/'global_input.csv')
        base = base[base.combo_id == base.combo_id.iloc[0]].copy()
        frames = []
        for system, label in [('og_nomon', 'OG Nomon'), ('oneclick', 'OneClick')]:
            rows = base.copy()
            rows['system'], rows['system_label'] = system, label
            rows['paired_offset_checksum'] = 'same'
            rows['num_clicks'] = 20 if system == 'og_nomon' else 10
            rows['num_corrections'] = rows['correction_rate_percent'] = rows['error_rate_percent'] = 0
            rows['Completion Fraction'] = rows.phrase_completed.astype(float)
            frames.append(rows)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/'source'
            source.mkdir()
            pd.concat(frames).to_csv(source/'input.csv', index=False)
            for kind in ('comparison', 'selected-comparison'):
                with self.subTest(kind=kind):
                    result = analyze(source, root/kind, kind=kind, table='input.csv',
                                     plots=kind == 'comparison')
                    self.assertTrue(result)
            paired = pd.read_csv(root/'selected-comparison'/'paired_results.csv')
            completed = paired[paired.paired_outcome == 'both_completed']
            self.assertTrue(completed.paired_click_reduction_percent.eq(50).all())
            self.assertTrue((root/'selected-comparison'/'comparison_report.md').exists())


if __name__ == '__main__': unittest.main()
