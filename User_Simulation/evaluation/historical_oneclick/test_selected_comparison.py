from __future__ import annotations
import unittest
import numpy as np
import pandas as pd
from User_Simulation.evaluation.historical_oneclick.selected_comparison import build_curve_points, build_paired_results, select_heldout_phrases, validate_condition

def _audit_row(phrase_id: str, length: int) -> dict:
    return {'phrase_id': phrase_id, 'phrase_text': f'phrase {phrase_id}', 'all_words_prediction_reachable': True, 'target_character_count': length}

class HeldoutPhraseTests(unittest.TestCase):

    def test_selection_is_deterministic_and_excludes_tuning_phrases(self):
        audit = pd.DataFrame([_audit_row(f'real_{index:03d}', index) for index in range(1, 9)])
        tuning = pd.DataFrame({'phrase_id': ['real_002', 'real_006']})
        first = select_heldout_phrases(audit, tuning, phrase_count=3, seed=54321)
        second = select_heldout_phrases(audit, tuning, phrase_count=3, seed=54321)
        self.assertTrue(first.equals(second))
        self.assertFalse(set(first['phrase_id']) & set(tuning['phrase_id']))
        self.assertEqual(first['Comparison Phrase ID'].tolist(), ['heldout_phrase_01', 'heldout_phrase_02', 'heldout_phrase_03'])

class CompletionSummaryTests(unittest.TestCase):

    def test_failed_phrases_remain_in_curve_denominator(self):
        frame = pd.DataFrame([{'user_id': 'A', 'system': system, 'system_label': label, 'phrase_completed': True, 'simulated_completion_time_s': 30.0} for system, label in [('og_nomon', 'OG Nomon'), ('oneclick', 'OneClick')]] + [{'user_id': 'A', 'system': system, 'system_label': label, 'phrase_completed': False, 'simulated_completion_time_s': np.nan} for system, label in [('og_nomon', 'OG Nomon'), ('oneclick', 'OneClick')]])
        curves = build_curve_points(frame)
        endpoints = curves[(curves['scope_user_id'] == 'A') & (curves['event_type'] == 'endpoint')]
        self.assertTrue(endpoints['phrase_attempts'].eq(2).all())
        self.assertTrue(endpoints['cumulative_completion_rate'].eq(0.5).all())

    def test_paired_metrics_only_use_mutually_completed_trials(self):
        rows = []
        for system in ['og_nomon', 'oneclick']:
            rows.extend([{'user_id': 'A', 'trial': 0, 'system': system, 'Comparison Phrase ID': 'p1', 'Target Phrase': 'one', 'paired_click_schedule_id': 'A_trial_00', 'paired_offset_checksum': 'same', 'phrase_completed': True, 'num_clicks': 20 if system == 'og_nomon' else 10, 'num_corrections': 0, 'correction_rate_percent': 0.0, 'simulated_attempt_time_s': 50.0 if system == 'og_nomon' else 40.0, 'simulated_completion_time_s': 50.0 if system == 'og_nomon' else 40.0, 'phrase_failure_reason': ''}, {'user_id': 'A', 'trial': 0, 'system': system, 'Comparison Phrase ID': 'p2', 'Target Phrase': 'two', 'paired_click_schedule_id': 'A_trial_00', 'paired_offset_checksum': 'same', 'phrase_completed': system == 'og_nomon', 'num_clicks': 30 if system == 'og_nomon' else 5, 'num_corrections': 0, 'correction_rate_percent': 0.0, 'simulated_attempt_time_s': 60.0, 'simulated_completion_time_s': 60.0 if system == 'og_nomon' else np.nan, 'phrase_failure_reason': '' if system == 'og_nomon' else 'word_attempts_exhausted'}])
        paired = build_paired_results(pd.DataFrame(rows))
        self.assertEqual(paired['paired_outcome'].tolist(), ['both_completed', 'og_only'])
        self.assertAlmostEqual(paired.loc[0, 'paired_click_reduction_percent'], 50.0)
        self.assertTrue(np.isnan(paired.loc[1, 'paired_click_reduction_percent']))

class ConditionValidationTests(unittest.TestCase):

    def test_exact_text_and_oneclick_time_accounting_are_enforced(self):
        frame = pd.DataFrame([{'user_id': 'A', 'trial': 0, 'system': 'oneclick', 'Comparison Phrase ID': 'p1', 'Target Phrase': 'exact phrase', 'Typed Text': 'exact phrase ', 'phrase_completed': True, 'simulated_attempt_time_s': 6.0, 'simulated_completion_time_s': 6.0, 'paired_click_schedule_id': 'A_trial_00', 'paired_offset_checksum': 'checksum', 'space_clock_period_s': 6.0, 'enter_clock_period_s': 6.0, 'letter_clock_time_s': 2.0, 'target_enter_clock_time_s': 3.0, 'undo_clock_time_s': 1.0}])
        validate_condition(frame, 'A', 'oneclick', 0, {'p1'}, space_period_s=6.0, enter_period_s=6.0)
        frame.loc[0, 'Typed Text'] = 'wrong phrase'
        with self.assertRaisesRegex(ValueError, 'exactly match'):
            validate_condition(frame, 'A', 'oneclick', 0, {'p1'}, space_period_s=6.0, enter_period_s=6.0)
