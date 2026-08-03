import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from OneClick_Core import config
from OneClick_Simulation.simulated_user import SimulatedUser
from OneClick_Text import kconfig
from OneClick_Text.keyboard import Keyboard


class SimulatedPressTimeTests(unittest.TestCase):
    def test_worked_example_is_point_six_three_seconds(self):
        delta = SimulatedUser._press_time_delta(
            cur_hour=0.25,
            ndt=1.0,
            period_s=2.2,
            click_offset_s=0.08,
        )

        self.assertAlmostEqual(delta, 0.63)

    def test_positive_offset_delays_and_negative_offset_advances(self):
        nominal = SimulatedUser._press_time_delta(0.25, 1.0, 2.0, 0.0)
        late = SimulatedUser._press_time_delta(0.25, 1.0, 2.0, 0.1)
        early = SimulatedUser._press_time_delta(0.25, 1.0, 2.0, -0.1)

        self.assertAlmostEqual(nominal, 0.5)
        self.assertAlmostEqual(late, 0.6)
        self.assertAlmostEqual(early, 0.4)

    def test_negative_delta_wraps_forward_one_period(self):
        delta = SimulatedUser._press_time_delta(
            cur_hour=0.49,
            ndt=1.0,
            period_s=2.0,
            click_offset_s=-0.08,
        )

        self.assertAlmostEqual(delta, 1.94)
        self.assertGreaterEqual(delta, 0.0)
        self.assertTrue(math.isfinite(delta))


class FixedPeriodSimulationTests(unittest.TestCase):
    def test_keyboard_constructs_all_clock_utilities_at_fixed_period(self):
        period = float(config.period_li[10])
        with patch("OneClick_Text.keyboard.LanguageModel") as language_model:
            language_model.return_value.get_key_probs.return_value = [
                0.0
            ] * len(kconfig.key_chars)
            keyboard = Keyboard(None, {"fixed_clock_period_s": period})

        self.assertAlmostEqual(keyboard.time_rotate, period)
        self.assertAlmostEqual(
            keyboard.bc.clock_inf.clock_util.time_rotate,
            period,
        )
        self.assertAlmostEqual(keyboard.word_clock_util.time_rotate, period)

    def test_keyboard_supports_independent_space_and_enter_periods(self):
        with patch("OneClick_Text.keyboard.LanguageModel") as language_model:
            language_model.return_value.get_key_probs.return_value = [
                0.0
            ] * len(kconfig.key_chars)
            keyboard = Keyboard(
                None,
                {
                    "fixed_space_clock_period_s": 1.5,
                    "fixed_enter_clock_period_s": 3.3,
                },
            )

        self.assertAlmostEqual(keyboard.time_rotate, 1.5)
        self.assertAlmostEqual(
            keyboard.bc.clock_inf.clock_util.time_rotate,
            1.5,
        )
        self.assertAlmostEqual(keyboard.word_clock_util.time_rotate, 3.3)

    def test_specialized_period_parameters_must_be_paired_and_not_mixed(self):
        with self.assertRaisesRegex(ValueError, "must be supplied together"):
            Keyboard(None, {"fixed_space_clock_period_s": 1.5})
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            Keyboard(
                None,
                {
                    "fixed_clock_period_s": 2.2,
                    "fixed_space_clock_period_s": 1.5,
                    "fixed_enter_clock_period_s": 3.3,
                },
            )

    def test_click_row_period_cannot_change_fixed_period(self):
        sim = SimulatedUser()
        sim.fixed_clock_period_s = 2.2
        sim.keyboard = SimpleNamespace(
            time_rotate=2.2,
            change_speed=lambda: self.fail("fixed-period mode changed speed"),
        )

        sim._apply_period({"Clock Period (s)": 1.5})

        self.assertEqual(sim.keyboard.time_rotate, 2.2)

    def test_press_actions_use_their_own_fixed_periods(self):
        sim = SimulatedUser()
        click = {
            "Clock Period (s)": 6.0,
            "Click Time Relative (s)": 0.0,
            "Dead Time (s)": 0.0,
        }
        sim.click_util = SimpleNamespace(sample=lambda: click)
        sim.fixed_clock_period_s = None
        sim.fixed_space_clock_period_s = 1.5
        sim.fixed_enter_clock_period_s = 3.0
        sim.num_clicks_phrase = 0
        sim.num_letter_presses_phrase = 0
        sim.letter_clock_time_s = 0.0
        sim.target_enter_clock_time_s = 0.0
        sim.undo_clock_time_s = 0.0
        sim_time = SimpleNamespace(cur=0.0)
        sim_time.time = lambda: sim_time.cur
        sim_time.set_time = lambda value: setattr(sim_time, "cur", value)
        letter_util = SimpleNamespace(
            num_divs_time=1.0,
            cur_hours={4: 0.25},
        )
        word_util = SimpleNamespace(
            num_divs_time=1.0,
            cur_hours={7: 0.25},
            time_rotate=3.0,
        )
        sim.keyboard = SimpleNamespace(
            time_rotate=1.5,
            sim_time=sim_time,
            bc=SimpleNamespace(
                clock_inf=SimpleNamespace(clock_util=letter_util)
            ),
            word_clock_util=word_util,
            increment_clocks=lambda: None,
            increment_word_clocks=lambda: None,
            on_press=lambda: None,
            on_enter=lambda: (None, 7),
        )

        sim._press_letter(4)
        sim._press_enter(7)
        sim._press_enter(7, action_kind="undo")

        self.assertAlmostEqual(sim.letter_clock_time_s, 0.375)
        self.assertAlmostEqual(sim.target_enter_clock_time_s, 0.75)
        self.assertAlmostEqual(sim.undo_clock_time_s, 0.75)
        self.assertAlmostEqual(sim.keyboard.sim_time.time(), 1.875)

    def test_dead_time_full_rotations_use_active_action_period(self):
        sim = SimulatedUser()
        click = {
            "Click Time Relative (s)": 0.05,
            "Dead Time (s)": 5.5,
        }

        letter_offset, letter_rotations = sim._click_components(click, 2.0)
        enter_offset, enter_rotations = sim._click_components(click, 3.0)

        self.assertEqual(letter_offset, enter_offset)
        self.assertAlmostEqual(letter_rotations, 4.0)
        self.assertAlmostEqual(enter_rotations, 3.0)


if __name__ == "__main__":
    unittest.main()
