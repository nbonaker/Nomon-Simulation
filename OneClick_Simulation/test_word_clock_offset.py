import unittest

from OneClick_Core import config
from OneClick_Core.clock_inference_engine import UserDelayModel
from OneClick_Text.keyboard import WordClockUtil


class FixedDelayModel:
    def __init__(self, offset):
        self.offset = offset

    def get_offset(self):
        return self.offset


class WordClockOffsetTests(unittest.TestCase):
    @staticmethod
    def _trained_delay_model(use_click_offset):
        model = UserDelayModel(use_click_offset=use_click_offset)
        for _ in range(config.bootstrap_n):
            model.update(0.2)
        return model

    def test_disabled_offset_returns_zero_after_learning(self):
        model = self._trained_delay_model(use_click_offset=False)

        self.assertEqual(model.get_offset(), 0.0)

    def test_enabled_offset_uses_learned_mean_after_bootstrap(self):
        model = self._trained_delay_model(use_click_offset=True)

        self.assertNotEqual(model.get_offset(), 0.0)
        self.assertEqual(model.get_offset(), model.mu)

    def test_confirmed_batch_rolls_back_atomically(self):
        model = UserDelayModel(use_click_offset=True)
        initial = (model.mu, model.sigma2, model.n_samples)

        model.update_many((0.1, 0.2, 0.3))
        self.assertEqual(model.n_samples, 3)

        model.rollback()
        self.assertEqual((model.mu, model.sigma2, model.n_samples), initial)

    def test_enabled_offset_waits_for_bootstrap_samples(self):
        model = UserDelayModel(use_click_offset=True)
        for _ in range(config.bootstrap_n - 1):
            model.update(0.2)

        self.assertEqual(model.get_offset(), 0.0)

    def test_word_selection_compensates_for_shared_switch_delay(self):
        util = WordClockUtil(time_rotate=4.0, delay_model=FixedDelayModel(0.2))
        target = 1
        earlier_neighbor = 2
        util.cur_hours = {
            target: util.num_divs_time // 2,
            earlier_neighbor: util.num_divs_time // 2 - 3,
        }

        selected = util.select_word(0.2, [target, earlier_neighbor])

        self.assertEqual(selected, target)


if __name__ == "__main__":
    unittest.main()
