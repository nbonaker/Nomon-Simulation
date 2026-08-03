import unittest

from OneClick_Text.keyboard import WordClockUtil


class FixedDelayModel:
    def __init__(self, offset):
        self.offset = offset

    def get_offset(self):
        return self.offset


class WordClockOffsetTests(unittest.TestCase):
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
