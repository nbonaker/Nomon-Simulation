from __future__ import division
from OneClick_Core.clock_inference_engine import ClockInference
from OneClick_Core import config


class BroderClocks:


    def __init__(self, parent):
        self.parent = parent
        self.clock_inf = ClockInference(parent, self)
        self.time_rotate = parent.time_rotate
        self.latest_time = parent.sim_time.time()

    def select(self):
        """Called on each simulated Space press."""
        time_in = self.parent.sim_time.time()
        time_diff = time_in - self.latest_time
        self.latest_time = time_in
        self.clock_inf.add_click(time_diff)
        # NOTE: letter clocks are NOT respaced per press (cf. oneclick/broderclocks.js
        # select(), which only adds the click). They are placed once per word from the
        # LM letter prior (Keyboard.place_letter_clocks) and then just tick.

    def init_follow_up(self):
        self.clock_inf.clock_util.init_round(self.clock_inf.clocks_li)

    def change_speed(self):
        self.time_rotate = self.parent.time_rotate
        self.clock_inf.clock_util.change_period(self.time_rotate)
