from __future__ import division
from OneClick_Core.clock_inference_engine import ClockInference
from OneClick_Core import config


class BroderClocks:


    def __init__(self, parent):
        self.parent = parent
        self.clock_inf = ClockInference(parent, self)
        self.time_rotate = parent.time_rotate
        self.latest_time = parent.sim_time.time()

    def select(self, target_index=None):
        """Called on each simulated Space press."""
        time_in = self.parent.sim_time.time()
        time_diff = time_in - self.latest_time
        self.latest_time = time_in
        target_time_in = self.clock_inf.add_click(time_diff, target_index)
        # Post-observation rephasing, when enabled, is owned by Keyboard so test-only
        # observation replacement happens before the phases are rebuilt.
        return target_time_in

    def init_follow_up(self):
        self.clock_inf.clock_util.init_round(self.clock_inf.clocks_li)

    def change_speed(self):
        self.time_rotate = self.parent.time_rotate
        self.clock_inf.clock_util.change_period(self.time_rotate)
