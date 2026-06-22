from __future__ import division
from numpy import pi, ceil
from OneClick_Core import config


class HourLocs:
    def __init__(self, num_divs_time):
        self.num_divs_time = num_divs_time
        self.hour_locs = []
        for index in range(num_divs_time):
            base = -pi / 2.0 + (2.0 * pi * index) / num_divs_time
            theta = -config.theta0 + base
            self.hour_locs.append([theta])


# Spreads clock phases so top-ranked clocks are maximally far apart.
class SpacedArray:
    def __init__(self, nels):
        rev_arr = []
        insert_pt = 0
        level = 0
        for index in range(nels):
            rev_arr.insert(insert_pt, index + 1)
            insert_pt += 2
            if insert_pt > 2 * (2 ** level - 1):
                insert_pt = 0
                level += 1
        rev_arr.insert(0, 0)

        self.arr = [0] * (nels + 1)
        for index in range(nels + 1):
            self.arr[rev_arr[index]] = index


class ClockUtil:
    def __init__(self, parent, bc, clock_inf):
        self.parent = parent
        self.bc = bc
        self.clock_inf = clock_inf

        self.cur_hours = [0.0] * len(self.parent.clock_centers)
        self.time_rotate = self.parent.time_rotate
        self.num_divs_time = int(ceil(self.parent.time_rotate / config.ideal_wait_s))
        self.spaced = SpacedArray(self.num_divs_time)
        self.hl = HourLocs(self.num_divs_time)

    def update_curhours(self, update_clocks_list):
        for count, sind in enumerate(update_clocks_list):
            self.cur_hours[sind] = self.spaced.arr[count % self.num_divs_time]

    def change_period(self, new_period):
        self.time_rotate = new_period
        self.clock_inf.time_rotate = new_period
        self.num_divs_time = int(ceil(self.time_rotate / config.ideal_wait_s))
        self.hl = HourLocs(self.num_divs_time)
        self.spaced = SpacedArray(self.num_divs_time)
        self.init_round(self.clock_inf.clocks_li)

    def increment(self, time_diff):
        steps = int(time_diff / self.time_rotate * self.num_divs_time)
        for clock in range(len(self.cur_hours)):
            self.cur_hours[clock] = (self.cur_hours[clock] + steps) % self.num_divs_time

    def init_round(self, clock_index_list):
        self.update_curhours(clock_index_list)
