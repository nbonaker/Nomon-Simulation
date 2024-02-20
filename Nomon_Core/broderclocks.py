#!/usr/bin/env python2

from __future__ import division
from Nomon_Core.clock_inference_engine import ClockInference
from Nomon_Core import config
from numpy import log


class BroderClocks:
    def __init__(self, parent):
        self.parent = parent
        self.clock_inf = ClockInference(self.parent, self)
        
        self.is_undo = False
        self.is_equalize = False
        self.is_win = self.clock_inf.is_winner()
        self.is_start = False

        self.latest_time = self.parent.sim_time.time()
        self.last_press_time = self.parent.sim_time.time()

        self.last_gap_time_li = []
        self.last_press_time_li = []
        
        self.time_rotate = self.parent.time_rotate
    
    def get_histogram(self):
        return self.clock_inf.kde.dens_li

    def save_click_time(self, last_press_time, last_gap_time, index):
        self.click_time_list.append((last_gap_time, index))

    def select(self):

        time_in = self.parent.sim_time.time()

        # update scores of each clock
        self.clock_inf.update_scores(time_in - self.latest_time)
        # update history of key presses
        if config.is_learning:
            self.clock_inf.update_history(time_in - self.latest_time)

        # proceed based on whether there was a winner
        if (self.clock_inf.is_winner()):

            if self.parent.is_simulation:

                self.parent.winner = True
                self.parent.winner_text = self.parent.clock_to_text(self.clock_inf.sorted_inds[0])

            # record winner
            self.clock_inf.win_history[0] = self.clock_inf.sorted_inds[0]
            # update number of bits recorded
            self.clock_inf.entropy.update_bits()
            # call parent program with choice
            (self.clock_inf.clocks_on, self.clock_inf.clocks_off, clock_score_prior, self.is_undo,
             self.is_equalize) = self.parent.make_choice(self.clock_inf.sorted_inds[0])
            # learn new scores
            if config.is_learning:
                self.clock_inf.learn_scores(self.is_undo)

            # reset time indices
            self.init_round(True, False, clock_score_prior)
        else:
            # update time indices
            self.init_round(False, False, [])

    def init_bits(self):
        self.bits_per_select = log(len(self.clock_inf.clocks_on)) / log(2)
        self.start_time = self.parent.sim_time.time()

        self.last_win_time = self.start_time
        self.num_bits = 0
        self.num_selects = 0
            
    def init_follow_up(self, clock_score_prior):
        # initialize
        self.init_round(False, True, clock_score_prior)
        ## history of click times
        self.clock_inf.clock_history = [[]]
        self.clock_inf.win_history = [-1]
        # whether the previous move was an "undo"
        self.just_undid = False
        ## bit rate initialize
        self.init_bits()
        
    def init_round(self, is_win, is_start, clock_score_prior):
        self.clock_inf.clock_util.init_round(self.clock_inf.clocks_li)
        self.clock_inf.clock_util.init_round(self.clock_inf.clocks_on)
        if (is_win or is_start):  # if won, restart everything
            if (is_win):
                # identify the undo button as the winner to highlight
                win_clock = self.clock_inf.sorted_inds[0]

            if (self.is_undo) and (not self.is_equalize):
                count = 0
                for clock in self.clock_inf.clocks_on:
                    self.clock_inf.cscores[clock] = 0
                    count += 1
                top_score = 0
            else:
                count = 0
                for clock in self.clock_inf.clocks_on:
                    self.clock_inf.cscores[clock] = clock_score_prior[count]
                    count += 1
                top_score = 0
        

        # update the sorted loading_text
        self.clock_inf.update_sorted_inds()
        self.clock_inf.clock_util.update_curhours(self.clock_inf.sorted_inds)

        self.clock_inf.handicap_cscores(is_win, is_start)
