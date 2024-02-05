#!/usr/bin/python

import numpy as np
import os
from Nomon_Core import config
from Nomon_Core.broderclocks import BroderClocks
from Nomon_Symbol import kconfig
from matplotlib import pyplot as plt


class SimTime:
    def __init__(self):
        self.cur_time = 0

    def time(self):
        return self.cur_time

    def set_time(self, t):
        self.cur_time = t


class Keyboard:
    def __init__(self, cwd=os.getcwd(), job_num=None, sub_call=False, custom_keys=None, num_jobs=0):

        self.sim_time = SimTime()
        self.is_simulation = True

        self.N_pred = 0
        # self.prob_thres = kconfig.prob_thres

        self.win_diff_base = config.win_diff_base
        self.rotate_index = config.default_rotate_ind
        self.time_rotate = config.period_li[self.rotate_index]

        if custom_keys is not None:
            self.key_chars = custom_keys
        else:
            self.key_chars = kconfig.emoji_keys

        # self.time = SimTime(self)
        self.prev_time = 0

        self.word_pred_on = 0

        # determine keyboard positions
        self.init_clock_locs()

        # set up "typed" text
        self.typed = ""
        self.btyped = ""
        self.context = ""
        self.last_add_li = [0]

        self.init_words()

        self.clear_text = False

        # generate prior for clocks
        self.gen_clock_prior(False)

        self.clock_spaces = np.zeros((len(self.clock_centers), 2))

        self.bc = BroderClocks(self)

        self.bc.init_follow_up(self.word_score_prior)

        self.clock_params = np.zeros((len(self.clock_centers), 8))

        self.consent = False

    def clock_to_text(self, index):
        typed = self.key_chars[index]
        return typed

    def init_clock_locs(self):
        y_spacing = kconfig.base_window_height / kconfig.num_rows
        x_spacing = kconfig.base_window_width / kconfig.num_cols

        clock_radius = min(y_spacing / 3.5, x_spacing / 6.0);

        # calculate the clock centers for the display
        self.clock_centers = []

        # for i in range(0, 7):
        #     for j in range(0, 10):
        #         if i == 6 and j % 2:
        #             continue
        #         self.clock_centers.append((x_spacing * (j) + clock_radius * 2, y_spacing * (i) + clock_radius * 2))
        for row_num, row in enumerate(kconfig.emoji_target_layout):
            for col_num in range(len(row)):
                if len(row) < kconfig.num_cols:
                    self.clock_centers.append((x_spacing * (col_num*2) + clock_radius * 2, y_spacing * (row_num) + clock_radius * 2))
                else:
                    self.clock_centers.append((x_spacing * (col_num) + clock_radius * 2, y_spacing * (row_num) + clock_radius * 2))

        self.N_keys = len(self.key_chars)
        self.win_diffs = []
        for key_char in self.key_chars:
            if (key_char == kconfig.mybad_char):
                self.win_diffs.append(config.win_diff_high)
            else:
                self.win_diffs.append(self.win_diff_base)

    def change_speed(self):
        self.bc.clock_inf.clock_util.change_period(self.time_rotate)

    def init_words(self):
        self.key_freq_li = np.array([np.log(1 / len(self.key_chars)) for i in range(len(self.key_chars))])

        self.word_pair = []
        self.words_on = [index for index in range(0, self.N_keys)]
        self.words_off = []

        index=0
        for key in range(0, self.N_keys):
            # self.words_on.append(index)
            self.word_pair.append((key,))
            index += 1

    def gen_clock_prior(self, is_undo):
        self.word_score_prior = []
        N_on = len(self.words_on)
        # print(N_on)
        if is_undo:
            for index in self.words_on:
                pair = self.word_pair[index]
                if len(pair) == 1:
                    key = pair[0]
                    if (self.key_chars[key] == kconfig.mybad_char):
                        prob = kconfig.undo_prob
                        self.word_score_prior.append(np.log(prob))
                    else:
                        self.word_score_prior.append(0)
                else:
                    self.word_score_prior.append(0)
        else:
            for index in self.words_on:
                pair = self.word_pair[index]

                key = pair[0]
                prob = self.key_freq_li[key]
                prob = prob + np.log(kconfig.rem_prob)
                if self.key_chars[key] == kconfig.mybad_char:
                    prob = np.log(kconfig.undo_prob)

                self.word_score_prior.append(prob)

    def increment_clocks(self):
        self.bc.clock_inf.clock_util.increment(self.words_on)

    def on_press(self):
        self.bc.select()

    def make_choice(self, index):
        is_undo = False
        is_equalize = False

        self.winner = True
        self.previous_winner = index
        # if selected a key
        # new_char = self.keys_li[self.index_to_wk[index]]
        new_char = self.key_chars[index]

        # special characters
        if new_char == kconfig.mybad_char: # selected undo

            # if added characters that turn
            if len(self.last_add_li) > 1:
                last_add = self.last_add_li.pop()

                if last_add > 0:  # if added text that turn
                    self.typed = self.typed[0:-last_add]
                elif last_add == -1:  # if backspaced that turn
                    letter = self.btyped[-1]
                    self.btyped = self.btyped[0:-1]
                    self.typed += letter
            new_char = ''
            is_undo = True

        else: # selected an emoji character
            self.last_add_li.append(1)
            self.typed += new_char

        self.init_words()
        # update the clock priors
        self.gen_clock_prior(is_undo)

        return self.words_on, self.words_off, self.word_score_prior, is_undo, is_equalize
