#!/usr/bin/python

######################################
# Copyright 2019 Nicholas Bonaker, Keith Vertanen, Emli-Mari Nel, Tamara Broderick
# This file is part of the Nomon software.
# Nomon is free software: you can redistribute it and/or modify it
# under the terms of the MIT License reproduced below.
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY
# OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
# LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
# EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR
#OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
#
# <https://opensource.org/licenses/mit-license.html>
######################################

from Nomon_Core import config
from Nomon_Simulation.keyboard_symbol import Keyboard
from Nomon_Simulation.text_stats import calc_MSD
import pandas as pd
import numpy as np
import sys
import os
import csv


class SimulatedUser:
    def __init__(self, cwd=os.getcwd(), job_num=None, custom_keys=None, num_jobs=0):

        self.cwd = os.getcwd()

        self.job_num = job_num
        self.num_jobs = num_jobs

        self.working_dir=cwd

        self.initialize_keyboard()

    def init_sim_data(self):
        # self.init_clocks()
        self.num_selections = 0
        self.sel_per_min = []

        self.num_chars = 0
        self.char_per_min = []

        self.num_words = 0

        self.num_presses = 0
        self.press_per_sel = []
        self.press_per_char = []
        self.press_per_word = []

        self.num_errors = 0
        self.error_rate_avg = []
        self.kde_errors = []
        self.kde_errors_avg = None

        # self.on_timer()
        self.winner = False
        self.winner_text = ""

    def initialize_keyboard(self):
        self.keyboard = Keyboard()

    def get_session_clicks(self):
        session_click_df = self.click_df[self.click_df["Session Num"] == self.session]

        self.session_click_times = list(session_click_df[["Click Time Relative (s)"]].to_numpy().T[0])
        self.session_dead_times = list(session_click_df[["Dead Time (s)"]].to_numpy().T[0])
        self.session_time_rotates = list(session_click_df[["Clock Period (s)"]].to_numpy().T[0])


    def parameter_metrics(self, parameters, num_clicks=500, trials=1, attribute=None):
        self.init_sim_data()
        # Load parameters or use defaults

        click_df = parameters["click_df"]
        self.calibration_clicks = click_df[click_df["Session Num"].isna()][["Click Time Relative (s)"]].to_numpy().T[0]
        self.click_df = click_df[click_df["Session Num"].notna()]

        self.num_sessions = int(click_df["Session Num"].max())

        if "time_rotate" in parameters:
            self.keyboard.rotate_index = parameters["time_rotate"]
            self.keyboard.time_rotate = config.period_li[self.keyboard.rotate_index]
            self.keyboard.change_speed()
        else:
            self.keyboard.rotate_index = config.default_rotate_ind
            self.keyboard.time_rotate = config.period_li[self.keyboard.rotate_index]

        full_results = []

        for trial in range(trials):
            self.initialize_keyboard()

            ####### prime kde with calibration data
            for yin in self.calibration_clicks:
                self.keyboard.bc.clock_inf.clock_util.clock_inf.kde.add_point(float(yin))

            for session in range(1, self.num_sessions+1):
                self.session = session

                self.update_progress_bar(trial, trials, session, self.num_sessions)

                self.get_session_clicks()

                # self.keyboard.rotate_index = int(self.click_dist_copy[0])
                self.keyboard.time_rotate = config.period_li[self.keyboard.rotate_index]
                # print(self.time_rotate)
                self.keyboard.change_speed()

                self.num_presses_total = 0
                self.phrase_num = 0

                while len(self.session_click_times) > 0:
                    target_phrase = [self.keyboard.key_chars[i] for i in np.random.randint(0, len(self.keyboard.key_chars)-3, 7)]

                    phrase_results = {}

                    self.num_presses = 0
                    self.num_corrections = 0
                    self.num_selections = 0
                    self.start_time = self.keyboard.sim_time.time()

                    self.phrase_num +=1
                    self.type_text(target_phrase, verbose=False)

                    self.num_presses_total += self.num_presses

                    if len(self.keyboard.typed) > 3 and self.num_selections > 0:
                        self.num_chars += int(len(self.keyboard.typed))
                        self.num_words += int(len(self.keyboard.typed))
                        self.num_errors += calc_MSD(self.keyboard.typed, target_phrase[:len(self.keyboard.typed)])[0]

                        phrase_results["session"] = session+1
                        phrase_results["trial"] = trial
                        phrase_results["phrase"] = self.phrase_num
                        phrase_results["click_load"] = self.num_presses/min(5, len(self.keyboard.typed))
                        phrase_results["entry_rate"] = min(5, len(self.keyboard.typed))/(self.keyboard.sim_time.time()-self.start_time)*60
                        phrase_results["correction_rate"] = self.num_corrections/self.num_selections
                        phrase_results["num_chars"] = len(self.keyboard.typed)

                        full_results.append(phrase_results)

                    # print(calc_MSD(self.typed, text))
                    #
                    # print(self.typed)
                    self.keyboard.typed = ""  # reset tracking and context for lm -- new sentence

                ########### reset clock scores
                (self.keyboard.bc.clock_inf.clocks_on, self.keyboard.bc.clock_inf.clocks_off, clock_score_prior, self.keyboard.bc.is_undo,
                 self.keyboard.bc.is_equalize) = self.keyboard.make_choice(self.keyboard.bc.clock_inf.sorted_inds[0])
                # learn new scores
                if config.is_learning:
                    self.keyboard.bc.clock_inf.learn_scores(self.keyboard.bc.is_undo)
                # reset time indices
                self.keyboard.bc.init_round(True, False, clock_score_prior)
                self.keyboard.typed = ""

        print("\n\n")

        self.result_df = pd.DataFrame(full_results)

    def update_progress_bar(self, trial, trials, session, num_sessions):
        if self.num_jobs == 0:
            progress = int((trial + session / num_sessions) / trials * 50)
        else:
            progress = int(((trial + session / num_sessions) / trials + self.job_num)/self.num_jobs * 50)

        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("[%-50s] %d%%" % ('=' * progress, 2 * progress))
        sys.stdout.flush()

    def update_sim_averages(self, num_trials):

        time_int = self.keyboard.sim_time.time() - self.prev_time
        self.prev_time = float(self.keyboard.sim_time.time())

        self.sel_per_min += [self.num_selections / (time_int / 60)]

        self.char_per_min += [self.num_chars / (time_int / 60)]

        if self.num_selections > 0:
            self.press_per_sel += [self.num_presses / self.num_selections]

            self.error_rate_avg += [self.num_errors / self.num_selections]
        else:
            self.press_per_sel += [float("inf")]

            self.error_rate_avg += [float("inf")]

        if self.num_chars > 0:
            self.press_per_char += [self.num_presses / self.num_chars]
        else:
            self.press_per_char += [float("inf")]

        if self.num_words > 0:
            self.press_per_word += [self.num_presses / self.num_words]
        else:
            self.press_per_word += [float("inf")]

    def type_text(self, text, verbose=False):
        self.target_text = text
        prev_target_text = self.target_text
        success = True

        while len(self.target_text) > 0 or not success and len(self.session_click_times) > 0:
            if success:
                prev_target_text = self.target_text
                target_clock, self.target_text = self.next_target(self.target_text)
            else:
                target_clock, self.target_text = self.next_target(prev_target_text)

            self.target_clock = target_clock

            if verbose:
                print("Target: ", self.keyboard.clock_to_text(target_clock), target_clock)

            success = self.select_clock(target_clock, verbose=verbose)
            if success is not None:
                # return

                if not success:
                    enter_press_num = self.num_presses
                    undo_depth = 1
                    while undo_depth > 0 and undo_depth <= 3:

                        undo_clock = len(self.keyboard.key_chars)-1
                        if verbose:
                            print("Target: ", "Undo ", undo_clock)
                        self.num_corrections += 1
                        undo_success = self.select_clock(undo_clock, verbose=verbose, undo_depth=undo_depth)

                        if undo_success is not None:
                            if undo_success:
                                undo_depth -= 1
                            elif success == "END":
                                return
                            else:
                                undo_depth += 1

                    if (undo_depth > 2):
                        success = True
                    else:
                        success = False

                if success == "END":
                    return

    def select_clock(self, target_clock, verbose=False, undo_depth=0):

        ndt = self.keyboard.bc.clock_inf.clock_util.num_divs_time
        num_press = 0
        time_elapsed = 0

        cur_clicks = []
        initial_dens_li = self.keyboard.bc.clock_inf.kde.dens_li
        for i in range(15):
            self.keyboard.winner = False
            if self.keyboard.bc.clock_inf.clock_util.cur_hours[target_clock] > ndt / 2:
                time_delta = (ndt * 3 / 2 - self.keyboard.bc.clock_inf.clock_util.cur_hours[
                    target_clock]) / ndt * self.keyboard.time_rotate
            else:
                time_delta = (ndt / 2 - self.keyboard.bc.clock_inf.clock_util.cur_hours[target_clock]) / ndt * self.keyboard.time_rotate

            if len(self.session_click_times) == 0:
                return "END"
            else:
                click_offset = float(self.session_click_times.pop())
                delay_offset = float(self.session_dead_times.pop())
                time_rotate = float(self.session_time_rotates.pop())
                if self.keyboard.time_rotate != time_rotate:
                    self.keyboard.time_rotate = time_rotate
                    self.keyboard.change_speed()
                    # print(self.keyboard.time_rotate)

            cur_clicks += [click_offset]

            # simulate gaze point
            # if self.gaze_scale == 0:
            #     self.gaze_x_loc = self.gaze_y_loc = 0
            # else:
            #     target_x, target_y = self.clock_locs[self.target_clock]
            #     self.gaze_x_loc = np.random.normal(target_x, self.gaze_scale*self.clock_radius/2)
            #     self.gaze_y_loc = np.random.normal(target_y, self.gaze_scale*self.clock_radius/2)


            full_rotation_factor = (delay_offset//self.keyboard.time_rotate)
            time_delta += full_rotation_factor*self.keyboard.time_rotate

            time_delta += click_offset

            time_elapsed += time_delta
            self.keyboard.sim_time.set_time(self.keyboard.sim_time.time() + time_delta)

            self.keyboard.increment_clocks()

            self.keyboard.on_press()
            self.num_presses += 1

            # recovery_time = 0.4
            # self.time.set_time(self.time.time() + recovery_time)
            #             # self.on_timer()
            if self.keyboard.winner:

                selected_clock = self.keyboard.previous_winner

                if verbose:
                    if undo_depth > 0:
                        tab = "    "
                    else:
                        tab = ""
                    print(tab + "    Typed \"" + self.keyboard.typed + "\"")
                self.keyboard.winner = False

                if selected_clock == target_clock:
                    self.num_selections += 1
                else:
                    return False
                return True

        return None

    def next_target(self, text):

        target_letter = text[0]

        return self.keyboard.key_chars.index(target_letter)*(self.keyboard.N_pred+1) + self.keyboard.N_pred, text[1:]



def main():
    with open('simulations/ajay_data/click_times_user_63.csv', newline='') as f:
        reader = csv.reader(f)
        user_data = list(reader)

    data_len = len(user_data)
    click_dists = user_data[:data_len//2]
    delay_dists = user_data[data_len//2:]

    click_dists = click_dists[-5:]
    delay_dists = delay_dists[-5:]


    sim = SimulatedUser()
    params = { "time_rotate": 14, "click_dist": click_dists, "delay_dist": delay_dists}

    sim.parameter_metrics(params, trials=1)
    print(sim.result_df)
    sim.result_df.to_csv('simulations/ajay_data/test_df_opt.csv', index=False)



if __name__ == "__main__":
    main()
