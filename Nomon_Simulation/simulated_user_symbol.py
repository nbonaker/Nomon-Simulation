#!/usr/bin/python

from Nomon_Core import config
from Nomon_Symbol.keyboard import Keyboard
from Nomon_Symbol import kconfig
from Nomon_Simulation import sim_config
from Nomon_Symbol.text_stats import calc_MSD
import pandas as pd
import numpy as np
import sys
import os


class ClickUtil:
    def __init__(self, parent, click_df, calibration_clicks, type):
        self.parent = parent
        self.calibration_clicks = calibration_clicks
        self.click_df = click_df
        self.shuffle_indices = np.array([])
        self.playthrough_index = 0
        self.clicks_remaining = self.click_df.shape[0]
        self.reshuffle()

        self.type = type

    def reshuffle(self):
        self.shuffle_indices = np.arange(self.click_df.shape[0])
        np.random.shuffle(self.shuffle_indices)

    def sample(self):
        if self.type == "playthrough":
            if self.playthrough_index < self.click_df.shape[0]:
                cur_click = self.click_df.iloc[self.playthrough_index]
                self.playthrough_index += 1
                self.clicks_remaining -= 1
                return cur_click

        elif self.type == "shuffle":
            if len(self.shuffle_indices) == 0:
                self.reshuffle()

            sample_index, self.shuffle_indices = self.shuffle_indices[0], self.shuffle_indices[1:]
            return self.click_df.loc[sample_index]

        if self.type == "loop":
            if self.playthrough_index == self.click_df.shape[0]:
                self.playthrough_index = 0

                # prime kde with calibration data before the first session
                self.parent.keyboard.bc.clock_inf.clock_util.clock_inf.kde.initialize_dens()
                for yin in self.calibration_clicks:
                    self.parent.keyboard.bc.clock_inf.clock_util.clock_inf.kde.add_point(float(yin))

            if self.playthrough_index < self.click_df.shape[0]:
                cur_click = self.click_df.iloc[self.playthrough_index]
                self.playthrough_index += 1
                # self.clicks_remaining -= 1
                return cur_click

class SimulatedUser:
    def __init__(self, cwd=os.getcwd(), job_num=None, custom_keys=None, num_jobs=0):
        # used for tracking overall progressbar
        self.job_num = job_num
        self.num_jobs = num_jobs

        self.working_dir=cwd

        # initialize the keyboard
        self.keyboard = Keyboard()
        print("Initializing Keyboard with the following layout:")
        for row in kconfig.emoji_target_layout:
            print(row)

    def init_sim_data(self):
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

        self.winner = False
        self.winner_text = ""

    def get_session_clicks(self):
        session_click_df = self.click_df[self.click_df["Session Num"] == self.session]

        self.session_click_times = list(session_click_df[["Click Time Relative (s)"]].to_numpy().T[0])
        self.session_dead_times = list(session_click_df[["Dead Time (s)"]].to_numpy().T[0])
        self.session_time_rotates = list(session_click_df[["Clock Period (s)"]].to_numpy().T[0])

    def parameter_metrics(self, parameters, trials=1, verbose=False):
        self.init_sim_data()

        click_df = parameters["click_df"]
        self.calibration_clicks = click_df[click_df["Session Num"].isna()][["Click Time Relative (s)"]].to_numpy().T[0]
        self.click_df = click_df[click_df["Session Num"].notna()]
        # used in progress bar to track simulation progress
        self.num_clicks_loaded = len(self.click_df["Click Time Relative (s)"])
        self.num_clicks_total = 0

        # get session numbers from click data frame
        self.sessions_li = pd.unique(self.click_df["Session Num"])
        self.num_sessions = int(self.sessions_li.max()) - int(self.sessions_li.min())

        # list to store sim results
        full_results = []

        for trial in range(trials):
            # reinitialize the keyboard for each trial
            self.keyboard = Keyboard()

            # prime kde with calibration data before the first session
            for yin in self.calibration_clicks:
                self.keyboard.bc.clock_inf.clock_util.clock_inf.kde.add_point(float(yin))

            # run through each session in the user data
            for session in range(1, self.num_sessions+1):
                self.session = session

                self.update_progress_bar(trial, trials, session, self.num_sessions)

                # get the click data for the current session
                self.get_session_clicks()

                self.num_presses_total = 0
                self.phrase_num = 0

                # type phrases until all clicks are used
                while len(self.session_click_times) > 0:
                    # construct a target phrase of 5 symbols followed by two periods ".."
                    target_phrase = [self.keyboard.key_chars[i] for i in np.random.randint(0, len(self.keyboard.key_chars)-3, 5)]
                    target_phrase += [kconfig.break_char]*2

                    phrase_results = {}

                    self.num_presses = 0
                    self.num_corrections = 0
                    self.num_selections = 0
                    self.start_time = self.keyboard.sim_time.time()
                    self.phrase_num +=1

                    # type the target phrase, verbose=True will show targets and selections
                    self.type_phrase(target_phrase, verbose=verbose)

                    self.num_presses_total += self.num_presses

                    # only save simulation data from phrases more than halfway finished
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
                    self.keyboard.typed = ""  # reset tracking and context for lm -- new sentence

                # reset keyboard and clock scores for next session
                (self.keyboard.bc.clock_inf.clocks_on, self.keyboard.bc.clock_inf.clocks_off, clock_score_prior, self.keyboard.bc.is_undo,
                 self.keyboard.bc.is_equalize) = self.keyboard.make_choice(self.keyboard.bc.clock_inf.sorted_inds[0])
                # learn new scores
                if config.is_learning:
                    self.keyboard.bc.clock_inf.learn_scores(self.keyboard.bc.is_undo)
                # reset time indices
                self.keyboard.bc.init_round(True, False, clock_score_prior)
                self.keyboard.typed = ""

        print("\n\n")

        # save simulation metrics to pandas df
        self.result_df = pd.DataFrame(full_results)

    def type_phrase(self, target_phrase, verbose=False):
        if verbose:
            print("\nPhrase: ", target_phrase)

        # start with first target in phrase
        target_clock, cur_target_phrase = self.next_target(target_phrase)

        # typing phrase with available number of click times
        while len(self.session_click_times) > 0:
            if verbose:
                print("Target: "+self.keyboard.clock_to_text(target_clock))

            # call function to select target clock
            clock_selected = self.select_clock(target_clock, verbose=verbose)

            # terminate phrase if clock not selected (ran out of clicks)
            if clock_selected is None:
                break

            # if correct clock selected
            elif clock_selected:
                # if phrase not finished, move to next target
                if len(cur_target_phrase) > 0:
                    target_clock, cur_target_phrase = self.next_target(cur_target_phrase)

                # terminate phrase when everything has been typed
                else:
                    break

            # incorrect clock selected, need to undo
            else:
                undo_depth = 1
                while 0 < undo_depth <= sim_config.max_undo_depth:
                    undo_clock = self.keyboard.key_chars.index(kconfig.mybad_char)
                    if verbose:
                        tab = "-----"*undo_depth
                        print(tab+"Target: Undo")
                    self.num_corrections += 1
                    undo_success = self.select_clock(undo_clock, verbose=verbose, undo_depth=undo_depth)

                    # clock not selected (ran out of clicks)
                    if undo_success is None:
                        break

                    # undo clock selected
                    elif undo_success:
                        undo_depth -= 1

                    # incorrect clock selected, need to undo
                    else:
                        undo_depth += 1

    def select_clock(self, target_clock, verbose=False, undo_depth=0):

        ndt = self.keyboard.bc.clock_inf.clock_util.num_divs_time
        time_elapsed = 0

        cur_clicks = []

        # try to select target clock until success or maximum number of clicks reached
        for i in range(sim_config.max_clicks_per_selection):
            self.keyboard.winner = False

            # time increment until a keypress is simulated
            time_delta = 0

            # add the time until the target clock is exactly at noon to the time increment
            if self.keyboard.bc.clock_inf.clock_util.cur_hours[target_clock] > ndt / 2:
                time_delta += (ndt * 3 / 2 - self.keyboard.bc.clock_inf.clock_util.cur_hours[
                    target_clock]) / ndt * self.keyboard.time_rotate
            else:
                time_delta += (ndt / 2 - self.keyboard.bc.clock_inf.clock_util.cur_hours[target_clock]) / ndt * self.keyboard.time_rotate

            # pop a new click time sample if available
            if len(self.session_click_times) > 0:
                click_offset = float(self.session_click_times.pop())
                delay_offset = float(self.session_dead_times.pop())
                time_rotate = float(self.session_time_rotates.pop())

                # update clock period if different than current value
                if self.keyboard.time_rotate != time_rotate:
                    self.keyboard.time_rotate = time_rotate
                    self.keyboard.change_speed()

            # otherwise terminate selection process when run out of clicks
            else:
                break

            cur_clicks += [click_offset]

            # determine the closest full rotation to the dead time sample and add to time increment
            full_rotation_factor = (delay_offset//self.keyboard.time_rotate)
            time_delta += full_rotation_factor*self.keyboard.time_rotate

            # add the click time sample to the time increment
            time_delta += click_offset

            # step the simulation forward by the time increment
            time_elapsed += time_delta
            self.keyboard.sim_time.set_time(self.keyboard.sim_time.time() + time_delta)
            self.keyboard.increment_clocks()

            # simulate a keypress at the new time
            self.keyboard.on_press()
            self.num_presses += 1

            # check if press resulted in a clock selection
            if self.keyboard.winner:

                selected_clock = self.keyboard.previous_winner
                if verbose:

                    tab = "-----"*undo_depth
                    print(tab + "Typed: " + self.keyboard.typed)
                self.keyboard.winner = False

                # return if selected clock was the correct target
                if selected_clock == target_clock:
                    self.num_selections += 1
                    return True
                else:
                    return False

        return None

    def next_target(self, text):
        target_char = text[0]
        return self.keyboard.key_chars.index(target_char), text[1:]

    def update_progress_bar(self, trial, trials, session, num_sessions):
        if self.num_jobs == 0:
            progress = int((trial + session / num_sessions) / trials * 50)
        else:
            progress = int(((trial + session / num_sessions) / trials + self.job_num)/self.num_jobs * 50)

        sys.stdout.write('\r')
        sys.stdout.write("[%-50s] %d%%" % ('=' * progress, 2 * progress))
        sys.stdout.flush()
