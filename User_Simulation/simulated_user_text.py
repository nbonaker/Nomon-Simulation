#!/usr/bin/python

from Nomon_Core import config
from Nomon_Text.keyboard import Keyboard
from Nomon_Text import kconfig
from Nomon_Simulation import sim_config
from Nomon_Text.text_stats import calc_MSD
from Nomon_Text.phrase_manager import Phrases
from Nomon_Text.kenlm.kenlm_lm import lognormalize_factor

import pandas as pd
import numpy as np
from scipy.stats import entropy
from matplotlib import pyplot as plt
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
    def __init__(self, cwd=os.getcwd(), job_num=None, num_jobs=0):
        # used for tracking overall progressbar
        self.job_num = job_num
        self.num_jobs = num_jobs

        self.working_dir = cwd

        # initialize the keyboard
        self.keyboard = None
        print("Initializing Keyboard with the following layout:")
        for row in kconfig.alpha_target_layout:
            print(row)

        self.track_entropy = False
        self.click_entropy = []

        self.cur_selection_cscores = []
        self.cumulative_cscores = []

    def get_session_clicks(self):
        session_click_df = self.click_df[self.click_df["Session Num"] == self.session_num]

        self.session_click_times = list(session_click_df[["Click Time Relative (s)"]].to_numpy().T[0])
        self.session_dead_times = list(session_click_df[["Dead Time (s)"]].to_numpy().T[0])
        self.session_time_rotates = list(session_click_df[["Clock Period (s)"]].to_numpy().T[0])

    def clear_sim_tracking(self):
        self.num_clicks_phrase = 0
        self.num_corrections_phrase = 0
        self.num_selections_phrase = 0
        self.num_chars_phrase = 0
        self.num_word_selections_phrase = 0
        self.start_time = self.keyboard.sim_time.time()
        self.winner = False
        self.winner_text = ""

    def parameter_metrics(self, parameters, trials=1, verbose=False):

        # initialize the click time samples
        click_df = parameters["click_df"]
        self.calibration_clicks = click_df[click_df["Session Num"].isna()][["Click Time Rlative (s)"]].to_numpy().T[0]
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

            self.click_util = ClickUtil(self, self.click_df, self.calibration_clicks, "playthrough")

            # reinitialize the keyboard for each trial
            self.keyboard = Keyboard(self, parameters=parameters)

            # initialize the phrases
            # check if phrases are to be shuffled consistently with seed
            if "phrase_shuffle_seed" in parameters:
                phrase_shuffle_seed = parameters["phrase_shuffle_seed"]

                # if function of trial number, compute seed from fn
                if callable(phrase_shuffle_seed):
                    cur_seed = phrase_shuffle_seed(trial)
                # if integer supplied, use it
                elif isinstance(phrase_shuffle_seed, int):
                    cur_seed = phrase_shuffle_seed
                # otherwise seed is invalid
                else:
                    raise TypeError("phrase_shuffle_seed is not an INT or Fn as expected")

            # if no seed supplied, then phrases will be randomly shuffled each trial and simulation
            else:
                cur_seed = None

            self.phrase_util = Phrases('../Nomon_Text/resources/watch-iv.txt', '../Nomon_Text/resources/watch-oov.txt',
                                       cur_seed)

            # prime kde with calibration data before the first session
            for yin in self.calibration_clicks:
                self.keyboard.bc.clock_inf.clock_util.clock_inf.kde.add_point(float(yin))

            # run through each session in the user data
            for rel_session_num, session_num in enumerate(self.sessions_li):

                # overwrite phrase queue if specified in simulation parameters
                if "phrase_df" in parameters:
                    phrase_df = parameters["phrase_df"]
                    session_phrases = phrase_df[phrase_df["Session Num"] == session_num]["Phrase Text"].values
                    self.phrase_util.phrases = [[phrase, "?"] for phrase in session_phrases]

                self.session_num = session_num

                # get the click data for the current session
                self.get_session_clicks()
                num_session_clicks = len(self.session_click_times)

                self.phrase_num = 0

                # type phrases until all clicks are used
                while len(self.session_click_times) > 0:
                    target_phrase, phrase_type = self.phrase_util.sample()

                    # stop trial when run out of phrases
                    if target_phrase is None:
                        break

                    target_phrase += " .."

                    phrase_results = {}

                    self.clear_sim_tracking()

                    self.phrase_num += 1

                    # type the target phrase, verbose=True will show targets and selections
                    self.type_phrase(target_phrase, verbose=verbose)

                    self.num_clicks_total += self.num_clicks_phrase

                    # only save simulation data from phrases more than halfway finished
                    if len(self.keyboard.typed) > len(target_phrase) // 2 and self.num_selections_phrase > 0:

                        phrase_results = self.calculate_phrase_results(target_phrase, phrase_type)
                        phrase_results["Session Num"] = int(self.session_num)
                        phrase_results["Trial Num"] = int(trial)
                        phrase_results["Phrase Num"] = int(self.phrase_num)

                        full_results.append(phrase_results)
                    self.keyboard.typed = ""  # reset tracking and context for lm -- new sentence

                    cur_progress = int((trial + self.num_clicks_total / self.num_clicks_loaded) / trials * 100)
                    self.update_progress_bar(cur_progress)

                # reset keyboard and clock scores for next session
                (self.keyboard.bc.clock_inf.clocks_on, self.keyboard.bc.clock_inf.clocks_off, clock_score_prior,
                 self.keyboard.bc.is_undo,
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

    def calculate_phrase_results(self, target_phrase, phrase_type):
        # calculate error rate (ignore differences in periods and spaces at end of phrase)
        typed_no_periods = self.keyboard.typed.replace("..", "")
        target_no_periods = target_phrase[:len(self.keyboard.typed)].replace("..", "")
        num_errors, error_rate_phrase = calc_MSD(typed_no_periods, target_no_periods)

        phrase_results = {}

        phrase_results["Target Phrase"] = target_phrase
        phrase_results["Phrase Type"] = phrase_type
        phrase_results["Typed Text"] = self.keyboard.typed

        phrase_results["Click Load (clicks/character)"] = \
            round(self.num_clicks_phrase / len(self.keyboard.typed), sim_config.data_save_precision)
        phrase_results["Click Load (clicks/selection)"] = \
            round(self.num_clicks_phrase / self.num_selections_phrase, sim_config.data_save_precision)

        phrase_results["Entry Rate (cpm)"] = \
            round(len(self.keyboard.typed) / (self.keyboard.sim_time.time() - self.start_time) * 60,
                  sim_config.data_save_precision)
        phrase_results["Entry Rate (wpm)"] = \
            round(len(self.keyboard.typed) / (self.keyboard.sim_time.time() - self.start_time) * 60 / 5,
                  sim_config.data_save_precision)

        phrase_results["Correction Rate (%)"] = \
            round(self.num_corrections_phrase / self.num_selections_phrase * 100,
                  sim_config.data_save_precision)

        phrase_results["Error Rate (%)"] = round(error_rate_phrase, sim_config.data_save_precision)

        phrase_results["Word Prediction Usage (%)"] = round(
            self.num_word_selections_phrase / self.num_selections_phrase * 100,
            sim_config.data_save_precision)

        return phrase_results

    def simulate_phrases(self, parameters, trials=1, verbose=False):
        # initialize the click time samples
        click_df = parameters["click_df"]
        self.calibration_clicks = click_df[click_df["Session Num"].isna()][["Click Time Relative (s)"]].to_numpy().T[0]
        self.click_df = click_df[click_df["Session Num"].notna()]

        # list to store sim results
        full_results = []

        for trial in range(trials):

            self.click_util = ClickUtil(self, self.click_df, self.calibration_clicks, "loop")

            # reinitialize the keyboard for each trial
            self.keyboard = Keyboard(self, parameters=parameters)

            self.session_num = 1

            # initialize the phrases
            # check if phrases are to be shuffled consistently with seed
            if "phrase_shuffle_seed" in parameters:
                phrase_shuffle_seed = parameters["phrase_shuffle_seed"]

                # if function of trial number, compute seed from fn
                if callable(phrase_shuffle_seed):
                    cur_seed = phrase_shuffle_seed(trial)
                # if integer supplied, use it
                elif isinstance(phrase_shuffle_seed, int):
                    cur_seed = phrase_shuffle_seed
                # otherwise seed is invalid
                else:
                    raise TypeError("phrase_shuffle_seed is not an INT or Fn as expected")

            # if no seed supplied, then phrases will be randomly shuffled each trial and simulation
            else:
                cur_seed = None

            self.phrase_util = Phrases('../Nomon_Text/resources/watch-iv.txt', '../Nomon_Text/resources/watch-oov.txt',
                                       cur_seed)
            # self.phrase_util = Phrases("../Nomon_Text/resources/lm_likely_phrases.txt", "../Nomon_Text/resources/lm_unlikely_phrases.txt", cur_seed)

            # overwrite phrase queue if specified in simulation parameters
            if "phrase_df" in parameters:
                phrase_df = parameters["phrase_df"]
                phrases = phrase_df["Phrase Text"].values
                self.phrase_util.phrases = [[phrase, "?"] for phrase in phrases]

            # prime kde with calibration data before the first session
            for yin in self.calibration_clicks:
                self.keyboard.bc.clock_inf.clock_util.clock_inf.kde.add_point(float(yin))

            self.num_phrases_total = len(self.phrase_util.phrases)

            self.phrase_num = 0

            # type phrases until all phrases are done
            while len(self.phrase_util.phrases) > 0:
                target_phrase, phrase_type = self.phrase_util.sample()

                target_phrase += " .."

                self.clear_sim_tracking()

                self.phrase_num += 1

                # type the target phrase, verbose=True will show targets and selections
                self.type_phrase(target_phrase, verbose=verbose)

                if len(self.keyboard.typed) > len(target_phrase) // 2 and self.num_selections_phrase > 0:
                    phrase_results = self.calculate_phrase_results(target_phrase, phrase_type)
                    phrase_results["Session Num"] = int(self.session_num)
                    phrase_results["Trial Num"] = int(trial)
                    phrase_results["Phrase Num"] = int(self.phrase_num)

                    full_results.append(phrase_results)

                self.keyboard.typed = ""  # reset tracking and context for lm -- new sentence

                cur_progress = int((trial+self.phrase_num/self.num_phrases_total)/trials*100)
                self.update_progress_bar(cur_progress)

            # reset keyboard and clock scores for next session
            (self.keyboard.bc.clock_inf.clocks_on, self.keyboard.bc.clock_inf.clocks_off, clock_score_prior,
             self.keyboard.bc.is_undo,
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
            print("\nPhrase: ", target_phrase, self.click_util.clicks_remaining)

        # cur_clicks = self.click_df["Click Time Relative (s)"].values[:self.click_util.playthrough_index]
        # cur_weights = np.power(0.96, np.arange(0, len(cur_clicks)))[::-1]
        # plt.hist(cur_clicks, weights=cur_weights, bins=40, density=True)
        # plt.plot(self.keyboard.bc.clock_inf.kde.x_li, np.array(self.keyboard.bc.clock_inf.kde.dens_li)/self.keyboard.bc.clock_inf.kde.Z*10)
        # plt.show()

        # start with first target in phrase
        target_clock, cur_target_phrase = self.next_target(target_phrase)

        self.target_clock = target_clock

        # type phrase while clicks are available (playthrough click_util mode)
        while self.click_util.clicks_remaining > 0:
            if verbose:
                print("Target: " + self.keyboard.clock_to_text(target_clock))

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
                    self.target_clock = target_clock

                # terminate phrase when everything has been typed
                else:
                    break

            # incorrect clock selected, need to undo
            else:

                undo_depth = 1
                while 0 < undo_depth <= sim_config.max_undo_depth:
                    undo_clock = self.keyboard.keys_li.index(kconfig.mybad_char) * (
                            self.keyboard.N_pred + 1) + self.keyboard.N_pred
                    self.target_clock = undo_clock
                    if verbose:
                        tab = "-----" * undo_depth
                        print(tab + "Target: Undo")
                    self.num_corrections_phrase += 1
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
                self.target_clock = target_clock

    def select_clock(self, target_clock, verbose=False, undo_depth=0):

        ndt = self.keyboard.bc.clock_inf.clock_util.num_divs_time
        time_elapsed = 0

        cur_clicks = []

        # try to select target clock until success or maximum number of clicks reached
        for i in range(sim_config.max_clicks_per_selection):
            self.keyboard.winner = False

            # calculate time increment until a keypress is simulated
            time_delta = 0

            # add the time until the target clock is exactly at noon to the time increment
            if self.keyboard.bc.clock_inf.clock_util.cur_hours[target_clock] > ndt / 2:
                time_delta += (ndt * 3 / 2 - self.keyboard.bc.clock_inf.clock_util.cur_hours[
                    target_clock]) / ndt * self.keyboard.time_rotate
            else:
                time_delta += (ndt / 2 - self.keyboard.bc.clock_inf.clock_util.cur_hours[
                    target_clock]) / ndt * self.keyboard.time_rotate

            # pop a new click time sample if available
            cur_click = self.click_util.sample()

            if cur_click is not None:
                click_offset = float(cur_click["Click Time Relative (s)"])
                delay_offset = float(cur_click["Dead Time (s)"])
                time_rotate = float(cur_click["Clock Period (s)"])

                # update clock period if different than current value
                if self.keyboard.time_rotate != time_rotate:
                    self.keyboard.time_rotate = time_rotate
                    self.keyboard.change_speed()

            # otherwise terminate selection process when run out of clicks
            else:
                break

            cur_clicks += [click_offset]

            # determine the closest full rotation to the dead time sample and add to time increment
            full_rotation_factor = (delay_offset // self.keyboard.time_rotate)
            time_delta += full_rotation_factor * self.keyboard.time_rotate

            # add the click time sample to the time increment
            time_delta += click_offset

            # step the simulation forward by the time increment
            time_elapsed += time_delta
            self.keyboard.sim_time.set_time(self.keyboard.sim_time.time() + time_delta)
            self.keyboard.increment_clocks()

            # save clock scores before click if entropy tracking is on
            if self.track_entropy:
                self.save_clock_cscores(i)
            # simulate a keypress at the new time
            self.keyboard.on_press()
            self.num_clicks_phrase += 1

            # check if press resulted in a clock selection
            if self.keyboard.winner:

                selected_clock = self.keyboard.previous_winner

                # check if word prediction was selected
                if (selected_clock - kconfig.N_pred) % (kconfig.N_pred + 1) != 0:
                    self.num_word_selections_phrase += 1

                if verbose:
                    tab = "-----" * undo_depth
                    print(tab + "Typed: " + self.keyboard.typed)
                self.keyboard.winner = False

                # return if selected clock was the correct target
                self.num_selections_phrase += 1
                if selected_clock == target_clock:
                    return True
                else:
                    return False

        return None

    def next_target(self, text):
        words = text.split(" ")
        if "" in words:
            words.remove("")

        if len(words) > 1:
            remaining_words = ""
            for word in words[1:]:
                if remaining_words != "":
                    remaining_words += " "
                remaining_words += word
            first_word = words[0]
        else:
            remaining_words = ""
            first_word = words[0]

        target_word = self.keyboard.context + first_word + " "
        if target_word in self.keyboard.word_list:
            words_list_flattened = [word for sublist in self.keyboard.words_li for word in sublist + [""]]
            return words_list_flattened.index(target_word), remaining_words

        target_letter = text[0]
        if target_letter == " ":
            target_letter = "_"
        return self.keyboard.keys_li.index(target_letter) * (self.keyboard.N_pred + 1) + self.keyboard.N_pred, text[1:]

    def clock_ind_to_object(self, index):
        object_name = ""

        # if index is a key char
        if (index - self.keyboard.N_pred) % (self.keyboard.N_pred + 1) == 0:
            object_name = self.keyboard.keys_li[self.keyboard.index_to_wk[index]]
        # if index is a word
        else:
            key = self.keyboard.index_to_wk[index] // self.keyboard.N_pred
            pred = self.keyboard.index_to_wk[index] % self.keyboard.N_pred
            object_name = self.keyboard.words_li[key][pred]

        object_name = object_name.replace(" ", "_")
        object_name = object_name.replace(kconfig.back_char, "BACKSPACE")
        object_name = object_name.replace(kconfig.mybad_char, "UNDO")

        return object_name

    def save_clock_cscores(self, click_num):
        sorted_inds = self.keyboard.bc.clock_inf.sorted_inds
        obj_names = [self.clock_ind_to_object(index) for index in sorted_inds]

        valid_cscores = [self.keyboard.bc.clock_inf.cscores[i] for i in self.keyboard.bc.clock_inf.sorted_inds]
        # normalize cscores for plot
        valid_cscores = np.array(valid_cscores)
        #
        if np.sum(valid_cscores) == 0:
            valid_cscores = -np.ones(valid_cscores.shape)
        valid_cscores -= lognormalize_factor(valid_cscores)

        cur_click_cscores = {
            "phrase_num": self.phrase_num,
            "click_num": click_num,
            "selection_num": self.num_selections_phrase,
            "target_clock_ind": self.target_clock,
            "target_clock_text": self.clock_ind_to_object(self.target_clock),
            "is_post_undo": self.keyboard.bc.is_undo,
            "num_clocks_on": len(sorted_inds),
            "cscore_entropy": entropy(np.exp(valid_cscores), base=2)
        }

        word_pred_num = 0
        for index in self.keyboard.words_on:
            obj_name = self.clock_ind_to_object(index)
            cscore = valid_cscores[sorted_inds.index(index)]

            # if index is a key char
            if (index - self.keyboard.N_pred) % (self.keyboard.N_pred + 1) == 0:
                cur_click_cscores[obj_name] = cscore
            else:
                cur_click_cscores["word_"+str(word_pred_num)+"_cscore"] = cscore
                cur_click_cscores["word_" + str(word_pred_num)+"_text"] = obj_name
                word_pred_num += 1

        for i in range(word_pred_num, kconfig.num_words_total):
            cur_click_cscores["word_" + str(i) + "_cscore"] = -float("inf")
            cur_click_cscores["word_" + str(i) + "_text"] = ""

        self.cur_selection_cscores += [cur_click_cscores]

        # if last click that leads to selection
        if click_num == -1:
            if len(self.cur_selection_cscores) > 1:
                sel_num_clicks = self.cur_selection_cscores[-2]["click_num"]+1
            else:
                sel_num_clicks = 1
            self.cur_selection_cscores[-1]["click_num"] = sel_num_clicks

            for previous_cscores in self.cur_selection_cscores:
                previous_cscores["num_clicks_selection"] = sel_num_clicks
                previous_cscores["selected_clock_ind"] = self.keyboard.previous_winner
                previous_cscores["selected_clock_text"] = self.clock_ind_to_object(self.keyboard.previous_winner)
                previous_cscores["is_correct_selection"] = self.keyboard.previous_winner == self.target_clock

            self.cumulative_cscores += self.cur_selection_cscores
            self.cur_selection_cscores = []

        # fig, ax = plt.subplots(1, figsize=(7, 8))
        # fig.tight_layout()
        # plt.subplots_adjust(left=0.15)
        # ax.barh(obj_names[::-1], valid_cscores[::-1])
        # ax.invert_xaxis()
        # plt.show()

    def update_progress_bar(self, progress):
        sys.stdout.write('\r')
        sys.stdout.write("[%-50s] %d%%" % ('=' * (progress//2), progress))
        sys.stdout.flush()

    def __del__(self):
        # ensure the kenlm instance is deleted upon sim completion (hogs memory)
        del self.keyboard.lm
        del self.keyboard


def main():
    # perfect user (always clicks at noon) sanity check
    sim = SimulatedUser()
    sim.clear_sim_tracking()
    sim.session_click_times = [0 for i in range(100)]
    sim.session_dead_times = [0 for i in range(100)]
    sim.session_time_rotates = [config.period_li[10] for i in range(100)]
    sim.type_phrase("hello my name is nick", verbose=True)


if __name__ == "__main__":
    main()
