#!/usr/bin/python

from Nomon_Core import config
from Nomon_Text.keyboard import Keyboard
from Nomon_Text import kconfig
from User_Simulation import sim_config
from Nomon_Text.text_stats import calc_MSD
from Nomon_Text.phrase_manager import Phrases
from Nomon_Text.textslinger_lm import lognormalize_factor

import pandas as pd
import numpy as np
from scipy.stats import entropy
from matplotlib import pyplot as plt
import sys
import os


SYNTHETIC_GROUP_ID_COL = "Synthetic Group ID"
SYNTHETIC_GROUP_CLICK_NUM_COL = "Synthetic Group Click Num"
SYNTHETIC_SELECTION_TYPE_COL = "Synthetic Selection Type"
SYNTHETIC_DEAD_TIME_CLIPPED_COL = "Synthetic Dead Time Clipped"
SYNTHETIC_SAMPLING_MODE_COL = "Synthetic Sampling Mode"
SYNTHETIC_REPLAY_SEQUENCE_COL = "Synthetic Replay Sequence"


class ClickUtil:
    def __init__(self, parent, click_df, calibration_clicks, type, seed=0):
        self.parent = parent
        self.calibration_clicks = calibration_clicks
        self.click_df = click_df
        self.shuffle_indices = np.array([])
        self.playthrough_index = 0
        self.clicks_remaining = self.click_df.shape[0]
        self.reshuffle()

        self.type = type
        self.rng = np.random.default_rng(seed)
        self.selection_bootstrap_enabled = {
            SYNTHETIC_GROUP_ID_COL,
            SYNTHETIC_GROUP_CLICK_NUM_COL,
            SYNTHETIC_SELECTION_TYPE_COL,
        }.issubset(click_df.columns)
        self.selection_group_pools = {}
        self.active_group = None
        self.active_group_index = 0
        self.active_requested_type = None
        self.selection_event = None
        sampling_modes = (
            click_df[SYNTHETIC_SAMPLING_MODE_COL].dropna().astype(str)
            if SYNTHETIC_SAMPLING_MODE_COL in click_df
            else pd.Series(dtype=str)
        )
        self.selection_sampling_mode = sampling_modes.iloc[0] if not sampling_modes.empty else "bootstrap"
        self.replay_group_pools = {}
        self.replay_group_indices = {}
        if self.selection_bootstrap_enabled:
            for _group_id, group_df in click_df.groupby(SYNTHETIC_GROUP_ID_COL, sort=False):
                ordered_group = group_df.sort_values(SYNTHETIC_GROUP_CLICK_NUM_COL).reset_index(drop=True)
                selection_type = str(ordered_group.iloc[0][SYNTHETIC_SELECTION_TYPE_COL])
                self.selection_group_pools.setdefault(selection_type, []).append(ordered_group)
                if self.selection_sampling_mode == "ordered_replay":
                    first_row = ordered_group.iloc[0]
                    context = (
                        int(first_row["Session Num"]),
                        int(first_row["Phrase Num"]),
                    )
                    self.replay_group_pools.setdefault(context, []).append(ordered_group)
            for context, groups in self.replay_group_pools.items():
                groups.sort(key=lambda group: int(group.iloc[0][SYNTHETIC_REPLAY_SEQUENCE_COL]))
                self.replay_group_indices[context] = 0

    def reshuffle(self):
        self.shuffle_indices = np.arange(self.click_df.shape[0])
        np.random.shuffle(self.shuffle_indices)

    def sample(self):
        if self.selection_bootstrap_enabled:
            if self.clicks_remaining <= 0:
                return None
            if self.active_group is None:
                if self.selection_event is not None:
                    return None
                self.begin_selection("character")
            if self.active_group_index >= len(self.active_group):
                if not self._load_selection_group(continuation=True):
                    return None

            cur_click = self.active_group.iloc[self.active_group_index]
            self.active_group_index += 1
            self.playthrough_index += 1
            self.clicks_remaining -= 1
            self.selection_event["clicks_consumed"] += 1
            if (
                self.selection_event["clicks_consumed"] == 1
                and bool(cur_click.get(SYNTHETIC_DEAD_TIME_CLIPPED_COL, False))
            ):
                self.selection_event["dead_time_clipped_count"] += 1
            return cur_click

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

    def begin_selection(self, selection_type, target_text=None):
        if not self.selection_bootstrap_enabled:
            return
        if self.selection_event is not None:
            raise RuntimeError("Cannot start a bootstrap selection before ending the current one")

        self.active_requested_type = selection_type
        self.selection_event = {
            "Trial Num": int(self.parent.trial_num),
            "Session Num": int(self.parent.session_num),
            "Phrase Num": int(self.parent.phrase_num),
            "Original Phrase Num": self.parent.current_original_phrase_num,
            "target_type": selection_type,
            "target_text": target_text,
            "sampled_group_count": 0,
            "primary_group_size": 0,
            "sampled_click_capacity": 0,
            "clicks_consumed": 0,
            "unused_clicks": 0,
            "exhausted_group_count": 0,
            "continuation_group_count": 0,
            "fallback_group_count": 0,
            "dead_time_clipped_count": 0,
            "outcome": None,
            "early_selection": False,
            "source_group_id": None,
            "source_selection_type": None,
            "source_selection": None,
            "source_session_num": None,
            "source_phrase_num": None,
            "source_selection_num": None,
            "target_type_matches_source": None,
        }
        self._load_selection_group(continuation=False)

    def _load_selection_group(self, continuation):
        if self.selection_sampling_mode == "ordered_replay":
            if continuation:
                self.selection_event["exhausted_group_count"] += 1
                self.active_group = None
                return False
            context = (int(self.parent.session_num), int(self.parent.current_original_phrase_num))
            pool = self.replay_group_pools.get(context, [])
            replay_index = self.replay_group_indices.get(context, 0)
            if replay_index >= len(pool):
                self.active_group = None
                return False
            group = pool[replay_index]
            self.replay_group_indices[context] = replay_index + 1
        else:
            pool = self.selection_group_pools.get(self.active_requested_type, [])
            if not pool:
                pool = self.selection_group_pools.get("character", [])
                if not pool and self.selection_group_pools:
                    pool = next(iter(self.selection_group_pools.values()))
                if pool:
                    self.selection_event["fallback_group_count"] += 1
            if not pool:
                self.active_group = None
                return False
            group = pool[int(self.rng.integers(0, len(pool)))]

        if continuation:
            self.selection_event["exhausted_group_count"] += 1
            self.selection_event["continuation_group_count"] += 1
        self.active_group = group
        self.active_group_index = 0
        group_size = int(len(group))
        self.selection_event["sampled_group_count"] += 1
        self.selection_event["sampled_click_capacity"] += group_size
        if not continuation:
            self.selection_event["primary_group_size"] = group_size
            first_row = group.iloc[0]
            source_type = str(first_row[SYNTHETIC_SELECTION_TYPE_COL])
            self.selection_event.update(
                {
                    "source_group_id": int(first_row[SYNTHETIC_GROUP_ID_COL]),
                    "source_selection_type": source_type,
                    "source_selection": first_row.get("Synthetic Source Selection"),
                    "source_session_num": first_row.get("Synthetic Source Session Num"),
                    "source_phrase_num": first_row.get("Synthetic Source Phrase Num"),
                    "source_selection_num": first_row.get("Synthetic Source Selection Num"),
                    "target_type_matches_source": self.active_requested_type == source_type,
                }
            )
        return True

    def end_selection(self, outcome, selected_text=None):
        if not self.selection_bootstrap_enabled or self.selection_event is None:
            return

        unused_clicks = 0
        if self.active_group is not None:
            unused_clicks = max(int(len(self.active_group) - self.active_group_index), 0)
        self.selection_event["unused_clicks"] = unused_clicks
        self.selection_event["outcome"] = outcome
        self.selection_event["selected_text"] = selected_text
        self.selection_event["early_selection"] = bool(
            outcome in {"correct", "incorrect"} and unused_clicks > 0
        )
        self.parent.selection_bootstrap_events.append(dict(self.selection_event))
        self.active_group = None
        self.active_group_index = 0
        self.active_requested_type = None
        self.selection_event = None


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
        self.perfect_word_prediction_clicks = sim_config.perfect_word_prediction_clicks
        self.perfect_word_prediction_click_offset_s = sim_config.perfect_word_prediction_click_offset_s
        self.word_prediction_decisions = []
        self.selection_bootstrap_events = []
        self.current_original_phrase_num = None
        self.trial_num = 0
        self.force_target_clock_selection = False

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

        self.perfect_word_prediction_clicks = parameters.get(
            "perfect_word_prediction_clicks",
            sim_config.perfect_word_prediction_clicks,
        )
        self.perfect_word_prediction_click_offset_s = parameters.get(
            "perfect_word_prediction_click_offset_s",
            sim_config.perfect_word_prediction_click_offset_s,
        )

        # initialize the click time samples
        click_df = parameters["click_df"]
        self.calibration_clicks = click_df[click_df["Session Num"].isna()][["Click Time Relative (s)"]].to_numpy().T[0]
        self.click_df = click_df[click_df["Session Num"].notna()]
        # used in progress bar to track simulation progress
        self.num_clicks_loaded = len(self.click_df["Click Time Relative (s)"])
        self.num_clicks_total = 0
        self.selection_bootstrap_seed = int(parameters.get("selection_bootstrap_seed", 0))
        self.record_attempted_phrases = bool(parameters.get("record_attempted_phrases", False))
        self.append_terminal_periods = bool(parameters.get("append_terminal_periods", True))
        self.force_target_clock_selection = bool(
            parameters.get("force_target_clock_selection", False)
        )

        # get session numbers from click data frame
        self.sessions_li = pd.unique(self.click_df["Session Num"])
        self.num_sessions = int(self.sessions_li.max()) - int(self.sessions_li.min())

        # list to store sim results
        full_results = []
        self.word_prediction_decisions = []
        self.selection_bootstrap_events = []

        for trial in range(trials):
            self.trial_num = trial

            self.click_util = ClickUtil(
                self,
                self.click_df,
                self.calibration_clicks,
                "playthrough",
                seed=self.selection_bootstrap_seed + trial,
            )

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

                self.phrase_metadata = []

                # overwrite phrase queue if specified in simulation parameters
                if "phrase_df" in parameters:
                    phrase_df = parameters["phrase_df"]
                    session_phrase_df = phrase_df[phrase_df["Session Num"] == session_num].sort_values("Phrase Num")
                    metadata_columns = [
                        column
                        for column in ["Phrase Num", "Phrase Text", "Comparison Phrase ID"]
                        if column in session_phrase_df
                    ]
                    session_phrase_rows = session_phrase_df[metadata_columns].to_dict("records")
                    self.phrase_util.phrases = [[row["Phrase Text"], "?"] for row in reversed(session_phrase_rows)]
                    self.phrase_metadata = [
                        {
                            "Original Phrase Num": int(row["Phrase Num"]),
                            **(
                                {"Comparison Phrase ID": row["Comparison Phrase ID"]}
                                if "Comparison Phrase ID" in row
                                else {}
                            ),
                        }
                        for row in reversed(session_phrase_rows)
                    ]

                self.session_num = session_num

                # get the click data for the current session
                self.get_session_clicks()
                num_session_clicks = len(self.session_click_times)

                self.phrase_num = 0

                # type phrases until all clicks are used
                while len(self.session_click_times) > 0:
                    target_phrase, phrase_type = self.phrase_util.sample()
                    phrase_metadata = self.phrase_metadata.pop() if self.phrase_metadata else {}

                    # stop trial when run out of phrases
                    if target_phrase is None:
                        break

                    if self.append_terminal_periods:
                        target_phrase += " .."

                    phrase_results = {}

                    self.clear_sim_tracking()

                    self.phrase_num += 1
                    self.current_original_phrase_num = phrase_metadata.get("Original Phrase Num")

                    # type the target phrase, verbose=True will show targets and selections
                    self.type_phrase(target_phrase, verbose=verbose)

                    self.num_clicks_total += self.num_clicks_phrase

                    # only save simulation data from phrases more than halfway finished
                    should_record = (
                        self.record_attempted_phrases
                        or (
                            len(self.keyboard.typed) > len(target_phrase) // 2
                            and self.num_selections_phrase > 0
                        )
                    )
                    if should_record:

                        phrase_results = self.calculate_phrase_results(target_phrase, phrase_type)
                        phrase_results["Session Num"] = int(self.session_num)
                        phrase_results["Trial Num"] = int(trial)
                        phrase_results["Phrase Num"] = int(self.phrase_num)
                        if "Original Phrase Num" in phrase_metadata:
                            phrase_results["Original Phrase Num"] = phrase_metadata["Original Phrase Num"]
                        if "Comparison Phrase ID" in phrase_metadata:
                            phrase_results["Comparison Phrase ID"] = phrase_metadata["Comparison Phrase ID"]

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
        if typed_no_periods:
            num_errors, error_rate_phrase = calc_MSD(typed_no_periods, target_no_periods)
        else:
            num_errors, error_rate_phrase = len(target_no_periods), 100.0 if target_no_periods else 0.0

        phrase_results = {}

        phrase_results["Target Phrase"] = target_phrase
        phrase_results["Phrase Type"] = phrase_type
        phrase_results["Typed Text"] = self.keyboard.typed

        typed_len = max(len(self.keyboard.typed), 1)
        selection_count = max(self.num_selections_phrase, 1)
        elapsed = self.keyboard.sim_time.time() - self.start_time

        phrase_results["Click Load (clicks/character)"] = \
            round(self.num_clicks_phrase / typed_len, sim_config.data_save_precision)
        phrase_results["Click Load (clicks/selection)"] = \
            round(self.num_clicks_phrase / selection_count, sim_config.data_save_precision)

        phrase_results["Entry Rate (cpm)"] = \
            round(len(self.keyboard.typed) / elapsed * 60,
                  sim_config.data_save_precision) if elapsed > 0 else 0
        phrase_results["Entry Rate (wpm)"] = \
            round(len(self.keyboard.typed) / elapsed * 60 / 5,
                  sim_config.data_save_precision) if elapsed > 0 else 0

        phrase_results["Correction Rate (%)"] = \
            round(self.num_corrections_phrase / selection_count * 100,
                  sim_config.data_save_precision)

        phrase_results["Error Rate (%)"] = round(error_rate_phrase, sim_config.data_save_precision)

        phrase_results["Num Selections"] = int(self.num_selections_phrase)
        phrase_results["Num Word Prediction Selections"] = int(self.num_word_selections_phrase)
        phrase_results["Word Prediction Usage (%)"] = round(
            self.num_word_selections_phrase / selection_count * 100,
            sim_config.data_save_precision)
        phrase_results["Num Clicks"] = int(self.num_clicks_phrase)
        phrase_results["Num Corrections"] = int(self.num_corrections_phrase)
        completion_fraction = min(len(self.keyboard.typed) / max(len(target_phrase), 1), 1.0)
        phrase_results["Completion Fraction"] = round(completion_fraction, sim_config.data_save_precision)
        exact_completion = self.keyboard.typed.rstrip() == target_phrase.rstrip()
        phrase_results["Phrase Completed"] = bool(exact_completion)
        phrase_results["Simulated Attempt Time (s)"] = float(elapsed)
        phrase_results["Simulated Completion Time (s)"] = (
            float(elapsed) if exact_completion else np.nan
        )

        return phrase_results

    def simulate_phrases(self, parameters, trials=1, verbose=False):
        self.perfect_word_prediction_clicks = parameters.get(
            "perfect_word_prediction_clicks",
            sim_config.perfect_word_prediction_clicks,
        )
        self.perfect_word_prediction_click_offset_s = parameters.get(
            "perfect_word_prediction_click_offset_s",
            sim_config.perfect_word_prediction_click_offset_s,
        )

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

        self.target_words_phrase = [
            word for word in target_phrase.split(" ")
            if word and word not in [".", ",", "?", "!", ".."]
        ]
        self.current_target_event_id = None

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
            self.update_word_prediction_decision(clock_selected)

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
        self.last_selected_clock = None
        target_type = self.selection_target_type(target_clock, undo_depth)
        target_text = self.clock_ind_to_object(target_clock)
        self.click_util.begin_selection(target_type, target_text=target_text)

        cur_clicks = []

        if self.force_target_clock_selection:
            cur_click = self.click_util.sample()
            if cur_click is None:
                self.click_util.end_selection("no_selection")
                return None

            time_rotate = float(cur_click["Clock Period (s)"])
            if self.keyboard.time_rotate != time_rotate:
                self.keyboard.time_rotate = time_rotate
                self.keyboard.change_speed()

            (
                self.keyboard.bc.clock_inf.clocks_on,
                self.keyboard.bc.clock_inf.clocks_off,
                clock_score_prior,
                self.keyboard.bc.is_undo,
                self.keyboard.bc.is_equalize,
            ) = self.keyboard.make_choice(target_clock)
            if config.is_learning:
                self.keyboard.bc.clock_inf.learn_scores(self.keyboard.bc.is_undo)
            self.keyboard.bc.init_round(True, False, clock_score_prior)

            self.num_clicks_phrase += 1
            self.num_selections_phrase += 1
            self.last_selected_clock = target_clock
            if self.is_word_prediction_clock(target_clock):
                self.num_word_selections_phrase += 1
            self.click_util.end_selection("correct", selected_text=target_text)
            return True

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
                if self.should_force_word_prediction_click(target_clock):
                    click_offset = self.perfect_word_prediction_click_offset_s

                # update clock period if different than current value
                if self.keyboard.time_rotate != time_rotate:
                    self.keyboard.time_rotate = time_rotate
                    self.keyboard.change_speed()

            # otherwise terminate selection process when run out of clicks
            else:
                break

            cur_clicks += [click_offset]

            # Dead time is the pause before a selection attempt, not before
            # every click needed to resolve that selection.
            if i == 0:
                full_rotation_factor = delay_offset // self.keyboard.time_rotate
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
                self.last_selected_clock = selected_clock

                # check if word prediction was selected
                if self.is_word_prediction_clock(selected_clock):
                    self.num_word_selections_phrase += 1

                if verbose:
                    tab = "-----" * undo_depth
                    print(tab + "Typed: " + self.keyboard.typed)
                self.keyboard.winner = False

                # return if selected clock was the correct target
                self.num_selections_phrase += 1
                selected_text = self.clock_ind_to_object(selected_clock)
                if selected_clock == target_clock:
                    self.click_util.end_selection("correct", selected_text=selected_text)
                    return True
                else:
                    self.click_util.end_selection("incorrect", selected_text=selected_text)
                    return False

        self.click_util.end_selection("no_selection")
        return None

    def selection_target_type(self, target_clock, undo_depth=0):
        if undo_depth > 0:
            return "correction"
        if self.is_word_prediction_clock(target_clock):
            return "word_prediction"
        return "character"

    def should_force_word_prediction_click(self, target_clock):
        if not self.perfect_word_prediction_clicks:
            return False
        return (target_clock - self.keyboard.N_pred) % (self.keyboard.N_pred + 1) != 0

    def update_word_prediction_decision(self, clock_selected):
        event_id = self.current_target_event_id
        if event_id is None:
            return

        event = self.word_prediction_decisions[event_id]
        selected_clock = self.last_selected_clock
        event["Selection Outcome"] = (
            "no_selection"
            if clock_selected is None
            else "correct"
            if clock_selected
            else "incorrect"
        )
        event["Selected Clock Index"] = selected_clock
        event["Selected Text"] = self.clock_ind_to_object(selected_clock) if selected_clock is not None else None
        event["Selected Word Prediction"] = (
            self.is_word_prediction_clock(selected_clock) if selected_clock is not None else False
        )
        event["Targeted Prediction Succeeded"] = bool(
            event["Targeted Prediction"] and clock_selected
        )

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

        if text[0] == " ":
            self.current_target_event_id = None
            return self.keyboard.keys_li.index("_") * (self.keyboard.N_pred + 1) + self.keyboard.N_pred, text[1:]

        target_word = self.keyboard.context + first_word + " "
        if target_word in self.keyboard.word_list:
            words_list_flattened = [word for sublist in self.keyboard.words_li for word in sublist + [""]]
            target_index = words_list_flattened.index(target_word)
            self.record_word_prediction_decision(text, first_word, target_word, target_index)
            return target_index, remaining_words

        target_letter = text[0]
        if target_letter == " ":
            target_letter = "_"
        target_index = self.keyboard.keys_li.index(target_letter) * (self.keyboard.N_pred + 1) + self.keyboard.N_pred
        self.record_word_prediction_decision(text, first_word, target_word, target_index)
        return target_index, text[1:]

    def record_word_prediction_decision(self, text, first_word, target_word, target_clock):
        if first_word in [".", ",", "?", "!", ".."]:
            self.current_target_event_id = None
            return

        active_words = [
            word for word in text.split(" ")
            if word and word not in [".", ",", "?", "!", ".."]
        ]
        word_index = len(self.target_words_phrase) - len(active_words) + 1
        prediction = self.find_word_prediction(target_word)
        event = {
            "Trial Num": int(getattr(self, "trial_num", 0)),
            "Session Num": int(self.session_num),
            "Phrase Num": int(self.phrase_num),
            "Original Phrase Num": self.current_original_phrase_num,
            "Word Index": int(word_index),
            "Target Word": target_word.strip(),
            "Current Prefix": self.keyboard.context,
            "Prefix Length": len(self.keyboard.context),
            "Prediction Visible": prediction is not None,
            "Prediction Rank": prediction["rank"] if prediction else None,
            "Prediction Key": prediction["key"] if prediction else None,
            "Prediction Clock Index": prediction["clock_index"] if prediction else None,
            "Saved Chars": len(target_word) - len(self.keyboard.context) if prediction else None,
            "Target Clock Index": int(target_clock),
            "Targeted Prediction": self.is_word_prediction_clock(target_clock),
            "Selection Outcome": None,
            "Selected Clock Index": None,
            "Selected Text": None,
            "Selected Word Prediction": False,
            "Targeted Prediction Succeeded": False,
        }
        self.current_target_event_id = len(self.word_prediction_decisions)
        self.word_prediction_decisions.append(event)

    def find_word_prediction(self, target_word):
        for key_index, word_predictions in enumerate(self.keyboard.words_li):
            for pred_index, predicted_word in enumerate(word_predictions):
                if predicted_word == target_word:
                    return {
                        "rank": pred_index + 1,
                        "key": self.keyboard.keys_li[key_index],
                        "clock_index": key_index * (self.keyboard.N_pred + 1) + pred_index,
                    }
        return None

    def is_word_prediction_clock(self, clock_index):
        return (clock_index - self.keyboard.N_pred) % (self.keyboard.N_pred + 1) != 0

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

        for i in range(word_pred_num, self.keyboard.num_words_total):
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
