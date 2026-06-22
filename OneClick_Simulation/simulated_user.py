from __future__ import division
import os
import sys
import numpy as np
import pandas as pd

from OneClick_Core import config
from OneClick_Text.keyboard import Keyboard
from OneClick_Text import kconfig
from OneClick_Simulation import sim_config
from Nomon_Text.text_stats import calc_MSD
from Nomon_Text.phrase_manager import Phrases


class ClickUtil:
    """
    Based on User_Simulation/simulated_user_text.py
    """

    def __init__(self, parent, click_df, calibration_clicks, type="playthrough"):
        self.parent = parent
        self.calibration_clicks = calibration_clicks
        self.click_df = click_df.reset_index(drop=True)
        self.type = type
        self.playthrough_index = 0
        self.clicks_remaining = self.click_df.shape[0]
        self.shuffle_indices = np.array([])
        self.reshuffle()

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
            return None

        elif self.type == "shuffle":
            if len(self.shuffle_indices) == 0:
                self.reshuffle()
            sample_index, self.shuffle_indices = self.shuffle_indices[0], self.shuffle_indices[1:]
            return self.click_df.loc[sample_index]

        elif self.type == "loop":
            if self.playthrough_index == self.click_df.shape[0]:
                self.playthrough_index = 0
                # re-prime the delay model with calibration data before looping
                self.parent.prime_delay_model()
            if self.playthrough_index < self.click_df.shape[0]:
                cur_click = self.click_df.iloc[self.playthrough_index]
                self.playthrough_index += 1
                return cur_click
        return None


class SimulatedUser:

    def __init__(self, cwd=os.getcwd(), job_num=None, num_jobs=0):
        self.job_num = job_num
        self.num_jobs = num_jobs
        self.working_dir = cwd
        self.keyboard = None
        self.result_df = None
        self.calibration_clicks = np.array([])

    # ------------------------------------------------------------------
    # Delay-model priming (analogue of Nomon's KDE priming)
    # ------------------------------------------------------------------

    def prime_delay_model(self):
        """Warm-start the Gaussian delay model from pre-session calibration clicks."""
        dm = self.keyboard.bc.clock_inf.delay_model
        for yin in self.calibration_clicks:
            if not np.isnan(yin):
                dm.update(float(yin))

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def parameter_metrics(self, parameters, trials=1, verbose=False):
        click_df = parameters["click_df"]
        self.calibration_clicks = \
            click_df[click_df["Session Num"].isna()][["Click Time Relative (s)"]].to_numpy().T[0]
        self.click_df = click_df[click_df["Session Num"].notna()]
        self.sessions_li = pd.unique(self.click_df["Session Num"])
        self.num_clicks_loaded = len(self.click_df)

        full_results = []

        for trial in range(trials):
            self.keyboard = Keyboard(self, parameters=parameters)

            phrase_shuffle_seed = parameters.get("phrase_shuffle_seed", None)
            if callable(phrase_shuffle_seed):
                cur_seed = phrase_shuffle_seed(trial)
            elif isinstance(phrase_shuffle_seed, int):
                cur_seed = phrase_shuffle_seed
            else:
                cur_seed = None

            self.phrase_util = Phrases(
                'Nomon_Text/resources/watch-iv.txt',
                'Nomon_Text/resources/watch-oov.txt',
                cur_seed,
            )

            # prime the delay model with calibration data before the first session
            self.prime_delay_model()

            for session_num in self.sessions_li:
                session_df = self.click_df[self.click_df["Session Num"] == session_num]
                self.click_util = ClickUtil(self, session_df, self.calibration_clicks, "playthrough")

                if "phrase_df" in parameters:
                    phrase_df = parameters["phrase_df"]
                    session_phrases = phrase_df[phrase_df["Session Num"] == session_num]["Phrase Text"].values
                    self.phrase_util.phrases = [[p, "?"] for p in session_phrases]

                self.phrase_num = 0

                while self.click_util.clicks_remaining > 0:
                    target_phrase, phrase_type = self.phrase_util.sample()
                    if target_phrase is None:
                        break

                    self._clear_phrase_tracking()
                    self.phrase_num += 1

                    self.type_phrase(target_phrase, verbose=verbose)

                    if (len(self.keyboard.typed) > len(target_phrase) // 2
                            and self.num_selections_phrase > 0):
                        results = self._calculate_phrase_results(target_phrase, phrase_type)
                        results["Session Num"] = int(session_num)
                        results["Trial Num"] = int(trial)
                        results["Phrase Num"] = int(self.phrase_num)
                        full_results.append(results)

                    # reset text/context for the next phrase
                    self.keyboard.typed = ""
                    self.keyboard.context = ""
                    self.keyboard.typed_versions = []
                    self.keyboard.place_letter_clocks()

        self.result_df = pd.DataFrame(full_results)
        print("\n\nDone. {} phrases recorded.".format(len(full_results)))

    # ------------------------------------------------------------------
    # Phrase / word typing
    # ------------------------------------------------------------------

    def type_phrase(self, target_phrase, verbose=False):
        words = target_phrase.strip().split()
        if verbose:
            print("\nPhrase:", target_phrase)

        for target_word in words:
            if self.click_util.clicks_remaining <= 0:
                break

            undo_depth = 0
            while undo_depth <= sim_config.max_undo_depth:
                outcome = self._attempt_word(target_word, verbose=verbose, undo_depth=undo_depth)
                if outcome == "ok":
                    break
                if outcome is None:        # out of clicks
                    return
                # outcome in {"wrong_committed", "miss"} -> a correction is needed
                self.num_corrections_phrase += 1
                if outcome == "wrong_committed":
                    if not self._undo_via_clock(verbose=verbose):
                        return             # out of clicks during undo
                undo_depth += 1
                # if we've exhausted the undo budget, give up on this word

    def _attempt_word(self, target_word, verbose=False, undo_depth=0):
        """
        One fresh attempt at a word. Returns:
          "ok"              committed the target word
          "wrong_committed" committed a different word (must be undone via the undo clock)
          "miss"            Enter landed on undo/empty (nothing committed; observations reset)
          None              ran out of clicks
        """
        # --- Space presses: one per letter, querying the API after each ---
        for letter in target_word:
            if self.click_util.clicks_remaining <= 0:
                return None
            letter_index = self._letter_index(letter)
            if self._press_letter(letter_index) is None:
                return None
            self.keyboard.update_word_list()
            if self.keyboard.word_clock_index(target_word) != kconfig.undo_word_index:
                break   # target word now in an active clock -> commit early

        # --- choose the word clock to target ---
        target_idx = self.keyboard.word_clock_index(target_word)
        if target_idx == kconfig.undo_word_index:
            # target not typeable as-is; best-effort commit of the literal argmax decode
            target_idx = kconfig.argmax_word_index

        # --- Enter press ---
        res = self._press_enter(target_idx)
        if res is None:
            return None
        word, selected_index = res
        self.num_selections_phrase += 1

        if word is not None and word.lower() == target_word.lower():
            self.keyboard.commit_word(word)
            self.num_word_selections_phrase += 1
            if verbose:
                print("  " * undo_depth + f"  [Enter] committed '{word}'")
            return "ok"

        if word is None:
            # undo/empty clock selected by timing noise; nothing committed
            if verbose:
                print("  " * undo_depth + "  [Enter] miss (undo/empty)")
            self.keyboard._reset_letter_round()
            return "miss"

        # wrong word committed
        self.keyboard.commit_word(word)
        if verbose:
            print("  " * undo_depth + f"  [Enter] wrong: got '{word}', wanted '{target_word}'")
        return "wrong_committed"

    def _undo_via_clock(self, verbose=False):
        """
        Revert the last (wrong) commit by selecting the undo clock with a real Enter
        press (ASSUMES THIS IS EASY TO HIT ACCURATLEY). Returns False if we run out of clicks.
        """
        self.keyboard.update_word_list()        # obs_len 0 -> valid = [undo_word_index]
        res = self._press_enter(kconfig.undo_word_index)
        if res is None:
            return False
        _, selected_index = res
        self.num_selections_phrase += 1
        if selected_index == kconfig.undo_word_index:
            self.keyboard.undo_word()
            if verbose:
                print("  [Enter] undo clock selected -> reverted")
        return True

    # ------------------------------------------------------------------
    # Timed presses (select_clock-style core: move target clock to noon, sample a real
    # click, advance sim_time, increment clocks, press)
    # ------------------------------------------------------------------

    def _press_letter(self, target_letter_index):
        """One Space press targeting a letter clock. Returns True or None (out of clicks)."""
        cur_click = self.click_util.sample()
        if cur_click is None:
            return None

        ndt = self.keyboard.bc.clock_inf.clock_util.num_divs_time
        self._apply_period(cur_click)
        click_offset, full_rotations = self._click_components(cur_click)

        cur_hour = self.keyboard.bc.clock_inf.clock_util.cur_hours[target_letter_index]
        time_delta = self._time_to_noon(cur_hour, ndt) * self.keyboard.time_rotate + full_rotations + click_offset

        self.keyboard.sim_time.set_time(self.keyboard.sim_time.time() + time_delta)
        self.keyboard.increment_clocks()
        self.keyboard.on_press()
        self.num_clicks_phrase += 1
        return True

    def _press_enter(self, target_word_index):
        """
        One Enter press targeting a word clock. Returns (committed_or_None, selected_index)
        or None (out of clicks).
        """
        cur_click = self.click_util.sample()
        if cur_click is None:
            return None

        ndt = self.keyboard.word_clock_util.num_divs_time
        self._apply_period(cur_click)
        click_offset, full_rotations = self._click_components(cur_click)

        cur_hour = self.keyboard.word_clock_util.cur_hours.get(target_word_index, 0)
        time_delta = self._time_to_noon(cur_hour, ndt) * self.keyboard.time_rotate + full_rotations + click_offset

        self.keyboard.sim_time.set_time(self.keyboard.sim_time.time() + time_delta)
        self.keyboard.increment_word_clocks()
        self.num_clicks_phrase += 1
        return self.keyboard.on_enter()

    # ------------------------------------------------------------------
    # Press helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _time_to_noon(cur_hour, ndt):
        if cur_hour > ndt / 2:
            return (ndt * 3 / 2 - cur_hour) / ndt
        return (ndt / 2 - cur_hour) / ndt

    def _apply_period(self, cur_click):
        time_rotate = float(cur_click["Clock Period (s)"])
        if self.keyboard.time_rotate != time_rotate:
            self.keyboard.rotate_index = int(np.argmin(np.abs(config.period_li - time_rotate)))
            self.keyboard.time_rotate = time_rotate
            self.keyboard.change_speed()

    def _click_components(self, cur_click):
        click_offset = float(cur_click["Click Time Relative (s)"])
        dead_time = float(cur_click["Dead Time (s)"]) if not np.isnan(cur_click["Dead Time (s)"]) else 0.0
        time_rotate = self.keyboard.time_rotate
        full_rotations = (dead_time // time_rotate) * time_rotate
        return click_offset, full_rotations

    def _letter_index(self, letter):
        letter_lower = letter.lower()
        if letter_lower not in kconfig.key_chars:
            letter_lower = "'"
        return kconfig.key_chars.index(letter_lower)

    # NOTE: _time_to_noon returns a fraction of one rotation; callers scale by time_rotate.
    # (kept separate from period scaling so the same helper serves letter & word clocks)

    # ------------------------------------------------------------------
    # Metrics (column set matches Nomon's calculate_phrase_results)
    # ------------------------------------------------------------------

    def _clear_phrase_tracking(self):
        self.num_clicks_phrase = 0
        self.num_corrections_phrase = 0
        self.num_selections_phrase = 0
        self.num_word_selections_phrase = 0
        self.start_time = self.keyboard.sim_time.time()

    def _calculate_phrase_results(self, target_phrase, phrase_type):
        typed_clean = self.keyboard.typed.rstrip()
        target_clean = target_phrase[:len(typed_clean)]
        _, error_rate = calc_MSD(typed_clean, target_clean)

        elapsed = self.keyboard.sim_time.time() - self.start_time
        n_chars = max(len(self.keyboard.typed), 1)
        n_sel = max(self.num_selections_phrase, 1)

        return {
            "Target Phrase": target_phrase,
            "Phrase Type": phrase_type,
            "Typed Text": self.keyboard.typed,
            "Click Load (clicks/character)": round(self.num_clicks_phrase / n_chars,
                                                   sim_config.data_save_precision),
            "Click Load (clicks/selection)": round(self.num_clicks_phrase / n_sel,
                                                   sim_config.data_save_precision),
            "Entry Rate (cpm)": round(n_chars / elapsed * 60, sim_config.data_save_precision) if elapsed > 0 else 0,
            "Entry Rate (wpm)": round(n_chars / elapsed * 60 / 5, sim_config.data_save_precision) if elapsed > 0 else 0,
            "Correction Rate (%)": round(self.num_corrections_phrase / n_sel * 100,
                                         sim_config.data_save_precision),
            "Error Rate (%)": round(error_rate, sim_config.data_save_precision),
            "Word Prediction Usage (%)": round(self.num_word_selections_phrase / n_sel * 100,
                                               sim_config.data_save_precision),
        }

    def update_progress_bar(self, progress):
        sys.stdout.write('\r')
        sys.stdout.write("[%-50s] %d%%" % ('=' * (progress // 2), progress))
        sys.stdout.flush()
