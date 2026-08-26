from __future__ import division
import os
import sys
import math
import json
import numpy as np
import pandas as pd

from OneClick_Core import config
from OneClick_Text.keyboard import Keyboard
from OneClick_Text import kconfig
from OneClick_Simulation import sim_config
from Nomon_Text.text_stats import calc_MSD
from Nomon_Text.phrase_manager import Phrases


FAILURE_STAGES = {
    "click_stream_exhausted_between_words": "between_words",
    "click_stream_exhausted_letters": "letters",
    "click_stream_exhausted_target_enter": "target_enter",
    "click_stream_exhausted_undo": "undo",
    "word_click_budget_between_attempts": "between_attempts",
    "word_click_budget_letters": "letters",
    "word_click_budget_target_enter": "target_enter",
    "undo_click_budget_exhausted": "undo",
    "target_not_displayed": "word_prediction",
    "target_enter_retries_exhausted": "target_enter",
    "word_attempts_exhausted": "word_attempts",
    "final_text_mismatch": "phrase_validation",
}


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
        self.record_attempted_phrases = bool(parameters.get("record_attempted_phrases", False))
        self.max_word_attempts = int(
            parameters.get("max_word_attempts", sim_config.max_word_attempts)
        )
        self.max_clicks_per_word = int(
            parameters.get("max_clicks_per_word", sim_config.max_clicks_per_word)
        )
        self.max_enter_attempts = int(
            parameters.get("max_enter_attempts", sim_config.max_enter_attempts)
        )
        if self.max_enter_attempts < 1:
            raise ValueError("max_enter_attempts must be at least 1")
        self.undo_mode = str(parameters.get("undo_mode", "protected"))
        # Preserve compatibility with saved callers while replacing the old
        # cascading behavior with protected correction semantics.
        if self.undo_mode == "competing":
            self.undo_mode = "protected"
        if self.undo_mode not in {"protected", "undo_only"}:
            raise ValueError("undo_mode must be 'protected' or 'undo_only'")
        self.stop_phrase_on_failed_word = bool(
            parameters.get("stop_phrase_on_failed_word", False)
        )
        self.perfect_letter_observations = bool(
            parameters.get("perfect_letter_observations", False)
        )
        fixed_clock_period = parameters.get("fixed_clock_period_s")
        fixed_space_period = parameters.get("fixed_space_clock_period_s")
        fixed_enter_period = parameters.get("fixed_enter_clock_period_s")
        has_specialized_period = (
            fixed_space_period is not None or fixed_enter_period is not None
        )
        if fixed_clock_period is not None and has_specialized_period:
            raise ValueError(
                "fixed_clock_period_s cannot be combined with specialized "
                "Space/Enter clock periods"
            )
        if (fixed_space_period is None) != (fixed_enter_period is None):
            raise ValueError(
                "fixed_space_clock_period_s and fixed_enter_clock_period_s "
                "must be supplied together"
            )
        self.fixed_clock_period_s = (
            None if fixed_clock_period is None else float(fixed_clock_period)
        )
        if self.fixed_clock_period_s is not None:
            fixed_space_period = self.fixed_clock_period_s
            fixed_enter_period = self.fixed_clock_period_s
        self.fixed_space_clock_period_s = (
            None if fixed_space_period is None else float(fixed_space_period)
        )
        self.fixed_enter_clock_period_s = (
            None if fixed_enter_period is None else float(fixed_enter_period)
        )
        for name, value in [
            ("fixed_clock_period_s", self.fixed_clock_period_s),
            ("fixed_space_clock_period_s", self.fixed_space_clock_period_s),
            ("fixed_enter_clock_period_s", self.fixed_enter_clock_period_s),
        ]:
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be a positive finite value")

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
                quiet="phrase_df" in parameters,
            )

            # prime the delay model with calibration data before the first session
            self.prime_delay_model()

            for session_num in self.sessions_li:
                session_df = self.click_df[self.click_df["Session Num"] == session_num]
                self.click_util = ClickUtil(self, session_df, self.calibration_clicks, "playthrough")

                if "phrase_df" in parameters:
                    phrase_df = parameters["phrase_df"]
                    session_phrase_df = phrase_df[
                        phrase_df["Session Num"] == session_num
                    ].copy()
                    if "Phrase Num" not in session_phrase_df:
                        session_phrase_df["Phrase Num"] = np.arange(
                            1, len(session_phrase_df) + 1
                        )
                    session_phrase_df = session_phrase_df.sort_values("Phrase Num")
                    metadata_columns = [
                        column
                        for column in [
                            "Phrase Num",
                            "Phrase Text",
                            "Phrase Type",
                            "Comparison Phrase ID",
                        ]
                        if column in session_phrase_df
                    ]
                    session_phrase_rows = session_phrase_df[metadata_columns].to_dict("records")
                    self.phrase_util.phrases = [
                        [row["Phrase Text"], row.get("Phrase Type", "?")]
                        for row in reversed(session_phrase_rows)
                    ]
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
                else:
                    self.phrase_metadata = []

                self.phrase_num = 0

                while self.click_util.clicks_remaining > 0:
                    target_phrase, phrase_type = self.phrase_util.sample()
                    phrase_metadata = self.phrase_metadata.pop() if self.phrase_metadata else {}
                    if target_phrase is None:
                        break

                    self._clear_phrase_tracking()
                    self.phrase_num += 1

                    self.type_phrase(target_phrase, verbose=verbose)

                    should_record = (
                        self.record_attempted_phrases
                        or (
                            len(self.keyboard.typed) > len(target_phrase) // 2
                            and self.num_selections_phrase > 0
                        )
                    )
                    if should_record:
                        results = self._calculate_phrase_results(target_phrase, phrase_type)
                        results["Session Num"] = int(session_num)
                        results["Trial Num"] = int(trial)
                        results["Phrase Num"] = int(self.phrase_num)
                        if "Original Phrase Num" in phrase_metadata:
                            results["Original Phrase Num"] = phrase_metadata["Original Phrase Num"]
                        if "Comparison Phrase ID" in phrase_metadata:
                            results["Comparison Phrase ID"] = phrase_metadata["Comparison Phrase ID"]
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

        for word_position, target_word in enumerate(words, start=1):
            if self.click_util.clicks_remaining <= 0:
                self._last_attempt_target_displayed = False
                self._record_phrase_failure(
                    "click_stream_exhausted_between_words",
                    target_word,
                    word_position,
                    attempts_for_word=0,
                    word_start_metrics=self._word_start_metrics(),
                    failure_limit="click_stream",
                )
                return

            self.num_target_words_phrase += 1
            completed = False
            attempts_for_word = 0
            word_start_clicks = self.num_clicks_phrase
            word_start_metrics = self._word_start_metrics()
            terminal_reason = None
            last_retry_reason = None
            failure_guard = ""
            failure_limit = ""
            while attempts_for_word < self.max_word_attempts:
                if self.num_clicks_phrase - word_start_clicks >= self.max_clicks_per_word:
                    failure_guard = "word_click_budget_between_attempts"
                    terminal_reason = last_retry_reason or failure_guard
                    failure_limit = "max_clicks_per_word"
                    if verbose:
                        print(
                            f"  [Word failed] '{target_word}' after "
                            f"{self.max_clicks_per_word} word-clicks"
                        )
                    break
                outcome = self._attempt_word(
                    target_word,
                    verbose=verbose,
                    undo_depth=attempts_for_word,
                    word_start_clicks=word_start_clicks,
                )
                attempts_for_word += 1
                self.num_word_attempts_phrase += 1
                if outcome == "ok":
                    self.num_completed_words_phrase += 1
                    # Nomon's selection count is the number of resolved clock
                    # selections, not merely the final correct word commit. Keep
                    # every click/selection (including correction work) for a word
                    # that eventually succeeds, and none from a terminal failure.
                    self.num_successful_word_clicks_phrase += (
                        self.num_clicks_phrase - word_start_metrics["clicks"]
                    )
                    self.num_successful_word_selections_phrase += (
                        self.num_selections_phrase - word_start_metrics["selections"]
                    )
                    self.num_successful_word_characters_phrase += len(target_word)
                    completed = True
                    break
                terminal_reason = outcome
                if outcome.startswith("click_stream_exhausted"):
                    failure_guard = outcome
                    failure_limit = "click_stream"
                    break
                if outcome.startswith("word_click_budget") or outcome == "undo_click_budget_exhausted":
                    failure_guard = outcome
                    failure_limit = "max_clicks_per_word"
                    break
                # Target display/selection retries start a fresh letter-entry
                # attempt while retaining their exact cause for terminal telemetry.
                last_retry_reason = outcome
                self.keyboard._reset_letter_round()
            if not completed:
                if terminal_reason is None:
                    terminal_reason = last_retry_reason or "word_attempts_exhausted"
                if not failure_limit:
                    failure_limit = "max_word_attempts"
                if not failure_guard:
                    failure_guard = "word_attempts_exhausted"
                if verbose:
                    print(
                        f"  [Word failed] '{target_word}' after "
                        f"{attempts_for_word} attempts"
                    )
                self.num_failed_words_phrase += 1
                self._record_phrase_failure(
                    terminal_reason,
                    target_word,
                    word_position,
                    attempts_for_word,
                    word_start_metrics,
                    failure_limit,
                    failure_guard,
                )
                self.keyboard._reset_letter_round()
                if self.stop_phrase_on_failed_word or terminal_reason.startswith(
                    "click_stream_exhausted"
                ):
                    return

    def _word_start_metrics(self):
        return {
            "clicks": self.num_clicks_phrase,
            "selections": self.num_selections_phrase,
            "letters": self.num_letter_presses_phrase,
            "target_enters": self.num_target_enter_attempts_phrase,
            "undos": self.num_undo_attempts_phrase,
        }

    def _record_phrase_failure(
        self,
        reason,
        target_word,
        word_position,
        attempts_for_word,
        word_start_metrics,
        failure_limit,
        failure_guard="",
    ):
        """Count every terminal path and preserve details for the first one."""
        event = {
            "reason": reason,
            "stage": FAILURE_STAGES.get(reason, "unknown"),
            "limit": failure_limit,
            "guard": failure_guard or reason,
            "target_word": target_word,
            "word_position": int(word_position),
            "word_attempt": int(attempts_for_word),
            "target_was_displayed": bool(self._last_attempt_target_displayed),
            "word_click_count": int(
                self.num_clicks_phrase - word_start_metrics["clicks"]
            ),
            "letter_press_count": int(
                self.num_letter_presses_phrase - word_start_metrics["letters"]
            ),
            "target_enter_attempt_count": int(
                self.num_target_enter_attempts_phrase - word_start_metrics["target_enters"]
            ),
            "undo_attempt_count": int(
                self.num_undo_attempts_phrase - word_start_metrics["undos"]
            ),
        }
        self.failure_events_phrase.append(event)
        self.failure_reason_counts_phrase[reason] = (
            self.failure_reason_counts_phrase.get(reason, 0) + 1
        )
        if self.phrase_failure_reason:
            return
        self.phrase_failure_reason = reason
        self.phrase_failure_stage = event["stage"]
        self.phrase_failure_limit = failure_limit
        self.phrase_failure_guard = event["guard"]
        self.failed_target_word = target_word
        self.failed_word_position = int(word_position)
        self.failed_word_attempt = int(attempts_for_word)
        self.failed_target_was_displayed = bool(self._last_attempt_target_displayed)
        self.failure_word_click_count = event["word_click_count"]
        self.failure_letter_press_count = event["letter_press_count"]
        self.failure_target_enter_attempt_count = event["target_enter_attempt_count"]
        self.failure_undo_attempt_count = event["undo_attempt_count"]

    def _attempt_word(self, target_word, verbose=False, undo_depth=0, word_start_clicks=0):
        """
        One fresh attempt at a word. Returns "ok" or an exact retry/failure
        reason suitable for terminal phrase telemetry.
        """
        self._last_attempt_target_displayed = False
        # --- Space presses: one per letter, querying the API after each ---
        for letter in target_word:
            if self.click_util.clicks_remaining <= 0:
                return "click_stream_exhausted_letters"
            if self.num_clicks_phrase - word_start_clicks >= self.max_clicks_per_word:
                return "word_click_budget_letters"
            letter_index = self._letter_index(letter)
            if self._press_letter(letter_index) is None:
                return "click_stream_exhausted_letters"
            if self.perfect_letter_observations:
                self._replace_latest_observation_with_perfect_letter(letter_index)
            self.keyboard.update_word_list()
            if self.keyboard.word_clock_index(target_word) != kconfig.undo_word_index:
                break   # target word now in an active clock -> commit early

        # --- choose the word clock to target and preserve the decoded state ---
        target_idx = self.keyboard.word_clock_index(target_word)
        target_is_displayed = target_idx != kconfig.undo_word_index
        self._last_attempt_target_displayed = target_is_displayed
        if not target_is_displayed:
            # target not typeable as-is; best-effort commit of the literal argmax decode
            target_idx = kconfig.argmax_word_index
        snapshot = self.keyboard.capture_word_attempt_state()

        # --- Enter presses: retry the same decoded state after timing errors ---
        for enter_attempt in range(self.max_enter_attempts):
            if self.num_clicks_phrase - word_start_clicks >= self.max_clicks_per_word:
                return "word_click_budget_target_enter"
            if enter_attempt > 0:
                self.num_enter_retries_phrase += 1

            res = self._press_enter(target_idx)
            if res is None:
                return "click_stream_exhausted_target_enter"
            self.num_target_enter_attempts_phrase += 1
            word, selected_index = res
            self.num_selections_phrase += 1

            if word is not None and word.lower() == target_word.lower():
                # The simulator knows the intended target, so this successful
                # selection is safe to use as a timing-calibration sample.
                prediction_source = self._prediction_source(selected_index)
                if prediction_source is None:
                    raise RuntimeError(
                        "successful word selection did not use a prediction clock: "
                        f"index={selected_index}"
                    )
                self.prediction_source_counts_phrase[prediction_source] += 1
                self.prediction_selection_events_phrase.append(
                    {
                        "target_word": target_word,
                        "selected_word": word,
                        "selected_index": int(selected_index),
                        "prediction_source": prediction_source,
                    }
                )
                self.keyboard.commit_word(word, confirmed_correct=True)
                self.num_word_selections_phrase += 1
                if verbose:
                    print("  " * undo_depth + f"  [Enter] committed '{word}'")
                return "ok"

            self.num_corrections_phrase += 1
            if word is None:
                # Undo/empty was selected, but no text changed. Keep the current
                # observations, predictions, and clock phases for another Enter.
                if verbose:
                    print("  " * undo_depth + "  [Enter] miss (undo/empty); retrying")
                continue

            # A real wrong commit clears the live word round. Undo every erroneous
            # commit, then restore the pre-Enter candidate state.
            self.num_wrong_word_commits_phrase += 1
            self.keyboard.commit_word(word, confirmed_correct=False)
            if verbose:
                print("  " * undo_depth + f"  [Enter] wrong: got '{word}', wanted '{target_word}'")
            undo_outcome = self._undo_via_clock(
                word_start_clicks=word_start_clicks,
                verbose=verbose,
            )
            if undo_outcome != "ok":
                return undo_outcome

            self.keyboard.restore_word_attempt_state(snapshot)
            self.num_restored_states_phrase += 1
            if not target_is_displayed:
                # Repeating Enter cannot make an absent target appear; acquire a
                # fresh set of noisy letter observations instead.
                return "target_not_displayed"

        if target_is_displayed:
            return "target_enter_retries_exhausted"
        return "target_not_displayed"

    def _undo_via_clock(self, word_start_clicks, verbose=False):
        """
        Remove one wrong commit through a protected correction round.

        Prediction clocks still compete in protected mode, but a non-Undo winner
        is treated as a correction miss rather than another commit. Returns True
        only after Undo wins; otherwise it returns an exact failure reason.
        """
        if self.undo_mode == "undo_only":
            # Correction-focused upper bound: only the Undo clock is active.
            self.keyboard.prepare_undo_round(undo_only=True)
        else:
            # Protected interface variant: prediction clocks remain visible and
            # compete, but their selection is suppressed while correction is latched.
            self.keyboard.prepare_undo_round(undo_only=False)

        while True:
            if self.num_clicks_phrase - word_start_clicks >= self.max_clicks_per_word:
                self.num_undo_failures_phrase += 1
                return "undo_click_budget_exhausted"

            res = self._press_enter(kconfig.undo_word_index, action_kind="undo")
            if res is None:
                self.num_undo_failures_phrase += 1
                return "click_stream_exhausted_undo"
            word, selected_index = res
            self.num_selections_phrase += 1
            self.num_undo_attempts_phrase += 1

            if selected_index == kconfig.undo_word_index:
                self.keyboard.undo_word()
                # This is the only event that meets the finalized correction
                # definition: an intentional, successful Undo after a wrong commit.
                self.num_corrective_undo_actions_phrase += 1
                if verbose:
                    print("  [Enter] undo clock selected -> reverted")
                return "ok"

            # Correction remains latched: a competing prediction can win the timed
            # selection, but it cannot mutate text or language-model context.
            self.num_corrections_phrase += 1
            if verbose:
                selected = "empty" if word is None else repr(word)
                print(f"  [Enter] protected undo miss selected {selected}; retrying")

    # ------------------------------------------------------------------
    # Timed presses (select_clock-style core: move target clock to noon, sample a real
    # click, advance sim_time, increment clocks, press)
    # ------------------------------------------------------------------

    def _press_letter(self, target_letter_index):
        """One Space press targeting a letter clock. Returns True or None (out of clicks)."""
        cur_click = self.click_util.sample()
        if cur_click is None:
            return None

        self._apply_period(cur_click)
        period_s = float(self.keyboard.time_rotate)
        ndt = self.keyboard.bc.clock_inf.clock_util.num_divs_time
        click_offset, full_rotations = self._click_components(
            cur_click,
            period_s,
        )
        self.dead_time_s_phrase += full_rotations

        cur_hour = self.keyboard.bc.clock_inf.clock_util.cur_hours[target_letter_index]
        time_delta = (
            self._press_time_delta(
                cur_hour,
                ndt,
                period_s,
                click_offset,
            )
            + full_rotations
        )

        self.keyboard.sim_time.set_time(self.keyboard.sim_time.time() + time_delta)
        self.keyboard.increment_clocks()
        self.keyboard.on_press(target_letter_index)
        self.num_clicks_phrase += 1
        self.num_letter_presses_phrase += 1
        self.letter_clock_time_s += time_delta
        return True

    def _press_enter(self, target_word_index, action_kind="target_enter"):
        """
        One Enter press targeting a word clock. Returns (committed_or_None, selected_index)
        or None (out of clicks).
        """
        if action_kind not in {"target_enter", "undo"}:
            raise ValueError("action_kind must be 'target_enter' or 'undo'")
        cur_click = self.click_util.sample()
        if cur_click is None:
            return None

        self._apply_period(cur_click)
        period_s = float(self.keyboard.word_clock_util.time_rotate)
        ndt = self.keyboard.word_clock_util.num_divs_time
        click_offset, full_rotations = self._click_components(
            cur_click,
            period_s,
        )
        self.dead_time_s_phrase += full_rotations

        cur_hour = self.keyboard.word_clock_util.cur_hours.get(target_word_index, 0)
        time_delta = (
            self._press_time_delta(
                cur_hour,
                ndt,
                period_s,
                click_offset,
            )
            + full_rotations
        )

        self.keyboard.sim_time.set_time(self.keyboard.sim_time.time() + time_delta)
        self.keyboard.increment_word_clocks()
        self.num_clicks_phrase += 1
        if action_kind == "undo":
            self.undo_clock_time_s += time_delta
        else:
            self.target_enter_clock_time_s += time_delta
        result = self.keyboard.on_enter()
        _, selected_index = result
        self.num_enter_presses_phrase += 1
        if selected_index != target_word_index:
            self.num_enter_misselections_phrase += 1
        return result

    # ------------------------------------------------------------------
    # Press helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _time_to_noon(cur_hour, ndt):
        if cur_hour > ndt / 2:
            return (ndt * 3 / 2 - cur_hour) / ndt
        return (ndt / 2 - cur_hour) / ndt

    @staticmethod
    def _press_time_delta(cur_hour, ndt, period_s, click_offset_s):
        """
        Clock-controlled time until a click targeting the next noon.

        Positive offsets are late and extend the nominal wait. If an early
        negative offset would place the click before the current simulation
        time, target the corresponding early phase before the following noon.
        """
        ndt = float(ndt)
        period_s = float(period_s)
        click_offset_s = float(click_offset_s)
        cur_hour = float(cur_hour)
        if (
            not all(
                math.isfinite(value)
                for value in (ndt, period_s, click_offset_s, cur_hour)
            )
            or ndt <= 0
            or period_s <= 0
        ):
            raise ValueError("clock phase, period, and click offset must be finite")

        current_phase = (cur_hour / ndt) % 1.0
        remaining_fraction = (0.5 - current_phase) % 1.0
        time_delta = remaining_fraction * period_s + click_offset_s
        if time_delta < -1e-12:
            time_delta += math.ceil(-time_delta / period_s) * period_s
        if time_delta < 0:
            time_delta = 0.0
        if not math.isfinite(time_delta):
            raise ValueError("calculated press time must be finite")
        return time_delta

    def _apply_period(self, cur_click):
        if (
            getattr(self, "fixed_space_clock_period_s", None) is not None
            or getattr(self, "fixed_clock_period_s", None) is not None
        ):
            return
        time_rotate = float(cur_click["Clock Period (s)"])
        if self.keyboard.time_rotate != time_rotate:
            self.keyboard.rotate_index = int(np.argmin(np.abs(config.period_li - time_rotate)))
            self.keyboard.time_rotate = time_rotate
            self.keyboard.change_speed()

    def _click_components(self, cur_click, active_period_s):
        click_offset = float(cur_click["Click Time Relative (s)"])
        dead_time = float(cur_click["Dead Time (s)"]) if not np.isnan(cur_click["Dead Time (s)"]) else 0.0
        active_period_s = float(active_period_s)
        full_rotations = (dead_time // active_period_s) * active_period_s
        return click_offset, full_rotations

    def _letter_index(self, letter):
        letter_lower = letter.lower()
        if letter_lower not in kconfig.key_chars:
            letter_lower = "'"
        return kconfig.key_chars.index(letter_lower)

    @staticmethod
    def _prediction_source(selected_index):
        """Classify an actual winning word-clock index without changing it."""
        if 0 <= selected_index < kconfig.best_base_index:
            return "prefix"
        if kconfig.best_base_index <= selected_index < kconfig.argmax_word_index:
            return "best"
        if selected_index == kconfig.argmax_word_index:
            return "argmax"
        return None

    def _replace_latest_observation_with_perfect_letter(self, letter_index):
        if not self.keyboard.bc.clock_inf.observations:
            return
        low_logprob = np.log(0.01 / max(len(kconfig.key_chars) - 1, 1))
        row = [low_logprob] * len(kconfig.key_chars)
        row[letter_index] = np.log(0.99)
        self.keyboard.bc.clock_inf.observations[-1] = row

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
        self.num_successful_word_clicks_phrase = 0
        self.num_successful_word_selections_phrase = 0
        self.num_successful_word_characters_phrase = 0
        self.num_target_words_phrase = 0
        self.num_completed_words_phrase = 0
        self.num_failed_words_phrase = 0
        self.num_word_attempts_phrase = 0
        self.num_enter_retries_phrase = 0
        self.num_wrong_word_commits_phrase = 0
        self.num_undo_attempts_phrase = 0
        self.num_undo_failures_phrase = 0
        self.num_restored_states_phrase = 0
        self.num_letter_presses_phrase = 0
        self.num_target_enter_attempts_phrase = 0
        self.num_corrective_undo_actions_phrase = 0
        self.num_enter_presses_phrase = 0
        self.num_enter_misselections_phrase = 0
        self.letter_clock_time_s = 0.0
        self.target_enter_clock_time_s = 0.0
        self.undo_clock_time_s = 0.0
        self.dead_time_s_phrase = 0.0
        self.failure_reason_counts_phrase = {}
        self.failure_events_phrase = []
        self.prediction_source_counts_phrase = {
            "prefix": 0,
            "best": 0,
            "argmax": 0,
        }
        self.prediction_selection_events_phrase = []
        self.phrase_failure_reason = ""
        self.phrase_failure_stage = ""
        self.phrase_failure_limit = ""
        self.phrase_failure_guard = ""
        self.failed_target_word = ""
        self.failed_word_position = 0
        self.failed_word_attempt = 0
        self.failed_target_was_displayed = False
        self.failure_word_click_count = 0
        self.failure_letter_press_count = 0
        self.failure_target_enter_attempt_count = 0
        self.failure_undo_attempt_count = 0
        self._last_attempt_target_displayed = False
        self.start_time = self.keyboard.sim_time.time()

    def _calculate_phrase_results(self, target_phrase, phrase_type):
        typed_clean = self.keyboard.typed.rstrip()
        target_clean = target_phrase[:len(typed_clean)]
        if typed_clean:
            _, error_rate = calc_MSD(typed_clean, target_clean)
        else:
            error_rate = 100.0 if target_phrase else 0.0

        elapsed = self.keyboard.sim_time.time() - self.start_time
        n_chars = max(len(self.keyboard.typed), 1)
        n_sel = max(self.num_selections_phrase, 1)
        phrase_completed = bool(typed_clean == target_phrase.rstrip())
        accounted_time = (
            self.letter_clock_time_s
            + self.target_enter_clock_time_s
            + self.undo_clock_time_s
        )
        active_typing_time = elapsed - self.dead_time_s_phrase
        accounting_error = elapsed - accounted_time
        if abs(accounting_error) > 1e-8:
            raise RuntimeError(
                "simulated phrase time does not equal the sum of press-stage times: "
                f"elapsed={elapsed}, accounted={accounted_time}"
            )
        if not phrase_completed and not self.phrase_failure_reason:
            self.phrase_failure_reason = "final_text_mismatch"
            self.phrase_failure_stage = FAILURE_STAGES["final_text_mismatch"]
            self.phrase_failure_limit = "phrase_validation"
            self.phrase_failure_guard = "final_text_mismatch"
            self.failure_reason_counts_phrase["final_text_mismatch"] = (
                self.failure_reason_counts_phrase.get("final_text_mismatch", 0) + 1
            )
            self.failure_events_phrase.append(
                {
                    "reason": "final_text_mismatch",
                    "stage": FAILURE_STAGES["final_text_mismatch"],
                    "limit": "phrase_validation",
                    "guard": "final_text_mismatch",
                    "target_word": "",
                    "word_position": 0,
                    "word_attempt": 0,
                    "target_was_displayed": False,
                    "word_click_count": 0,
                    "letter_press_count": 0,
                    "target_enter_attempt_count": 0,
                    "undo_attempt_count": 0,
                }
            )

        click_burden = (
            self.num_successful_word_clicks_phrase
            / self.num_successful_word_selections_phrase
            if self.num_successful_word_selections_phrase > 0
            else np.nan
        )
        clicks_per_character = (
            self.num_successful_word_clicks_phrase
            / self.num_successful_word_characters_phrase
            if self.num_successful_word_characters_phrase > 0
            else np.nan
        )
        correction_rate = (
            self.num_corrective_undo_actions_phrase / self.num_completed_words_phrase
            if self.num_completed_words_phrase > 0
            else np.nan
        )
        enter_misselection_rate = (
            self.num_enter_misselections_phrase / self.num_enter_presses_phrase
            if self.num_enter_presses_phrase > 0
            else np.nan
        )
        completion_rate = (
            self.num_completed_words_phrase / self.num_target_words_phrase
            if self.num_target_words_phrase > 0
            else np.nan
        )
        prediction_selection_count = sum(
            self.prediction_source_counts_phrase.values()
        )
        if prediction_selection_count != self.num_word_selections_phrase:
            raise RuntimeError(
                "successful prediction selections were not classified exactly once"
            )

        def prediction_usage_percent(source):
            if prediction_selection_count == 0:
                return np.nan
            return (
                self.prediction_source_counts_phrase[source]
                / prediction_selection_count
                * 100
            )

        return {
            "Target Phrase": target_phrase,
            "Phrase Type": phrase_type,
            "Typed Text": self.keyboard.typed,
            # Backward-compatible name, now limited to successfully completed words.
            "Click Load (clicks/character)": round(
                clicks_per_character, sim_config.data_save_precision
            ),
            "Clicks per Character": round(
                clicks_per_character, sim_config.data_save_precision
            ),
            # Backward-compatible name, now using only eventually successful words.
            "Click Load (clicks/selection)": round(
                click_burden, sim_config.data_save_precision
            ),
            "Click Burden (clicks/successful selection)": round(
                click_burden, sim_config.data_save_precision
            ),
            "Entry Rate (cpm)": round(n_chars / elapsed * 60, sim_config.data_save_precision) if elapsed > 0 else 0,
            "Entry Rate (wpm)": round(n_chars / elapsed * 60 / 5, sim_config.data_save_precision) if elapsed > 0 else 0,
            "Correction Rate": round(correction_rate, sim_config.data_save_precision),
            "Correction Rate (%)": round(
                correction_rate * 100, sim_config.data_save_precision
            ),
            "Enter Misselection Rate": round(
                enter_misselection_rate, sim_config.data_save_precision
            ),
            "Enter Misselection Rate (%)": round(
                enter_misselection_rate * 100, sim_config.data_save_precision
            ),
            "Error Rate (%)": round(error_rate, sim_config.data_save_precision),
            "Word Prediction Usage (%)": round(self.num_word_selections_phrase / n_sel * 100,
                                               sim_config.data_save_precision),
            "Num Clicks": int(self.num_clicks_phrase),
            "Num Corrections": int(self.num_corrections_phrase),
            "Num Selections": int(self.num_selections_phrase),
            "Num Word Prediction Selections": int(self.num_word_selections_phrase),
            "Prediction Selection Count": int(prediction_selection_count),
            "Prefix Prediction Selection Count": int(
                self.prediction_source_counts_phrase["prefix"]
            ),
            "Best Prediction Selection Count": int(
                self.prediction_source_counts_phrase["best"]
            ),
            "Argmax Prediction Selection Count": int(
                self.prediction_source_counts_phrase["argmax"]
            ),
            "Prefix Prediction Usage (%)": round(
                prediction_usage_percent("prefix"), sim_config.data_save_precision
            ),
            "Best Prediction Usage (%)": round(
                prediction_usage_percent("best"), sim_config.data_save_precision
            ),
            "Argmax Prediction Usage (%)": round(
                prediction_usage_percent("argmax"), sim_config.data_save_precision
            ),
            "Prediction Selection Events": json.dumps(
                self.prediction_selection_events_phrase, sort_keys=True
            ),
            "Successful Word Click Count": int(self.num_successful_word_clicks_phrase),
            "Successful Word Selection Count": int(
                self.num_successful_word_selections_phrase
            ),
            "Successful Word Character Count": int(
                self.num_successful_word_characters_phrase
            ),
            "Target Word Count": int(self.num_target_words_phrase),
            "Completed Word Count": int(self.num_completed_words_phrase),
            "Failed Word Count": int(self.num_failed_words_phrase),
            "Word Attempt Count": int(self.num_word_attempts_phrase),
            "Enter Retry Count": int(self.num_enter_retries_phrase),
            "Wrong Word Commit Count": int(self.num_wrong_word_commits_phrase),
            "Undo Attempt Count": int(self.num_undo_attempts_phrase),
            "Undo Failure Count": int(self.num_undo_failures_phrase),
            "Restored State Count": int(self.num_restored_states_phrase),
            "Letter Press Count": int(self.num_letter_presses_phrase),
            "Target Enter Attempt Count": int(self.num_target_enter_attempts_phrase),
            "Corrective Undo Action Count": int(
                self.num_corrective_undo_actions_phrase
            ),
            "Enter Press Count": int(self.num_enter_presses_phrase),
            "Enter Misselection Count": int(self.num_enter_misselections_phrase),
            "Simulated Attempt Time (s)": float(elapsed),
            "Simulated Dead Time (s)": float(self.dead_time_s_phrase),
            "Active Typing Time (s)": float(active_typing_time),
            "Simulated Completion Time (s)": (
                float(elapsed) if phrase_completed else np.nan
            ),
            "Letter Clock Time (s)": float(self.letter_clock_time_s),
            "Target Enter Clock Time (s)": float(self.target_enter_clock_time_s),
            "Undo Clock Time (s)": float(self.undo_clock_time_s),
            "Simulated Time Accounting Error (s)": float(accounting_error),
            "Phrase Failure Reason": self.phrase_failure_reason,
            "Phrase Failure Stage": self.phrase_failure_stage,
            "Phrase Failure Limit": self.phrase_failure_limit,
            "Phrase Failure Guard": self.phrase_failure_guard,
            "Failure Reason Counts": json.dumps(
                self.failure_reason_counts_phrase, sort_keys=True
            ),
            "Failure Events": json.dumps(
                self.failure_events_phrase, sort_keys=True
            ),
            "Failed Target Word": self.failed_target_word,
            "Failed Word Position": int(self.failed_word_position),
            "Failed Word Attempt": int(self.failed_word_attempt),
            "Failed Target Was Displayed": bool(self.failed_target_was_displayed),
            "Failure Word Click Count": int(self.failure_word_click_count),
            "Failure Letter Press Count": int(self.failure_letter_press_count),
            "Failure Target Enter Attempt Count": int(
                self.failure_target_enter_attempt_count
            ),
            "Failure Undo Attempt Count": int(self.failure_undo_attempt_count),
            "Word Completion Rate (%)": round(
                completion_rate * 100,
                sim_config.data_save_precision,
            ),
            "Completion Rate": round(completion_rate, sim_config.data_save_precision),
            "Completion Fraction": round(
                min(len(self.keyboard.typed) / max(len(target_phrase), 1), 1.0),
                sim_config.data_save_precision,
            ),
            "Phrase Completed": phrase_completed,
        }

    def update_progress_bar(self, progress):
        sys.stdout.write('\r')
        sys.stdout.write("[%-50s] %d%%" % ('=' * (progress // 2), progress))
        sys.stdout.flush()
