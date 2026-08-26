from __future__ import division
from dataclasses import dataclass
import math

from numpy import ceil

from OneClick_Core import config
from OneClick_Core.broderclocks import BroderClocks
from OneClick_Core.clock_util import ClockUtil, SpacedArray, HourLocs
from OneClick_Text import kconfig
from OneClick_Text.language_model import LanguageModel


def _logprob(item):
    if "logprob" in item:
        return float(item["logprob"])
    if "logProb" in item:
        return float(item["logProb"])
    return -99.0


def _argmax(row):
    return max(range(len(row)), key=lambda i: row[i])


@dataclass
class WordAttemptSnapshot:
    """Deep-copied state needed to retry a decoded word after correction."""

    observations: list
    words_by_letter: dict
    best_words: list
    argmax_word: str
    valid_word_indices: list
    word_clock_phases: dict
    typed: str
    context: str
    typed_versions: list
    pending_delay_samples: list


class SimTime:
    def __init__(self):
        self.cur_time = 0.0

    def time(self):
        return self.cur_time

    def set_time(self, t):
        self.cur_time = t


class WordClockUtil:

    def __init__(self, time_rotate, delay_model):
        self.time_rotate = time_rotate
        # Space and Enter use the same physical switch, so word-clock selection
        # must use the same calibrated click-delay model as letter observations.
        self.delay_model = delay_model
        self.num_divs_time = int(ceil(time_rotate / config.ideal_wait_s))
        self.spaced = SpacedArray(self.num_divs_time)
        self.hl = HourLocs(self.num_divs_time)
        self.cur_hours = {}        # logical word index -> current phase (int)
        self.latest_time = 0.0
        self.last_selected_loc = 0.0

    def init_round(self, valid_indices):
        """Equispace the active word-clock indices over the clock face."""
        self.cur_hours = {}
        n_active = len(valid_indices)
        if n_active == 0:
            return
        for k, idx in enumerate(valid_indices):
            self.cur_hours[idx] = int(k * self.num_divs_time / n_active) % self.num_divs_time

    def increment(self, time_diff):
        steps = int(time_diff / self.time_rotate * self.num_divs_time)
        for i in self.cur_hours:
            self.cur_hours[i] = (self.cur_hours[i] + steps) % self.num_divs_time

    def change_period(self, new_period):
        self.time_rotate = new_period
        self.num_divs_time = int(ceil(new_period / config.ideal_wait_s))
        self.spaced = SpacedArray(self.num_divs_time)
        self.hl = HourLocs(self.num_divs_time)

    def select_word(self, time_diff_in, valid_indices):
        """
        Based on clock_inference_engine_word.js select_word(). Compensate for the
        user's learned switch delay before comparing word-clock distances.
        """
        offset = self.delay_model.get_offset()
        best_idx = valid_indices[0] if valid_indices else kconfig.undo_word_index
        best_dist = float('inf')
        best_loc = 0.0
        for idx in valid_indices:
            if idx not in self.cur_hours:
                continue
            time_in = (self.cur_hours[idx] * self.time_rotate / self.num_divs_time
                       + time_diff_in
                       - self.time_rotate * config.frac_period)
            dist = abs(time_in - offset)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
                best_loc = time_in
        self.last_selected_loc = best_loc
        return best_idx


class Keyboard:

    def __init__(self, parent, parameters=None):
        if parameters is None:
            parameters = {}
        self.parent = parent
        self.sim_time = SimTime()
        self.is_simulation = True
        use_click_offset = parameters.get(
            "use_click_offset", config.use_click_offset
        )
        if not isinstance(use_click_offset, bool):
            raise ValueError("use_click_offset must be a boolean")
        self.use_click_offset = use_click_offset
        self.delay_learning_mode = parameters.get(
            "delay_learning_mode", "enter_only"
        )
        if self.delay_learning_mode not in {"enter_only", "all_confirmed_clicks"}:
            raise ValueError(
                "delay_learning_mode must be 'enter_only' or "
                "'all_confirmed_clicks'"
            )
        self.word_clock_mode = parameters.get("word_clock_mode", "fixed")
        if self.word_clock_mode not in {"fixed", "adaptive"}:
            raise ValueError("word_clock_mode must be 'fixed' or 'adaptive'")
        sigma_margin = parameters.get("sigma_margin")
        self.sigma_margin = None if sigma_margin is None else float(sigma_margin)
        if self.word_clock_mode == "adaptive" and (
            self.sigma_margin is None
            or not math.isfinite(self.sigma_margin)
            or self.sigma_margin <= 0
        ):
            raise ValueError(
                "adaptive word-clock mode requires a positive finite sigma_margin"
            )

        # Clock periods. The legacy fixed period applies to both selection
        # stages; the specialized parameters must always be supplied together.
        legacy_period = parameters.get("fixed_clock_period_s")
        fixed_space_period = parameters.get("fixed_space_clock_period_s")
        fixed_enter_period = parameters.get("fixed_enter_clock_period_s")
        has_specialized_period = (
            fixed_space_period is not None or fixed_enter_period is not None
        )
        if legacy_period is not None and has_specialized_period:
            raise ValueError(
                "fixed_clock_period_s cannot be combined with specialized "
                "Space/Enter clock periods"
            )
        if (fixed_space_period is None) != (fixed_enter_period is None):
            raise ValueError(
                "fixed_space_clock_period_s and fixed_enter_clock_period_s "
                "must be supplied together"
            )
        if legacy_period is not None:
            fixed_space_period = legacy_period
            fixed_enter_period = legacy_period

        if fixed_space_period is None:
            self.rotate_index = config.default_rotate_ind
            self.time_rotate = float(config.period_li[self.rotate_index])
            word_time_rotate = self.time_rotate
        else:
            fixed_space_period = float(fixed_space_period)
            fixed_enter_period = float(fixed_enter_period)
            if (
                not math.isfinite(fixed_space_period)
                or fixed_space_period <= 0
                or not math.isfinite(fixed_enter_period)
                or fixed_enter_period <= 0
            ):
                raise ValueError(
                    "fixed Space/Enter clock periods must be positive finite values"
                )
            self.rotate_index = int(
                min(
                    range(len(config.period_li)),
                    key=lambda index: abs(
                        float(config.period_li[index]) - fixed_space_period
                    ),
                )
            )
            self.time_rotate = fixed_space_period
            word_time_rotate = fixed_enter_period

        # Letter clocks: one per key char (27 total)
        self.clock_centers = list(range(len(kconfig.key_chars)))

        # Text state
        self.typed = ""
        self.typed_versions = []     # undo stack: previous typed strings
        self.context = ""            # left context for the language model

        # Language model
        self.lm = LanguageModel(parameters.get("oneclick_lm_config"))

        # Word-clock content for the current click prefix (rebuilt each API call):
        self.words_by_letter = {}    # next-letter -> [prefix completion text, ...] (<= n_pred)
        self.best_words = []         # EOW BEST decodings (exact click length), <= n_best
        self.argmax_word = ""        # literal per-click argmax decode (no API correction)
        self.valid_word_indices = [] # populated prefix slots + best slots + argmax + undo

        # Letter BroderClocks
        self.bc = BroderClocks(self)
        self.bc.init_follow_up()
        self.place_letter_clocks()

        # Word clock utility (Enter-level selection)
        self.word_clock_util = WordClockUtil(
            word_time_rotate,
            self.bc.clock_inf.delay_model,
        )
        self.word_clock_util.init_round([])

        self._last_enter_time_in = None
        self._last_commit_updated_delay = False
        self._pending_delay_samples = []

    # ------------------------------------------------------------------
    # Letter-clock placement (LM letter prior, once per word — matches Nomon)
    # ------------------------------------------------------------------

    def place_letter_clocks(self):

        ci = self.bc.clock_inf
        key_probs = self.lm.get_key_probs(self.context)
        for i, idx in enumerate(ci.clocks_li):
            ci.cscores[idx] = key_probs[i] if i < len(key_probs) else 0.0
        ci.update_sorted_inds()
        ci.clock_util.update_curhours(ci.sorted_inds)

    # ------------------------------------------------------------------
    # Clock advancement (called by SimulatedUser before each press)
    # ------------------------------------------------------------------

    def increment_clocks(self):
        """Advance letter clock phases based on elapsed sim_time."""
        time_diff = self.sim_time.time() - self.bc.latest_time
        self.bc.clock_inf.clock_util.increment(time_diff)
        self.bc.latest_time = self.sim_time.time()

    def increment_word_clocks(self):
        """Advance word clock phases based on elapsed sim_time."""
        time_diff = self.sim_time.time() - self.word_clock_util.latest_time
        self.word_clock_util.increment(time_diff)
        self.word_clock_util.latest_time = self.sim_time.time()

    # ------------------------------------------------------------------
    # Space press (letter click)
    # ------------------------------------------------------------------

    def on_press(self, target_letter_index=None):
        """Simulate a Space press: append an observation row in bc."""
        target_time_in = self.bc.select(target_letter_index)
        if (
            self.delay_learning_mode == "all_confirmed_clicks"
            and target_time_in is not None
        ):
            self._pending_delay_samples.append(target_time_in)

    # ------------------------------------------------------------------
    # Word list management
    # ------------------------------------------------------------------

    def update_word_list(self):
        """
        Based on oneclick/keyboard.js and update_inline_word_clocks(). Query the word API with the current observations and rebuild the word-clock
        content (words_by_letter / best_words / argmax_word) and valid_word_indices.
        Then equispace the active word clocks. 
        """
        ci = self.bc.clock_inf
        obs = ci.observations
        obs_len = len(obs)
        observations = ci.format_observations(kconfig.key_chars)
        prefix, best = self.lm.get_word_predictions(self.context, observations)

        # Prefix completions -> letter cells, filed under the NEXT letter (charAt(obs_len)).
        # A completion no longer than the click prefix has no next letter -> skipped.
        self.words_by_letter = {}
        seen = set()
        prefix_clock_count = 0
        ranked_prefix_indices = []
        for item in sorted(prefix, key=_logprob, reverse=True):
            if prefix_clock_count >= kconfig.max_prefix_clocks:
                break
            t = item.get("text", "")
            if not t or len(t) <= obs_len:
                continue
            tl = t.lower()
            if tl in seen:
                continue
            nxt = t[obs_len].lower()
            if nxt not in kconfig.key_chars:
                continue
            bucket = self.words_by_letter.setdefault(nxt, [])
            if len(bucket) < kconfig.n_pred:
                slot = len(bucket)
                bucket.append(t)
                seen.add(tl)
                prefix_clock_count += 1
                letter_index = kconfig.key_chars.index(nxt)
                ranked_prefix_indices.append(
                    letter_index * kconfig.n_pred + slot
                )

        # BEST decodings -> EOW cell. Error-corrected readings exactly as long as the
        # click count.
        self.best_words = []
        seen_b = set()
        for item in sorted(best, key=_logprob, reverse=True):
            t = item.get("text", "")
            if not t or len(t) != obs_len:
                continue
            if t.lower() in seen_b:
                continue
            self.best_words.append(t)
            seen_b.add(t.lower())
            if len(self.best_words) >= kconfig.n_best:
                break

        # Argmax-actual-typed: the literal per-click argmax decode (commit exactly what
        # was clicked, with no API correction). Available after >= 1 click.
        self.argmax_word = ""
        if obs_len > 0:
            self.argmax_word = "".join(kconfig.key_chars[_argmax(row)] for row in obs)

        # Build valid_word_indices over the fixed logical index space. Fixed mode
        # retains its existing layout; adaptive mode uses the requested ranking.
        if getattr(self, "word_clock_mode", "fixed") == "adaptive":
            prediction_priority = []
            for index in range(max(len(self.best_words), len(ranked_prefix_indices))):
                if index < len(self.best_words):
                    prediction_priority.append(kconfig.best_base_index + index)
                if index < len(ranked_prefix_indices):
                    prediction_priority.append(ranked_prefix_indices[index])
            n_max = self._adaptive_word_clock_limit()
            valid = prediction_priority[: n_max - 2]
            valid.extend([kconfig.argmax_word_index, kconfig.undo_word_index])
        else:
            valid = []
            for letter, bucket in self.words_by_letter.items():
                li = kconfig.key_chars.index(letter)
                for slot in range(len(bucket)):
                    valid.append(li * kconfig.n_pred + slot)
            for i in range(len(self.best_words)):
                valid.append(kconfig.best_base_index + i)
            if self.argmax_word:
                valid.append(kconfig.argmax_word_index)
            valid.append(kconfig.undo_word_index)
        self.valid_word_indices = valid

        self.word_clock_util.init_round(valid)
        self.word_clock_util.latest_time = self.sim_time.time()

    def _adaptive_word_clock_limit(self):
        """Calculate the current adaptive word-clock capacity."""
        current_sigma = math.sqrt(self.bc.clock_inf.delay_model.sigma2)
        n_max = math.floor(
            self.word_clock_util.time_rotate
            / (2 * self.sigma_margin * current_sigma)
        )
        return max(2, min(8, n_max))

    def clock_to_word(self, index):
        """
        Resolve a winning word-clock logical index to the string it commits.
        Returns None for the undo clock or an empty/missing slot. Analogue of
        Nomon's clock_to_text; based on oneclick/keyboard.js:355-374.
        """
        if index == kconfig.undo_word_index:
            return None
        if index == kconfig.argmax_word_index:
            return self.argmax_word or None
        if kconfig.best_base_index <= index < kconfig.argmax_word_index:
            i = index - kconfig.best_base_index
            return self.best_words[i] if i < len(self.best_words) else None
        # prefix slot
        letter_idx = index // kconfig.n_pred
        slot = index % kconfig.n_pred
        if letter_idx >= len(kconfig.key_chars):
            return None
        letter = kconfig.key_chars[letter_idx]
        bucket = self.words_by_letter.get(letter, [])
        return bucket[slot] if slot < len(bucket) else None

    def word_clock_index(self, target_word):
        """
        Return the logical word-clock index whose clock currently holds target_word
        (BEST slot > argmax > prefix slot), or kconfig.undo_word_index if no active
        clock holds it. Used by the simulator's next_target.
        """
        tw = target_word.lower()
        for i, w in enumerate(self.best_words):
            index = kconfig.best_base_index + i
            if w.lower() == tw and index in self.valid_word_indices:
                return index
        if (
            self.argmax_word
            and self.argmax_word.lower() == tw
            and kconfig.argmax_word_index in self.valid_word_indices
        ):
            return kconfig.argmax_word_index
        for letter, bucket in self.words_by_letter.items():
            for slot, w in enumerate(bucket):
                index = kconfig.key_chars.index(letter) * kconfig.n_pred + slot
                if w.lower() == tw and index in self.valid_word_indices:
                    return index
        return kconfig.undo_word_index

    # ------------------------------------------------------------------
    # Enter press (word commit)
    # ------------------------------------------------------------------

    def on_enter(self, valid_word_indices=None):
        """
        Simulate an Enter press. Returns (committed_word_text_or_None, selected_index).
        Records the selected clock's click time for the delay-model update on commit.
        """
        if valid_word_indices is None:
            valid_word_indices = list(self.valid_word_indices)

        time_in = self.sim_time.time()
        # increment_word_clocks() has already advanced cur_hours to the current time,
        # so the time_diff handed to select_word is 0.
        time_diff = time_in - self.word_clock_util.latest_time
        selected_index = self.word_clock_util.select_word(time_diff, valid_word_indices)
        self._last_enter_time_in = self.word_clock_util.last_selected_loc

        if selected_index == kconfig.undo_word_index:
            return None, selected_index
        return self.clock_to_word(selected_index), selected_index

    # ------------------------------------------------------------------
    # Word-attempt state preservation
    # ------------------------------------------------------------------

    def capture_word_attempt_state(self):
        """Capture an independent snapshot of the decoded pre-Enter state."""
        return WordAttemptSnapshot(
            observations=[list(row) for row in self.bc.clock_inf.observations],
            words_by_letter={
                letter: list(words) for letter, words in self.words_by_letter.items()
            },
            best_words=list(self.best_words),
            argmax_word=self.argmax_word,
            valid_word_indices=list(self.valid_word_indices),
            word_clock_phases=dict(self.word_clock_util.cur_hours),
            typed=self.typed,
            context=self.context,
            typed_versions=list(self.typed_versions),
            pending_delay_samples=list(
                getattr(self, "_pending_delay_samples", ())
            ),
        )

    def restore_word_attempt_state(self, snapshot):
        """Restore a pre-Enter snapshot without replaying elapsed correction time."""
        self.bc.clock_inf.observations = [list(row) for row in snapshot.observations]
        self.words_by_letter = {
            letter: list(words) for letter, words in snapshot.words_by_letter.items()
        }
        self.best_words = list(snapshot.best_words)
        self.argmax_word = snapshot.argmax_word
        self.valid_word_indices = list(snapshot.valid_word_indices)
        self.word_clock_util.cur_hours = dict(snapshot.word_clock_phases)
        self.word_clock_util.latest_time = self.sim_time.time()
        self.typed = snapshot.typed
        self.context = snapshot.context
        self.typed_versions = list(snapshot.typed_versions)
        self._pending_delay_samples = list(snapshot.pending_delay_samples)
        self._last_enter_time_in = None
        self._last_commit_updated_delay = False

    def prepare_undo_round(self, undo_only=False):
        """Build either the protected competing or Undo-only correction display."""
        if not undo_only:
            self.update_word_list()
            return
        self.words_by_letter = {}
        self.best_words = []
        self.argmax_word = ""
        self.valid_word_indices = [kconfig.undo_word_index]
        self.word_clock_util.init_round(self.valid_word_indices)
        self.word_clock_util.latest_time = self.sim_time.time()

    # ------------------------------------------------------------------
    # Committing / undoing
    # ------------------------------------------------------------------

    def commit_word(self, text, confirmed_correct=False):
        """Commit a word and train timing only when the target is confirmed correct."""
        self.typed_versions.append(self.typed)
        self.typed += text + " "
        self.context = self.typed

        self._last_commit_updated_delay = bool(
            confirmed_correct and self._last_enter_time_in is not None
        )
        if self._last_commit_updated_delay:
            delay_model = self.bc.clock_inf.delay_model
            if (
                getattr(self, "delay_learning_mode", "enter_only")
                == "all_confirmed_clicks"
            ):
                delay_model.update_many(
                    (*self._pending_delay_samples, self._last_enter_time_in)
                )
            else:
                delay_model.update(self._last_enter_time_in)
        self._last_enter_time_in = None

        self._reset_letter_round()

    def undo_word(self):
        """Revert the last committed word and roll back the delay model."""
        if self.typed_versions:
            self.typed = self.typed_versions.pop()
            self.context = self.typed
        # Wrong/unconfirmed commits do not update the delay model, so undoing one
        # must not roll back the most recent valid calibration sample.
        if getattr(self, "_last_commit_updated_delay", False):
            self.bc.clock_inf.delay_model.rollback()
        self._last_commit_updated_delay = False
        self._reset_letter_round()

    def _reset_letter_round(self):
        """Clear observations and re-place letter clocks for the next word."""
        self._pending_delay_samples = []
        self.bc.clock_inf.reset_observations()
        self.bc.latest_time = self.sim_time.time()
        self.place_letter_clocks()
        # Clear word-clock content until the next API call.
        self.words_by_letter = {}
        self.best_words = []
        self.argmax_word = ""
        self.valid_word_indices = []
        self.word_clock_util.init_round([])

    # ------------------------------------------------------------------
    # Speed change
    # ------------------------------------------------------------------

    def change_speed(self):
        self.time_rotate = config.period_li[self.rotate_index]
        self.bc.change_speed()
        self.word_clock_util.change_period(self.time_rotate)
