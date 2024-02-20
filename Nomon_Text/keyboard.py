#!/usr/bin/python

import numpy as np
import os
from Nomon_Core import config
from Nomon_Core.broderclocks import BroderClocks
from Nomon_Text import kconfig
from Nomon_Text.kenlm.kenlm_lm import LanguageModel
from matplotlib import pyplot as plt

from scipy.stats import entropy

def plot_priors(words_on, word_score_prior, keys_li, N_pred, words_li, index_to_wk):
    items = []
    priors = []
    for i, index in enumerate(words_on):
        text = ""
        # if selected a key
        if (index - N_pred) % (N_pred + 1) == 0:
            text = keys_li[index_to_wk[index]]
            # if selected a word
        else:
            key = index_to_wk[index] // N_pred
            pred = index_to_wk[index] % N_pred
            text = words_li[key][pred]

        items += [text]
        priors += [word_score_prior[i]]

    item_priors = np.vstack([items, priors]).T.tolist()
    item_priors = sorted(item_priors, key=lambda x: float(x[1]))
    items = [i[0] for i in item_priors]
    priors = [float(i[1]) for i in item_priors]

    plt.barh(items, priors)
    plt.tight_layout()
    plt.show()


class SimTime:
    def __init__(self):
        self.cur_time = 0

    def time(self):
        return self.cur_time

    def set_time(self, t):
        self.cur_time = t


class Keyboard:
    def __init__(self, parent, cwd=os.getcwd(), job_num=None, sub_call=False, parameters={}, num_jobs=0):

        self.parent = parent

        self.sim_time = SimTime()
        self.is_simulation = True

        self.N_pred = kconfig.N_pred
        self.prob_thres = kconfig.prob_thres

        self.win_diff_base = config.win_diff_base
        self.rotate_index = config.default_rotate_ind
        self.time_rotate = config.period_li[self.rotate_index]

        self.key_chars = kconfig.key_chars

        # determine keyboard positions
        self.init_locs()

        # self.time = SimTime(self)
        self.prev_time = 0

        self.word_pred_on = True

        # determine keyboard positions
        # self.init_clock_locs()

        # set up "typed" text
        self.left_context = ""
        self.typed = ""
        self.btyped = ""
        self.context = ""
        self.old_context_li = [""]
        self.last_add_li = [0]

        # initialize the language model if in parameters
        if "lm_files" in parameters:
            word_lm_path, char_lm_path, vocab_path, char_path = parameters["lm_files"]
        else:
            cwd = os.getcwd()
            cwd = os.path.dirname(cwd)
            word_lm_path = os.path.join(os.path.join(cwd, 'Nomon_Text/resources'),
                                        'lm_word_dec19.kenlm')
            char_lm_path = os.path.join(os.path.join(cwd, 'Nomon_Text/resources'), 'lm_char_dec19.kenlm')
            vocab_path = os.path.join(os.path.join(cwd, 'Nomon_Text/resources'), 'vocab_lower_100k.txt')
            char_path = os.path.join(os.path.join(cwd, 'Nomon_Text/resources'), 'char_set.txt')

        self.lm = LanguageModel(word_lm_path, char_lm_path, vocab_path, char_path)

        self.init_words()

        self.clear_text = False

        # generate prior for clocks
        self.gen_word_prior(False)

        self.clock_spaces = np.zeros((len(self.clock_centers), 2))

        self.bc = BroderClocks(self)

        self.bc.init_follow_up(self.word_score_prior)

        self.clock_params = np.zeros((len(self.clock_centers), 8))

        self.consent = False

    def clock_to_text(self, index):

        if (index - self.N_pred) % (self.N_pred + 1) == 0:
            typed = self.keys_li[self.index_to_wk[index]]
        else:
            key = self.index_to_wk[index] // self.N_pred
            pred = self.index_to_wk[index] % self.N_pred
            typed = self.words_li[key][pred]
        return typed

    def init_locs(self):
        # size of keyboard
        self.N_rows = len(self.key_chars)
        self.N_keys_row = []
        self.N_keys = 0
        self.N_alpha_keys = 0
        for row in range(0, self.N_rows):
            n_keys = len(self.key_chars[row])
            for col in range(0, n_keys):
                if not isinstance(self.key_chars[row][col], list):
                    if self.key_chars[row][col].isalpha() and (len(self.key_chars[row][col]) == 1):
                        self.N_alpha_keys = self.N_alpha_keys + 1
                    elif self.key_chars[row][col] == kconfig.space_char and (len(self.key_chars[row][col]) == 1):
                        self.N_alpha_keys = self.N_alpha_keys + 1
                    elif self.key_chars[row][col] == kconfig.break_chars[0] and (
                            len(self.key_chars[row][col]) == 1):
                        self.N_alpha_keys = self.N_alpha_keys + 1

            self.N_keys_row.append(n_keys)
            self.N_keys += n_keys

        # print "NKEYS is " + str(self.N_keys)
        # print "And N_alpha_keys is " + str(self.N_alpha_keys)

        # width difference when include letter
        word_clock_offset = 7 * kconfig.clock_rad
        rect_offset = word_clock_offset - kconfig.clock_rad
        word_offset = 8.5 * kconfig.clock_rad
        rect_end = rect_offset + kconfig.word_w

        # clock, key, word locations
        self.clock_centers = []
        self.win_diffs = []
        self.word_locs = []
        self.char_locs = []
        self.rect_locs = []
        self.keys_li = []
        self.keys_ref = []
        index = 0  # how far into the clock_centers matrix
        word = 0  # word index
        key = 0  # key index
        # self.N_pred = 2 # number of words per key
        self.key_height = 6.5 * kconfig.clock_rad
        self.w_canvas = 0
        self.index_to_wk = []  # overall index to word or key index
        for row in range(0, self.N_rows):
            y = row * self.key_height
            self.w_canvas = max(self.w_canvas, self.N_keys_row[row] * (6 * kconfig.clock_rad + kconfig.word_w))
            for col in range(0, self.N_keys_row[row]):
                x = col * (6 * kconfig.clock_rad + kconfig.word_w)
                # predictive words
                for word_index in range(self.N_pred):
                    self.clock_centers.append([x + word_clock_offset, y + (1 + word_index * 2) * kconfig.clock_rad])
                    self.word_locs.append([x + word_offset, y + (1 + word_index * 2) * kconfig.clock_rad])
                    # self.clock_centers.append([x + word_clock_offset, y + 3 * kconfig.clock_rad])
                    # self.clock_centers.append([x + word_clock_offset, y + 5 * kconfig.clock_rad])
                    self.index_to_wk.append(word + word_index)
                # win diffs
                self.win_diffs.extend([self.win_diff_base for i in range(self.N_pred)])
                # word position
                # self.word_locs.append([x + word_offset, y + 1 * kconfig.clock_rad])
                # self.word_locs.append([x + word_offset, y + 3 * kconfig.clock_rad])
                # self.word_locs.append([x + word_offset, y + 5 * kconfig.clock_rad])
                # rectangles
                self.rect_locs.append([x + rect_offset, y, x + rect_end, y + 2 * kconfig.clock_rad])
                self.rect_locs.append(
                    [x + rect_offset, y + 2 * kconfig.clock_rad, x + rect_end, y + 4 * kconfig.clock_rad])
                self.rect_locs.append(
                    [x + rect_offset, y + 4 * kconfig.clock_rad, x + rect_end, y + 6 * kconfig.clock_rad])
                # indices

                # self.index_to_wk.append(word)
                # self.index_to_wk.append(word + 1)
                # self.index_to_wk.append(word + 2)
                index += self.N_pred
                word += self.N_pred

                ## key character
                # reference to index of key character
                key_char = self.key_chars[row][col]
                self.keys_li.append(self.key_chars[row][col])
                self.keys_ref.append(index)
                self.index_to_wk.append(key)
                # key character position
                self.char_locs.append([x + 2 * kconfig.clock_rad, y + 3 * kconfig.clock_rad])
                # clock position for key character
                self.clock_centers.append([x + 1 * kconfig.clock_rad, y + 3 * kconfig.clock_rad])
                # rectangles
                self.rect_locs.append([x, y, x + rect_offset, y + 6 * kconfig.clock_rad])
                # win diffs
                if (key_char == kconfig.mybad_char) or (key_char == kconfig.yourbad_char) or (
                        key_char == kconfig.back_char):  # or (key_char == kconfig.break_char)
                    self.win_diffs.append(config.win_diff_high)
                else:
                    self.win_diffs.append(self.win_diff_base)
                index += 1
                key += 1

    def change_speed(self):
        self.bc.clock_inf.clock_util.change_period(self.time_rotate)

    def init_words(self):
        (self.words_li, self.word_freq_li, self.key_freq_li) = self.lm.get_words(self.left_context, self.context,
                                                                                 self.keys_li,
                                                                                 num_words_total=kconfig.num_words_total)

        self.word_id = []
        self.word_pair = []
        word = 0
        index = 0
        windex = 0
        self.words_on = []
        self.words_off = []
        self.word_list = []
        # self.flag_args = []

        if self.word_pred_on == True:
            temp_word_list = [word_item for sublist in self.words_li for word_item in sublist]
            for word_item in temp_word_list:
                if word_item != '':
                    self.word_list.append(word_item)

            # print "TURNED ON AND WORD LIST IS" + str(self.word_list)

        len_con = len(self.context)
        for key in range(0, self.N_alpha_keys):
            for pred in range(0, self.N_pred):
                word_str = self.words_li[key][pred]
                len_word = len(word_str)
                if (len_con > 1) and (len_word > kconfig.max_chars_display):
                    word_str = "+" + word_str[len_con:len_word]
                self.word_pair.append((key, pred))
                if word_str == '':
                    self.words_off.append(index)
                else:
                    # word predictions on
                    if self.word_pred_on:
                        self.words_on.append(index)
                    else:
                        self.words_off.append(index)

                windex += 1
                word += 1
                index += 1
            self.words_on.append(index)
            self.word_pair.append((key,))
            index += 1
        for key in range(self.N_alpha_keys, self.N_keys):
            for pred in range(0, self.N_pred):
                word_str = self.words_li[key][pred]
                self.word_pair.append((key, pred))
                self.words_off.append(index)
                index += 1
            self.words_on.append(index)
            self.word_pair.append((key,))
            index += 1
        self.typed_versions = ['']

    def draw_words(self):
        (self.words_li, self.word_freq_li, self.key_freq_li) = self.lm.get_words(self.left_context, self.context,
                                                                                 self.keys_li,
                                                                                 num_words_total=kconfig.num_words_total)
        word = 0
        index = 0
        self.words_on = []
        self.words_off = []
        self.word_list = []

        # if word prediction on
        if self.word_pred_on:
            temp_word_list = [word_item for sublist in self.words_li for word_item in sublist]
            for word_item in temp_word_list:
                if word_item != '':
                    self.word_list.append(word_item)

        len_con = len(self.context)

        windex = 0
        for key in range(0, self.N_alpha_keys):
            for pred in range(0, self.N_pred):
                word_str = self.words_li[key][pred]
                len_word = len(word_str)
                if len_con > 1 and len_word > kconfig.max_chars_display:
                    word_str = "+" + word_str[len_con:len_word]
                if word_str == '':
                    self.words_off.append(index)
                else:
                    # word predictions on
                    if self.word_pred_on:
                        self.words_on.append(index)
                    else:
                        self.words_off.append(index)



                windex += 1
                word += 1
                index += 1
            self.words_on.append(index)

            self.word_pair.append((key,))
            index += 1
        for key in range(self.N_alpha_keys, self.N_keys):
            for pred in range(0, self.N_pred):
                self.word_pair.append((key, pred))
                self.words_off.append(index)
                index += 1
            self.words_on.append(index)
            self.word_pair.append((key,))
            index += 1

        # self.mainWidget.update_clocks()
        # self.init_clocks()

    def gen_word_prior(self, is_undo):
        self.word_score_prior = []
        N_on = len(self.words_on)
        if not is_undo:
            for index in self.words_on:
                pair = self.word_pair[index]
                # word case
                if len(pair) == 2:
                    (key, pred) = pair
                    prob = self.word_freq_li[key][pred] + np.log(kconfig.rem_prob)
                    self.word_score_prior.append(prob)
                else:
                    key = pair[0]
                    prob = self.key_freq_li[key]
                    prob = prob + np.log(kconfig.rem_prob)
                    if self.keys_li[key] == kconfig.mybad_char or self.keys_li[key] == kconfig.yourbad_char:
                        prob = np.log(kconfig.undo_prob)
                    # if self.keys_li[key] in kconfig.break_chars:
                    #     prob = np.log(kconfig.break_prob)
                    if self.keys_li[key] == kconfig.back_char:
                        prob = np.log(kconfig.back_prob)
                    if self.keys_li[key] == kconfig.clear_char:
                        prob = np.log(kconfig.undo_prob)

                    self.word_score_prior.append(prob)
        else:
            for index in self.words_on:
                pair = self.word_pair[index]
                if len(pair) == 1:
                    key = pair[0]
                    if (self.keys_li[key] == kconfig.mybad_char) or (self.keys_li[key] == kconfig.yourbad_char):
                        prob = kconfig.undo_prob
                        self.word_score_prior.append(np.log(prob))
                    else:
                        self.word_score_prior.append(0)
                else:
                    self.word_score_prior.append(0)

    def increment_clocks(self):
        self.bc.clock_inf.clock_util.increment(self.words_on)

    def on_press(self):
        self.bc.select()

    def make_choice(self, index):

        is_undo = False
        is_equalize = False

        # highlight winner
        self.previous_winner = index
        # self.highlight_winner(index)

        # save clock score distribution entropy before new BC round begins
        if self.parent.track_entropy:
            self.parent.save_clock_cscores(-1)

        # initialize talk string
        talk_string = ""

        # if selected a key
        if (index - self.N_pred) % (self.N_pred + 1) == 0:
            new_char = self.keys_li[self.index_to_wk[index]]
            # special characters
            if new_char == kconfig.space_char:
                if len(self.context) > 1:
                    talk_string = self.context
                else:
                    talk_string = "space"

                new_char = ' '
                self.old_context_li.append(self.context)
                self.context = ""
                self.last_add_li.append(1)
            elif new_char == kconfig.mybad_char or new_char == kconfig.yourbad_char:
                talk_string = new_char
                # if added characters that turn
                if len(self.last_add_li) > 1:
                    last_add = self.last_add_li.pop()
                    self.context = self.old_context_li.pop()
                    if last_add > 0:  # if added text that turn
                        self.typed = self.typed[0:-last_add]
                    elif last_add == -1:  # if backspaced that turn
                        letter = self.btyped[-1]
                        self.btyped = self.btyped[0:-1]
                        self.typed += letter
                if new_char == kconfig.yourbad_char:
                    is_equalize = True
                new_char = ''
                is_undo = True
            elif new_char == kconfig.back_char:
                talk_string = new_char
                # if delete the last character that turn
                self.old_context_li.append(self.context)
                # print(self.context)
                lt = len(self.typed)
                if lt > 0:  # typed anything yet?
                    self.btyped += self.typed[-1]
                    self.last_add_li.append(-1)
                    self.typed = self.typed[0:-1]
                    lt -= 1
                    if lt == 0:
                        self.context = ""
                    elif len(self.context) > 0:
                        self.context = self.context[0:-1]
                    elif not (self.typed[-1]).isalpha():
                        self.context = ""
                    else:
                        i = -1
                        while (i >= -lt) and (self.typed[i].isalpha()):
                            i -= 1
                        self.context = self.typed[i + 1:lt]
                new_char = ''
            elif new_char == kconfig.clear_char:
                talk_string = 'clear'

                new_char = '_'
                self.old_context_li.append(self.context)
                self.context = ""
                self.last_add_li.append(1)

                self.clear_text = True

            elif new_char.isalpha() or new_char == "'":
                talk_string = new_char
                self.old_context_li.append(self.context)
                self.context += new_char
                self.last_add_li.append(1)

            if new_char in [".", ",", "?", "!"]:
                talk_string = "Full stop"
                self.old_context_li.append(self.context)
                self.context = ""
                self.typed += new_char
                if " " + new_char in self.typed:
                    self.last_add_li.append(1)
                else:
                    self.last_add_li.append(1)
                self.typed = self.typed.replace(" " + new_char, new_char + " ")
            else:
                self.typed += new_char

        # if selected a word
        else:
            key = self.index_to_wk[index] // self.N_pred
            pred = self.index_to_wk[index] % self.N_pred
            new_word = self.words_li[key][pred]
            new_selection = new_word
            length = len(self.context)
            talk_string = new_word.rstrip(kconfig.space_char)  # talk string
            if length > 0:
                self.typed = self.typed[0:-length]
            self.typed += new_word
            self.last_add_li.append(len(new_word) - len(self.context))
            self.old_context_li.append(self.context)
            self.context = ""

        # update the screen
        if self.context != "":
            self.left_context = self.typed[:-len(self.context)]
        else:
            self.left_context = self.typed

        self.draw_words()
        # self.draw_typed()
        # update the word prior
        self.gen_word_prior(is_undo)
        # print(self.typed)
        # plot_priors(self.words_on, self.word_score_prior, self.keys_li, self.N_pred, self.words_li, self.index_to_wk)


        return self.words_on, self.words_off, self.word_score_prior, is_undo, is_equalize
