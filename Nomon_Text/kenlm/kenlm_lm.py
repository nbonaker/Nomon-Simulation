#!/usr/bin/python

# This is an example script to run the WordPredictor class and get
# a list of words that begins with the prefix and a each of the valid
# character in the character list appened to it. For example, given
# a prefix 'a' and if the chracters in the vocabulary are [a,b,c], it
# will return a list of words that begin with 'aa', 'ab' and 'ac'.

import os, sys
import numpy as np
from Nomon_Text import kconfig

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Nomon_Text.kenlm.predictor import WordPredictor
from Nomon_Text.kenlm.char_predictor import CharacterPredictor


def lognormalize_factor(x):
    a = np.logaddexp.reduce(x.flatten())
    return a


class LanguageModel():
    def __init__(self, word_lm_filename, char_lm_filename, vocab_filename, char_filename, parent=None):
        self.lm_filename = word_lm_filename
        self.vocab_filename = vocab_filename

        self.word_predictor = WordPredictor(word_lm_filename, vocab_filename)
        self.char_predictor = CharacterPredictor(char_lm_filename, char_filename)

        # Define how many predictions you want for each character
        # By default it is set to 0 and will return all possible
        # words
        if parent is not None:
            self.parent = parent
            self.num_predictions = self.parent.N_pred
            self.prob_thres = self.parent.prob_thres
            self.num_words_total = self.parent.num_words_total
        else:
            self.parent = None
            self.num_predictions = kconfig.N_pred
            self.prob_thres = 0
            self.num_words_total = 26*self.num_predictions
        self.min_log_prob = -float("inf")

        # The default vocab_id is ''
        self.vocab_id = ''

    def get_words(self, context, prefix, keys_li, num_words_total=kconfig.num_words_total):
        # num_words_total = 0
        if self.parent is not None:
            self.num_predictions = self.parent.N_pred
            self.prob_thres = self.parent.prob_thres
            self.num_words_total = self.parent.num_words_total

        self.context = context
        self.prefix = prefix
        # print("prefix: ", prefix, ", context: ", context)

        word_preds = []
        word_probs = []

        lm_results = self.word_predictor.get_words_with_context(prefix, context, self.vocab_id, self.num_predictions,
                                                                self.min_log_prob)
        flattened_results = [freq for sublist in lm_results for freq in sublist]
        flattened_results.sort(key=lambda x: -x[1])
        flattened_results = [word_pair[0] for word_pair in flattened_results[:num_words_total]]

        # print(flattened_results)

        word_dict = {}
        for word_list in lm_results:
            if len(word_list) > 0:
                cur_word_list = []
                for word_pair in word_list:
                    if word_pair[0] in flattened_results:
                        cur_word_list.append(word_pair)
                word_dict[word_list[0][0][len(prefix)]] = cur_word_list

        for key in keys_li:
            key_word_preds = ["", "", ""]
            key_word_probs = [-float("inf"), -float("inf"), -float("inf")]
            if key in word_dict:
                index = 0
                for word_tuple in word_dict[key]:
                    if word_tuple[1] >= self.min_log_prob:
                        key_word_preds[index] = word_tuple[0] + " "
                        key_word_probs[index] = word_tuple[1]
                    index += 1
            word_preds += [key_word_preds]
            word_probs += [key_word_probs]


        key_probs = np.array(self.get_char_probs(context, prefix, keys_li))

        word_probs = np.array(word_probs)

        # key_probs = np.log(np.ones(key_probs.shape)/key_probs.shape[0])
        # key_probs = -np.ones(key_probs.shape)
        # word_probs = np.where(word_probs != -float("inf"), -1, -float("inf"))

        joint_normalize_factor = lognormalize_factor(np.hstack([key_probs.flatten(), word_probs.flatten()]))

        key_probs = key_probs - joint_normalize_factor
        word_probs = word_probs - joint_normalize_factor

        #
        # word_probs = np.where(word_probs >= nth_min_log_prob, word_probs, -float("inf"))
        word_preds = np.where(word_probs != -float("inf"), word_preds, "")
        # word_preds = np.where(word_probs >= nth_min_log_prob, word_preds, "")

        return word_preds.tolist(), word_probs.tolist(), key_probs

    def get_char_probs(self, context, prefix, keys_li):
        key_results = dict(self.char_predictor.get_characters(context+prefix))
        key_results[kconfig.space_char] = key_results[" "]
        del key_results[" "]

        key_probs = np.array([max(key_results[key], np.log(1/50)) if key in key_results else -float("inf") for key in keys_li])
        # key_probs = np.array([key_results[key] if key in key_results else -float("inf") for key in keys_li])

        return key_probs


def main():
    cwd = os.getcwd()
    word_lm_path = os.path.join(os.path.join(cwd, '../resources'),
                                'lm_word_dec19.kenlm')
    char_lm_path = os.path.join(os.path.join(cwd, '../resources'), 'lm_char_dec19.kenlm')
    vocab_path = os.path.join(os.path.join(cwd, '../resources'), 'vocab_lower_100k.txt')
    char_path = os.path.join(os.path.join(cwd, '../resources'), 'char_set.txt')

    LM = LanguageModel(word_lm_path, char_lm_path, vocab_path, char_path)
    print(LM.get_words("united states of ", "amer", list("abcdefghijklmnopqrstuvwxyz'_")))



if __name__ == "__main__":
    main()
