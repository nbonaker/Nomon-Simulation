######################################
#    Copyright 2009 Tamara Broderick
#    This file is part of Nomon Keyboard.
#
#    Nomon Keyboard is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Nomon Keyboard is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Nomon Keyboard.  If not, see <http://www.gnu.org/licenses/>.
######################################


import numpy as np
import re
import json


class Phrases:
    def __init__(
        self,
        iv_phrases_file_name,
        oov_phrases_file_name,
        shuffle_seed=None,
        quiet=False,
    ):
        iv_phrases_file = open(iv_phrases_file_name, "r")
        iv_phrases_text = iv_phrases_file.read()
        iv_phrases_file.close()

        oov_phrases_file = open(oov_phrases_file_name, "r")
        oov_phrases_text = oov_phrases_file.read()
        oov_phrases_file.close()

        self.phrases = []

        # parse phrase text from csv file
        iv_phrases = [phrase[phrase.index("\t") + 1:] for phrase in iv_phrases_text.split("\n") if "\t" in phrase]
        oov_phrases = [phrase[phrase.index("\t") + 1:] for phrase in oov_phrases_text.split("\n") if "\t" in phrase]

        iv_oov_split_ratio = 2

        split_ind = 0
        while len(iv_phrases) > 0 and len(oov_phrases) > 0:
            is_oov = split_ind % (iv_oov_split_ratio + 1) == 0

            # alternate draws between iv and oov phrases according to ratio
            # add option to seed the shuffle for across-simulation consistency
            if shuffle_seed is None:
                if is_oov:
                    cur_phrase = oov_phrases.pop(np.random.randint(0, len(oov_phrases)))
                else:
                    cur_phrase = iv_phrases.pop(np.random.randint(0, len(iv_phrases)))
            else:
                np.random.seed(shuffle_seed)
                if is_oov:
                    cur_phrase = oov_phrases.pop(np.random.randint(0, len(oov_phrases)))
                else:
                    cur_phrase = iv_phrases.pop(np.random.randint(0, len(iv_phrases)))

            # remove characters not in nomon's keys
            cur_phrase = re.sub(r"[^a-z \']+", '', cur_phrase.lower())
            cur_phrase = re.sub(r"  ", ' ', cur_phrase.lower())
            self.phrases.append([cur_phrase, "oov" if is_oov else "iv"])

            split_ind += 1

        self.num_phrases = len(self.phrases)
        if not quiet:
            print("loaded "+str(self.num_phrases)+" phrases")

        self.cur_phrase = None

    def sample(self):
        if len(self.phrases) > 0:
            self.cur_phrase = self.phrases.pop()
            return self.cur_phrase
        else:
            return [None, None]


def main():
    phrases = Phrases("resources/lm_likely_phrases.txt", "resources/lm_unlikely_phrases.txt")
    print(json.dumps(phrases.phrases))


if __name__ == "__main__":
    main()
