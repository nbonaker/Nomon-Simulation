#!/usr/bin/python

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

### Configuration settings for the SimulatedUser module ###

### SimulatedUser setup ###
# characters in the keys
space_char = '_'
mybad_char = '@'
# yourbad_char = 'Yours'
yourbad_char = 'Undo+'
break_chars = ['.', ',', '?', '!']
back_char = '#'
clear_char = '$'

# word length to display in completions
max_chars_display = 11
num_words_total = 17
## alphabetic
# always put alpha-numeric keys first (self.N_alpha_keys)

main_chars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                   'u', 'v', 'w', 'x', 'y', 'z', '\'']
key_chars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                   'u', 'v', 'w', 'x', 'y', 'z', '\'', '.', ',', '?', '!', '#', '$', '@', '_',]

alpha_target_layout = [['a', 'b', 'c', 'd', 'e'],
                 ['f', 'g', 'h', 'i', 'j'],
                 ['k', 'l', 'm', 'n', 'o'],
                 ['p', 'q', 'r', 's', 't'],
                 ['u', 'v', 'w', 'x', 'y'],
                 ['z', "\'", 'BREAKUNIT', 'BACKUNIT', 'UNDOUNIT']]

## qwerty
qwerty_target_layout = [['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'],
                    ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', '\''],
                    ['z', 'x', 'c', 'v', 'b', 'n', 'm', ",", ".", "?"],
                    [mybad_char, back_char, clear_char, space_char, "!"]]

### Events ###
# event selection
joy_evt = "<<JoyFoo>>"
key_evt = "<space>"
# event to use as switch
target_evt = key_evt

### Speech ###
# talk_winner_on = False

### Word display parameters ###
## sizes
# base window size (for relative size calculations)
base_window_width = 1200
base_window_height = 700
# clock radius
base_clock_rad = 10  # 10

pre_clock_rad = 200
clock_rad = 10  # 10
# word width
base_word_w = 160
word_w = 160
# words per key
N_pred = 3


### Language model ###
# probability threshold for inclusion of word in the display
prob_thres = 0.008
# undo prior prob
undo_prob = 1.0 / 40
# back prior prob
back_prob = 1.0 / 40
# remaining, non-special probability
rem_prob = 1.0 - undo_prob - back_prob