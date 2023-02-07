#!/usr/bin/python

######################################
# Copyright 2019 Nicholas Bonaker, Keith Vertanen, Emli-Mari Nel, Tamara Broderick
# This file is part of the Nomon software.
# Nomon is free software: you can redistribute it and/or modify it
# under the terms of the MIT License reproduced below.
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY
# OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
# LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
# EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR
#OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
#
# <https://opensource.org/licenses/mit-license.html>
######################################

import emoji

### Configuration settings for the SimulatedUser module ###
### Keyboard Layout setup ###
# characters in the keys
mybad_char = '@'
space_char = '_'
clear_char = '$'
break_char = "."

emoji_file = open("resources/emojis.txt", "r")
emoji_text = emoji_file.read()
emoji_file.close()
emoji_keys = emoji_text.split("\n")
emoji_keys = [emoji.emojize(key, use_aliases=True) for key in emoji_keys]
emoji_keys += [break_char, space_char, clear_char, mybad_char]

num_rows = 10
num_cols = 7
num_emoji_keys = 61

emoji_target_layout = []
for i in range(num_emoji_keys):
    if i % num_rows == 0:
        emoji_target_layout.append([])
    emoji_target_layout[-1].append(emoji_keys[i])

emoji_target_layout[-1] += [break_char, space_char, clear_char, mybad_char]


# base window size (for relative size calculations)
base_window_width = 1200
base_window_height = 700

# undo prior prob
undo_prob = 1.0 / 40
# remaining, non-special probability
rem_prob = 1.0 - undo_prob
