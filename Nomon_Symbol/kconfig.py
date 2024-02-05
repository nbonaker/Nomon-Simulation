#!/usr/bin/python

import emoji

### Configuration settings for the SimulatedUser module ###
### Keyboard Layout setup ###
# characters in the keys
mybad_char = '@'
space_char = '_'
clear_char = '$'
break_char = "."

emoji_file = open("../Nomon_Symbol/resources/emojis.txt", "r")
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

