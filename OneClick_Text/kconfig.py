# Keyboard layout matching oneclick/kconfig.js

key_chars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
             'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', "'"]

eow_cell = "EOW"

alpha_target_layout = [
    ['a', 'b', 'c', 'd', 'e', 'f', 'g'],
    ['h', 'i', 'j', 'k', 'l', 'm', 'n'],
    ['o', 'p', 'q', 'r', 's', 't', 'u'],
    ['v', 'w', 'x', 'y', 'z', "'", eow_cell],
]

n_pred = 3    # prefix word predictions per letter cell
n_best = 3    # BEST (error-corrected) decodings in EOW cell

# ----------------------------------------------------------------------------
# Word-clock logical index space (mirrors oneclick/clock_inference_engine_word.js
# lines 204-212). All word clocks live in ONE positionally-indexed array, exactly
# like Nomon keeps letters+words+specials in one array via init_locs/index_to_wk:
#
#   [0 .. n_inline-1]                  prefix slots: letter_idx * n_pred + slot
#                                      (a completion filed under its NEXT letter)
#   [best_base_index .. +n_best-1]     EOW-cell BEST (error-corrected) decodings
#   argmax_word_index                  literal per-click argmax decode (no API correction)
#   undo_word_index                    Undo
# ----------------------------------------------------------------------------
n_letters = len(key_chars)                       # 27
n_inline = n_letters * n_pred                    # 81 prefix slots
best_base_index = n_inline                        # 81
argmax_word_index = n_inline + n_best             # 84
undo_word_index = n_inline + n_best + 1           # 85
n_word_clocks_total = undo_word_index + 1         # 86
