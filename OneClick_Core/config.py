import numpy as np

# Clock animation parameters (matching oneclick/config.js)
period_li = np.arange(21)
period_li = 6 * np.e ** (-period_li / 10)

default_rotate_ind = 5
ideal_wait_s = 0.04
frac_period = 4.0 / 8.0          # 0.5 — starting phase offset
theta0 = frac_period * 2.0 * np.pi   # = pi

# UserDelayModel priors (matching oneclick/config.js)
mu0 = 0.05              # initial mean click delay (s)
sigma0_sq = 0.0144      # 0.12^2 — initial variance
lambda_decay = 0.9      # EMA decay factor
sigma2_min = 0.0025     # minimum variance floor
bootstrap_n = 3         # samples before using learned offset
use_click_offset = False   # whether to apply mu offset on word selection

# API config
num_prefix_fetch = 25   # pool size for prefix word predictions
num_best_fetch = 10     # pool size for BEST (error-corrected) decodings
