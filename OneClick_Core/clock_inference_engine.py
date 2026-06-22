from __future__ import division
import math
import numpy as np
from OneClick_Core.clock_util import ClockUtil
from OneClick_Core import config


class UserDelayModel:
    """Gaussian click-delay model: N(mu, sigma2). Updated on word commit, rolled back on Undo."""

    def __init__(self):
        self.mu = config.mu0
        self.sigma2 = config.sigma0_sq
        self.n_samples = 0
        self.last_update = None   # snapshot for rollback

    def update(self, dt):
        self.last_update = (self.mu, self.sigma2, self.n_samples)
        mu_old = self.mu
        self.mu = config.lambda_decay * self.mu + (1 - config.lambda_decay) * dt
        diff = dt - mu_old
        self.sigma2 = config.lambda_decay * self.sigma2 + (1 - config.lambda_decay) * diff * diff
        self.sigma2 = max(self.sigma2, config.sigma2_min)
        self.n_samples += 1

    def rollback(self):
        if self.last_update is not None:
            self.mu, self.sigma2, self.n_samples = self.last_update
            self.last_update = None

    def get_offset(self):
        return self.mu if self.n_samples >= config.bootstrap_n else 0.0

    def log_likelihood(self, yin):
        diff = yin - self.mu
        return -0.5 * math.log(2 * math.pi * self.sigma2) - diff * diff / (2 * self.sigma2)


class ClockInference:

    def __init__(self, parent, bc):
        self.parent = parent
        self.bc = bc
        self.clock_util = ClockUtil(parent, bc, self)
        self.clocks_li = list(range(len(parent.clock_centers)))  # 0..26

        self.cscores = [0.0] * len(parent.clock_centers)
        self.sorted_inds = list(self.clocks_li)
        self.time_rotate = parent.time_rotate

        self.observations = []   # list of 27-length float arrays, one per Space press
        self.delay_model = UserDelayModel()

    def add_click(self, time_diff_in):
        """
        Record one Space press: append a row of 27 Gaussian log-likelihoods (one per
        letter clock) to the observation matrix. 
        """
        row = []
        for clock in self.clocks_li:
            time_in = (self.clock_util.cur_hours[clock] * self.time_rotate
                       / self.clock_util.num_divs_time
                       + time_diff_in
                       - self.time_rotate * config.frac_period)
            ll = self.delay_model.log_likelihood(time_in)
            row.append(ll)
        self.observations.append(row)

    def format_observations(self, key_chars):
        """Convert observations to the distribs format expected by the word API."""
        result = []
        for row in self.observations:
            distrib = [{"text": key_chars[i], "logProb": float(row[i])} for i in range(len(key_chars))]
            result.append({"distrib": distrib})
        return result

    def reset_observations(self):
        self.observations = []
        self.cscores = [0.0] * len(self.clocks_li)
        self.sorted_inds = list(self.clocks_li)

    def update_sorted_inds(self):
        self.sorted_inds = sorted(self.clocks_li, key=lambda i: -self.cscores[i])
