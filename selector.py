# -*- coding: utf-8 -*-

import ctypes
import multiprocessing as mp
import numpy as np


class Selector():
    def __init__(self, main_num, oppo_num, min_weight_factor=0.1):
        self.main_num = main_num
        self.oppo_num = oppo_num
        self.comb_num = main_num * oppo_num
        self.comb_ids = list(range(self.comb_num))
        self.min_weight_factor = min_weight_factor
        # Shared arrays
        self.lock = mp.Lock()
        # opponent's winrate
        self.oppo_winrate = mp.RawArray(ctypes.c_double, np.full([self.comb_num], 0.5))
        self.regret_sum = mp.RawArray(ctypes.c_double, np.zeros([self.comb_num]))
        # weights for selecting main-oppo pair
        self.weights = mp.RawArray(ctypes.c_double, np.full([self.comb_num], 1.0 / self.comb_num))

    def select(self):
        with self.lock:
            np_regret_sum = np.asarray(self.regret_sum)
            norm_sum = np.sum(np_regret_sum)
            if norm_sum > 0:
                np_weights = np.maximum(np_regret_sum / norm_sum, 0.0) * (1 - self.min_weight_factor) \
                        + self.min_weight_factor / self.comb_num
            else:
                np_weights = np.full([self.comb_num], 1.0 / self.comb_num)
                for i in range(self.comb_num):
                    self.weights[i] = np_weights[i]
        idx = random.choices(self.ids, weights=np_weights)[0]
        main_idx = idx // self.oppo_num
        oppo_idx = (idx - main_idx * self.oppo_num) % self.main_num
        return main_idx, oppo_idx

    def update(self, main_idx, oppo_idx, result):
        idx = main_idx * self.oppo_num + oppo_idx
        with self.lock:
            # Exponential smoothing
            self.oppo_winrate[idx] = self.oppo_winrate[idx] * 0.9 + (1 - result) * 0.1
            # The more one lose, the more it needs to be trained
            np_oppo_winrate = np.asarray(self.oppo_winrate)
            np_weights = np.asarray(self.weights)
            # Expected utility (lose)
            expected_utility = np.dot(np_oppo_winrate, np_weights)
            # Regret-Matching+ train
            delta_utility = np_oppo_winrate - expected_utility
            for i in range(self.comb_num):
                self.regret_sum[i] = max(self.regret_sum[i] + delta_utility[i], 0)
