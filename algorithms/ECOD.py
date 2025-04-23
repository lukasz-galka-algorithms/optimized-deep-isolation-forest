# -*- coding: utf-8 -*-
"""Example of using ECOD for outlier detection
"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause

import time

import numpy as np
from numpy import percentile

from .stat_models import column_ecdf
from sklearn.utils import check_array
from scipy.stats import skew as skew_sp

def skew(X, axis=0):
    return np.nan_to_num(skew_sp(X, axis=axis))


def _parallel_ecdf(n_dims, X):
    U_l_mat = np.zeros([X.shape[0], n_dims])
    U_r_mat = np.zeros([X.shape[0], n_dims])

    for i in range(n_dims):
        U_l_mat[:, i: i + 1] = column_ecdf(X[:, i: i + 1])
        U_r_mat[:, i: i + 1] = column_ecdf(X[:, i: i + 1] * -1)
    return U_l_mat, U_r_mat

class ECOD:
    def __init__(self, contamination=0.1, n_jobs=1):
        if (isinstance(contamination, (float, int))):

            if not (0. < contamination <= 0.5):
                raise ValueError("contamination must be in (0, 0.5], "
                                 "got: %f" % contamination)

        self.contamination = contamination

        self.n_jobs = n_jobs
        self._classes = 2

    def fit(self, X, Y=None):
        cpu_stage_1_start_time = time.perf_counter_ns()

        self.decision_scores_ = self.decision_function(X)
        self.X_train = X
        self._process_decision_scores()

        cpu_stage_1_stop_time = time.perf_counter_ns()
        self.cpu_stages_fit_time = cpu_stage_1_stop_time - cpu_stage_1_start_time
        self.gpu_stages_fit_time = 0.0

    def decision_function(self, X):
        if hasattr(self, 'X_train'):
            original_size = X.shape[0]
            X = np.concatenate((self.X_train, X), axis=0)
        self.U_l = -1 * np.log(column_ecdf(X))
        self.U_r = -1 * np.log(column_ecdf(-X))

        skewness = np.sign(skew(X, axis=0))
        self.U_skew = self.U_l * -1 * np.sign(
            skewness - 1) + self.U_r * np.sign(skewness + 1)

        self.O = np.maximum(self.U_l, self.U_r)
        self.O = np.maximum(self.U_skew, self.O)

        if hasattr(self, 'X_train'):
            decision_scores_ = self.O.sum(axis=1)[-original_size:]
        else:
            decision_scores_ = self.O.sum(axis=1)
        return decision_scores_.ravel()

    def _process_decision_scores(self):
        if isinstance(self.contamination, (float, int)):
            self.threshold_ = percentile(self.decision_scores_,
                                         100 * (1 - self.contamination))
            self.labels_ = (self.decision_scores_ > self.threshold_).astype(
                'int').ravel()

        else:
            self.labels_ = self.contamination.eval(self.decision_scores_)
            self.threshold_ = self.contamination.thresh_
            if not self.threshold_:
                self.threshold_ = np.sum(self.labels_) / len(self.labels_)

    def algorithm_name(self):
        return "ECOD"