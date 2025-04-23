# -*- coding: utf-8 -*-
"""Example of using SGAE for outlier detection
"""
# Repository: https://github.com/urbanmobility/SGM
# License: MIT licence

import os
import time

import numpy as np
import torch.nn.init
from sklearn.preprocessing import scale, StandardScaler
from torch import optim, nn
from torch.utils.data import DataLoader

from algorithms.score_guided_autoencoder import ScoreGuidedAutoencoder

def calculate_norm(data, x_dec):
    ''' Calculate l2 norm
    '''
    delta = (data - x_dec).detach().cpu().numpy()
    norm = np.linalg.norm(delta, ord=2, axis=1)
    return norm

def recog_anomal(data, x_dec, thresh):
    ''' Recognize anomaly
    '''
    norm = calculate_norm(data, x_dec)
    anomal_flag = norm.copy()
    anomal_flag[norm < thresh] = 0
    anomal_flag[norm >= thresh] = 1
    return anomal_flag

def data_batch(x_train, batch_size):
    ''' Generate data batch, return tensor.
    '''
    n = len(x_train)
    while(1):
        idx = np.random.choice(n, batch_size, replace=False)
        data = x_train[idx]
        data = torch.FloatTensor(data)
        yield data

class SGAE:
    def __init__(self, epochs = 100, lr=1e-4, early_stop = True, batch_size=1024, device='cuda', seed = 42,
                 lambda_outliers = 20, lambda_DE=0.01, a=6, epsilon=90, hidden_dim='auto',inject_noise = False,cont_rate=0.01, verbose=False):
        self.epochs = epochs
        self.lr = lr
        self.early_stop = early_stop
        self.batch_size = batch_size
        self.device = device
        self.batch_size = batch_size
        self.seed = seed
        self.lambda_outliers = lambda_outliers
        self.lambda_DE = lambda_DE
        self.a = a
        self.epsilon = epsilon
        self.hidden_dim = hidden_dim
        self.inject_noise = inject_noise
        self.cont_rate = cont_rate
        self.verbose = verbose
        return

    def fit(self, X, Y=None):
        cpu_stage_1_start_time = time.perf_counter_ns()

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        if self.inject_noise:
            idx = np.arange(len(X_scaled))
            dim = X_scaled.shape[1]
            all_num = len(X_scaled)
            noise_num = int(all_num * self.cont_rate / (1 - self.cont_rate))
            noise = np.empty((noise_num, dim))
            swap_rate = 0.05
            swap_feature_num = int(dim * swap_rate)
            if swap_feature_num < 1:
                swap_feature_num = 1
            for i in np.arange(noise_num):
                swap_idx = np.random.choice(idx, 2, replace=False)
                swap_feature = np.random.choice(dim, swap_feature_num, replace=False)
                noise[i] = X_scaled[swap_idx[0]].copy()
                noise[i, swap_feature] = X_scaled[swap_idx[1], swap_feature]

            X_scaled = np.append(X_scaled, noise, axis=0)

        cpu_stage_1_stop_time = time.perf_counter_ns()

        self.model = ScoreGuidedAutoencoder(X_scaled.shape[1], self.hidden_dim, seed=self.seed).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        cpu_epoch_time = 0.0
        for epoch in range(self.epochs):
            self.model.train()
            # Calculate norm threshold in batches
            norms = []
            self.model.eval()  # Switch to evaluation mode for calculating norms
            with torch.no_grad():
                data_loader = DataLoader(X_scaled, batch_size=self.batch_size, drop_last=False, pin_memory=True,
                                             shuffle=False)
                for batch in data_loader:
                    batch = batch.float().to(self.device)
                    _, dec_train, _ = self.model(batch)
                    cpu_epoch_stage_1_start_time = time.perf_counter_ns()
                    batch_norm = calculate_norm(batch, dec_train)
                    norms.extend(batch_norm)
                    cpu_epoch_stage_1_stop_time = time.perf_counter_ns()
                    cpu_epoch_time += cpu_epoch_stage_1_stop_time - cpu_epoch_stage_1_start_time

            cpu_epoch_stage_2_start_time = time.perf_counter_ns()
            self.norm_thresh = np.percentile(norms, self.epsilon)

            loss = 0
            recon_error = 0
            dist_error = 0
            cpu_epoch_stage_2_stop_time = time.perf_counter_ns()
            cpu_epoch_time += cpu_epoch_stage_2_stop_time - cpu_epoch_stage_2_start_time

            data_loader = DataLoader(X_scaled, batch_size=self.batch_size, drop_last=False, pin_memory=True, shuffle=True)
            for batch in data_loader:
                batch = batch.float().to(self.device)
                scores, x_dec, _ = self.model(batch)

                cpu_epoch_stage_3_start_time = time.perf_counter_ns()
                anomal_flag = recog_anomal(batch, x_dec, self.norm_thresh)
                anomal_flag = torch.tensor(anomal_flag).to(self.device)
                cpu_epoch_stage_3_stop_time = time.perf_counter_ns()
                cpu_epoch_time += cpu_epoch_stage_3_stop_time - cpu_epoch_stage_3_start_time

                loss_batch, recon_error_batch, dist_error_batch = self.model.loss_function(
                    batch, x_dec, scores, anomal_flag, self.lambda_DE, self.a, self.lambda_outliers, self.device)

                cpu_epoch_stage_4_start_time = time.perf_counter_ns()
                loss += loss_batch.item()
                recon_error += recon_error_batch.item()
                dist_error += dist_error_batch.item()
                cpu_epoch_stage_4_stop_time = time.perf_counter_ns()
                cpu_epoch_time += cpu_epoch_stage_4_stop_time - cpu_epoch_stage_4_start_time

                self.optimizer.zero_grad()
                loss_batch.backward()
                nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
                self.optimizer.step()

        cpu_stage_2_start_time = time.perf_counter_ns()
        cpu_stages_elapsed_time = cpu_stage_1_stop_time - cpu_stage_1_start_time + cpu_epoch_time
        gpu_stages_elapsed_time = cpu_stage_2_start_time - cpu_stage_1_stop_time - cpu_epoch_time
        self.cpu_stages_fit_time = cpu_stages_elapsed_time
        self.gpu_stages_fit_time = gpu_stages_elapsed_time


    def decision_function(self, X):
        score_array = []
        self.model.eval()
        X_scaled = self.scaler.transform(X)

        with torch.no_grad():
            data_loader = DataLoader(X_scaled, batch_size=self.batch_size, drop_last=False, pin_memory=True, shuffle=False)
            for batch in data_loader:
                batch = batch.float().to(self.device)
                scores, _, _ = self.model(batch)
                scores = scores.detach().cpu().numpy()
                score_array.extend(scores)

        return np.array(score_array).ravel()

    def algorithm_name(self):
        return "SGAE"
