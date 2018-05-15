#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 Mohit Rathore <mrmohitrathoremr@gmail.com>
# Licensed under the GNU General Public License v3.0 - https://www.gnu.org/licenses/gpl-3.0.en.html

import numpy as np


import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Dense(object):
    def __init__(self, units, input_dim=None, trainable=True, ln=None):
        self.ln = ln

        self.units = units
        self.biases = None
        self.weights = None
        self.trainable = trainable

        if input_dim:
            # x=1 single row
            self.biases = np.random.randn(1, self.units)
            self.weights = np.random.randn(input_dim, self.units)

    def initialize_params(self, input_dim):
        self.biases = np.random.randn(1, self.units)
        self.weights = np.random.randn(input_dim, self.units)

    def forward(self, X, return_deriv=False):
        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=1)

        if self.weights is None:
            self.biases = np.random.randn(1, self.units)
            self.weights = np.random.randn(X.shape[0], self.units)

        self.input = X
        self.output = (np.dot(X.T, self.weights) + self.biases).T

        if return_deriv:
            return self.output, 0

        return self.output

    def back_propogate(self, delta):
        der_error_bias = delta.T
        der_error_weight = np.dot(delta, self.input.T).T
        delta = np.dot(self.weights, delta)
        return delta, der_error_bias, der_error_weight

    def optimize(self, optimizer, der_cost_bias, der_cost_weight):
        if self.trainable:
            self.weights = optimizer.update_weights(self.weights, der_cost_weight)
            self.biases = optimizer.update_weights(self.biases, der_cost_bias)
