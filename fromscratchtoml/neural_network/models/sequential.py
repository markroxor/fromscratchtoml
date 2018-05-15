#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 Mohit Rathore <mrmohitrathoremr@gmail.com>
# Licensed under the GNU General Public License v3.0 - https://www.gnu.org/licenses/gpl-3.0.en.html

import numpy as np
from .. import losses

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Sequential(object):
    def __init__(self):
        self.layers = []

    def compile(self, optimizer, loss):
        self.optimizer = optimizer
        self.loss = getattr(losses, loss)

    def fit(self, X, y, epochs, batch_size):
        for _ in range(epochs):
            for current_batch in range(0, X.shape[0], batch_size):
                batch_X = X[current_batch: current_batch + batch_size]
                batch_y = y[current_batch: current_batch + batch_size]
                self.__update_batch(batch_X, batch_y)

    def __update_batch(self, X, Y):
        der_cost_bias = None
        der_cost_weight = None

        for x, y in zip(X, Y):
            delta_der_cost_bias, delta_der_cost_weight = self.back_propogation(x, y)

            if der_cost_bias is None:
                der_cost_bias, der_cost_weight = delta_der_cost_bias, delta_der_cost_weight
            else:
                der_cost_bias += delta_der_cost_bias
                der_cost_weight += delta_der_cost_weight

        # updates weights in each layer
        for layer in self.layers:
            layer.optimize(self.optimizer, der_cost_bias, der_cost_weight)

        return

    def back_propogation(self, X, y):
        y_pred, y_pred_deriv = self.predict(X, return_deriv=True)
        loss, loss_grad = self.loss(y_pred, y, return_deriv=True)
        delta = loss_grad * y_pred_deriv

        delta_nabla_b = []
        delta_nabla_w = []

        delta_nabla_b.append(np.array(delta))
        delta_nabla_w.append(np.dot(delta, self.layers[-3].output.T))

        for i in reversed(range(len(self.layers) - 1)):
            # updates delta
            delta = self.layers[i + 1].back_propogate(delta)

            if hasattr(self.layers[i + 1], 'activation'):
                delta_nabla_b.append(np.array(delta))
                delta_nabla_w.append(np.dot(delta, self.layers[i + 1].input.transpose()))
            else:
                delta_nabla_b.append([0])
                delta_nabla_w.append([0])

        return np.array(delta_nabla_b[::-1]), np.array(delta_nabla_w[::-1])

    def predict(self, X, return_deriv=False):
        z = X
        for layer in self.layers:
            z, z_deriv = layer.forward(z, return_deriv=True)

        if return_deriv:
            return z, z_deriv

        return z

    def evaluate(self, X, y, batch_size):
        '''compute loss batch by batch.'''
        pass

    def add(self, layer):
        if hasattr(layer, 'input_dim') and layer.input_dim is None:
            layer.initialize_params(input_dim=self.layers[-1].units)
        self.layers.append(layer)
