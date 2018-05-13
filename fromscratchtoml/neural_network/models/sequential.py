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
            y_pred, y_pred_deriv = self.predict(x, return_deriv=True)
            delta_der_cost_bias, delta_der_cost_weight = self.forward_propogation(x, y)
            if delta_der_cost_bias is None:
                der_cost_bias, der_cost_weight = delta_der_cost_bias, delta_der_cost_weight
            else:
                der_cost_bias += delta_der_cost_bias
                der_cost_weight += delta_der_cost_weight

        # updates weights in each layer
        for layer in self.layers:
            layer.optimize(der_cost_bias, der_cost_weight)

        return
        y_pred, y_pred_deriv = self.predict(X, return_deriv=True)
        loss, loss_grad = self.loss(y_pred, y, return_deriv=True)
        delta = loss_grad * y_pred_deriv
        self.back_propogation(X, y, delta)
        return loss

    def forward_propogation(self, X):
        for layer in self.layers:
            act, act_deriv = layer.forward(X)
            output = act
            output_deriv = act_deriv
        return output, output_deriv

    def back_propogation(self, X, y, delta):
        for i in reversed(range(len(self.layers) - 1)):
            delta = self.layer[i + 1].back_propogate(delta)

    def predict(self, X, return_deriv=False):
        z = X
        for layer in self.layers:
            z, z_deriv = layer.forward(z)

        if return_deriv:
            return z, z_deriv

        return z

    def evaluate(self, X, y, batch_size):
        '''compute loss batch by batch.'''
        pass

    def add(self, layer):
        if hasattr(layer, 'input_dim') and layer.input_dim is None:
            layer.initialize_params(input_dim=self.layers[-1].units, optimizer=self.optimizer)
        self.layers.append(layer)
