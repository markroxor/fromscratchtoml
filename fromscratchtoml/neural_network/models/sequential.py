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
                loss = self.__update_batch(batch_X, batch_y)
            print(loss, "loss")

    def __update_batch(self, X, Y):
        der_error_bias = None
        der_error_weight = None

        for x, y in zip(X, Y):
            delta_der_error_bias, delta_der_error_weight, loss = self.back_propogation(x, y)
            if der_error_bias is None:
                der_error_bias, der_error_weight = delta_der_error_bias, delta_der_error_weight
            else:
                der_error_bias += delta_der_error_bias
                der_error_weight += delta_der_error_weight

        # updates weights in each layer
        # for layer, db, dw in zip(self.layers, der_error_bias, der_error_weight):
        #     layer.optimize(self.optimizer, db, dw)

        return loss

    def back_propogation(self, X, y):
        y_pred, y_pred_deriv = self.predict(X, return_deriv=True)

        loss, loss_grad = self.loss(y_pred, y, return_deriv=True)

        delta = loss_grad

        der_error_biases = []
        der_error_weights = []

        for layer in reversed(self.layers):
            # updates delta
            delta, der_error_bias, der_error_weight = layer.back_propogate(delta)

            if hasattr(layer, 'weights'):
                layer.optimize(self.optimizer, der_error_bias, der_error_weight)

            der_error_biases.append(der_error_bias)
            der_error_weights.append(der_error_weight)

        return np.array(der_error_biases[::-1]), np.array(der_error_weights[::-1]), loss

    def predict(self, X, return_deriv=False):
        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=1)
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
