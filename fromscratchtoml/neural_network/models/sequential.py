#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 Mohit Rathore <mrmohitrathoremr@gmail.com>
# Licensed under the GNU General Public License v3.0 - https://www.gnu.org/licenses/gpl-3.0.en.html

from __future__ import print_function
import numpy as np

from fromscratchtoml.toolbox import progress, binary_visualize
from .. import losses

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Sequential(object):
    def __init__(self, verbose=False, vis_each_epoch=False, seed=None):
        self.layers = []
        self.verbose = verbose
        self.vis_each_epoch = vis_each_epoch

    def compile(self, optimizer, loss):
        self.optimizer = optimizer
        self.loss = getattr(losses, loss)

    def accuracy(self, X, y):
        y_pred = self.predict(X, one_hot=True)
        diff_arr = y - y_pred
        total_samples = y.shape[0]

        errors = np.count_nonzero(diff_arr) / 2
        return (100 - (errors / (total_samples * 0.01)))

    def fit(self, X, y, epochs, batch_size=None):
        if batch_size is None:
            batch_size = X.shape[0]

        for epoch in progress(range(epochs)):
            for current_batch in range(0, X.shape[0], batch_size):
                batch_X = X[current_batch: current_batch + batch_size]
                batch_y = y[current_batch: current_batch + batch_size]
                self.__update_batch(batch_X, batch_y)

            if self.verbose or epoch == epochs - 1:
                y_pred = self.predict(X, one_hot=True)
                loss = self.loss(y_pred, y)
                acc = self.accuracy(X, y)
                print("\nepoch: {}/{} ".format(epoch + 1, epochs), end="")
                print(" acc: {:0.2f} ".format(acc), end="")
                print(" loss: {:0.3f} ".format(loss))
                if self.vis_each_epoch:
                    binary_visualize(X, clf=self, draw_contour=True)

    def __update_batch(self, X, Y):
        # der_error_bias = None
        # der_error_weight = None

        for x, y in zip(X, Y):
            delta_der_error_bias, delta_der_error_weight = self.back_propogation(x, y)
            # if der_error_bias is None:
            #     der_error_bias, der_error_weight = delta_der_error_bias, delta_der_error_weight
            # else:
            #     der_error_bias += delta_der_error_bias
            #     der_error_weight += delta_der_error_weight

        # updates weights in each layer
        # for layer, db, dw in zip(self.layers, der_error_bias, der_error_weight):
        #     layer.optimize(self.optimizer, db, dw)

    def back_propogation(self, x, y):
        y_pred, y_pred_deriv = self.forwardpass(x, return_deriv=True)

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

        return np.array(der_error_biases[::-1]), np.array(der_error_weights[::-1])

    def forwardpass(self, x, return_deriv=False):
        z = x

        for layer in self.layers:
            z, z_deriv = layer.forward(z, return_deriv=True)

        if return_deriv:
            return z, z_deriv

        return z

    def predict(self, X, one_hot=False):
        Z = []
        for x in X:
            z = self.forwardpass(x)

            t = np.zeros_like(z)
            if one_hot:
                # returns one hot
                t[np.argmax(z)] = 1
                Z.append(t.flatten())
            else:
                # returns class
                Z.append(np.argmax(z))

        return np.array(Z)

    def add(self, layer):
        self.layers.append(layer)
