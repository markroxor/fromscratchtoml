#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 Mohit Rathore <mrmohitrathoremr@gmail.com>
# Licensed under the GNU General Public License v3.0 - https://www.gnu.org/licenses/gpl-3.0.en.html

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Layer(object):
    """
    This parent class of all layers.

    Examples
    --------
    >>> from fromscratchtoml.neural_network.models import Sequential
    >>> from fromscratchtoml.neural_network.optimizers import StochasticGradientDescent
    >>> from fromscratchtoml.neural_network.layers import Dense, Activation
    >>> X1 = np.array([[0, 0],[0, 1],[1, 0], [1, 1]])
    >>> y1 = np.array([[1,0], [0,1], [0,1], [1,0]])
    >>> model = Sequential()
    >>> model.add(Dense(5, input_dim=2, seed=1))
    >>> model.add(Activation('sigmoid'))
    >>> model.add(Dense(5, seed=2))
    >>> model.add(Activation('sigmoid'))
    >>> model.add(Dense(2, seed=3))
    >>> sgd = StochasticGradientDescent(learning_rate=0.1)
    >>> model.compile(optimizer=sgd, loss="mean_squared_error")
    >>> model.fit(X1, y1, batch_size=4, epochs=100)
    >>> model.predict(X1, one_hot=True)
    """
    def optimize(self, optimizer, der_cost_bias, der_cost_weight):
        """
        Optimize the weights corresponding to the optimizer function supplied.

        Parameters
        ----------
        optimizer : fromscratchtoml.neural_network.optimizers
            The optimizing procedure followed for updating the weights.
        der_cost_bias : numpy.ndarray
            The derivative of error with respect to bias.
        der_cost_weights : numpy.ndarray
            The derivative of error with respect to weights.
        """
        if self.trainable:
            self.weights = optimizer.update_weights(self.weights, der_cost_weight)
            self.biases = optimizer.update_weights(self.biases, der_cost_bias)
        return
