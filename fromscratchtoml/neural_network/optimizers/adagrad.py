#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 Mohit Rathore <mrmohitrathoremr@gmail.com>
# Licensed under the GNU General Public License v3.0 - https://www.gnu.org/licenses/gpl-3.0.en.html

from fromscratchtoml import np

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# optimizes/updates the weights
class Adagrad(object):
    """
    A sequence of multiple layers.

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

    def __init__(self, learning_rate=0.01):
        """
        Initialising the optimizer parameters.

        Parameters
        ----------
        learning_rate : float
            the rate of change of weights. The higher the learning rate - more is the change in the parameters.
        """
        self.learning_rate = learning_rate
        self.accumulated_sq_gradient = 0

    def update_weights(self, w, dEdW):
        """
        Updates the parameters.

        Parameters
        ----------
        w : numpy.ndarray
            The weight to be updated.
        grad_wrt_w : numpy.ndarray
            The derivative of error with respect to weight.
        """

        self.accumulated_sq_gradient += np.square(dEdW)
        return w - self.learning_rate * dEdW / (np.sqrt(self.accumulated_sq_gradient) + 1e-8)
