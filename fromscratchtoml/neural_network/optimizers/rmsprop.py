#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author - Mohit Rathore <mrmohitrathoremr@gmail.com>
# Licensed under The MIT License - https://opensource.org/licenses/MIT

from fromscratchtoml import np

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# optimizes/updates the weights
class RMSprop(object):
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

    def __init__(self, learning_rate=0.001, decay=0.9):
        """
        Initialising the optimizer parameters.

        Parameters
        ----------
        learning_rate : float
            the rate of change of weights. The higher the learning rate - more is the change in the parameters.
        decay : float
            the leakage of previous accumulated gradient square into current.
        """
        self.learning_rate = learning_rate
        self.decay = decay
        self.accumulated_sq_gradient = 0

    def update_weights(self, w, dEdW):
        """
        Updates the parameters.

        Parameters
        ----------
        w : numpy.ndarray
            The weight to be updated.
        dEdW : numpy.ndarray
            The derivative of error with respect to weight.
        """

        self.accumulated_sq_gradient = self.decay * self.accumulated_sq_gradient + (1 - self.decay) * np.square(dEdW)
        return w - self.learning_rate * dEdW / (np.sqrt(self.accumulated_sq_gradient) + 1e-8)
