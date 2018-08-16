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


class l1(object):
    """
    The l1 regularization.

    Examples
    --------
    >>> from fromscratchtoml.neural_network.models import Sequential
    >>> from fromscratchtoml.neural_network.optimizers import StochasticGradientDescent
    >>> from fromscratchtoml.neural_network.layers import Dense, Activation
    >>> X1 = np.array([[0, 0],[0, 1],[1, 0], [1, 1]])
    >>> y1 = np.array([[1,0], [0,1], [0,1], [1,0]])
    >>> model = Sequential()
    >>> model.add(Dense(5, kernel_regularizer=l1(0.01), input_dim=2, seed=1))
    >>> model.add(Activation('sigmoid'))
    >>> model.add(Dense(5, kernel_regularizer=l1(0.01), seed=2))
    >>> model.add(Activation('sigmoid'))
    >>> model.add(Dense(2, kernel_regularizer=l1(0.01), seed=3))
    >>> sgd = StochasticGradientDescent(learning_rate=0.1)
    >>> model.compile(optimizer=sgd, loss="mean_squared_error")
    >>> model.fit(X1, y1, batch_size=4, epochs=100)
    >>> model.predict(X1, one_hot=True)
    """

    def __init__(self, lmda=0.01):
        """
        Initialising the lamda value.

        Parameters
        ----------
        lmda : float
            The regularization coefficient.

        """
        self.lmda = lmda

    def value(self, weights, batch_size):
        return self.lmda * np.absolute(weights) / (1.0 * batch_size)

    def grad(self, weights, batch_size):
        return self.lmda / (1.0 * batch_size)


class l2(object):
    """
    The l2 regularization.

    Examples
    --------
    >>> from fromscratchtoml.neural_network.models import Sequential
    >>> from fromscratchtoml.neural_network.optimizers import StochasticGradientDescent
    >>> from fromscratchtoml.neural_network.layers import Dense, Activation
    >>> X1 = np.array([[0, 0],[0, 1],[1, 0], [1, 1]])
    >>> y1 = np.array([[1,0], [0,1], [0,1], [1,0]])
    >>> model = Sequential()
    >>> model.add(Dense(5, kernel_regularizer=l2(0.01), input_dim=2, seed=1))
    >>> model.add(Activation('sigmoid'))
    >>> model.add(Dense(5, kernel_regularizer=l2(0.01), seed=2))
    >>> model.add(Activation('sigmoid'))
    >>> model.add(Dense(2, kernel_regularizer=l2(0.01), seed=3))
    >>> sgd = StochasticGradientDescent(learning_rate=0.1)
    >>> model.compile(optimizer=sgd, loss="mean_squared_error")
    >>> model.fit(X1, y1, batch_size=4, epochs=100)
    >>> model.predict(X1, one_hot=True)
    """

    def __init__(self, lmda=0.01):
        """
        Initialising the lamda value.

        Parameters
        ----------
        lmda : float
            The regularization coefficient.

        """
        self.lmda = lmda

    def value(self, weights, batch_size):
        return (self.lmda / 2.0) * np.square(weights) / (1.0 * batch_size)

    def grad(self, weights, batch_size):
        return self.lmda * weights / (1.0 * batch_size)


class l1_l2(object):
    """
    The l1_l2 regularization AKA elastic net.

    Examples
    --------
    >>> from fromscratchtoml.neural_network.models import Sequential
    >>> from fromscratchtoml.neural_network.optimizers import StochasticGradientDescent
    >>> from fromscratchtoml.neural_network.layers import Dense, Activation
    >>> X1 = np.array([[0, 0],[0, 1],[1, 0], [1, 1]])
    >>> y1 = np.array([[1,0], [0,1], [0,1], [1,0]])
    >>> model = Sequential()
    >>> model.add(Dense(5, kernel_regularizer=l1_l2(0.01), input_dim=2, seed=1))
    >>> model.add(Activation('sigmoid'))
    >>> model.add(Dense(5, kernel_regularizer=l1_l2(0.01), seed=2))
    >>> model.add(Activation('sigmoid'))
    >>> model.add(Dense(2, kernel_regularizer=l1_l2(0.01), seed=3))
    >>> sgd = StochasticGradientDescent(learning_rate=0.1)
    >>> model.compile(optimizer=sgd, loss="mean_squared_error")
    >>> model.fit(X1, y1, batch_size=4, epochs=100)
    >>> model.predict(X1, one_hot=True)
    """

    def __init__(self, lmda1=0.01, lmda2=0.01):
        """
        Initialising the lamda value.

        Parameters
        ----------
        lmda1 : float
            The l1 regularization coefficient.
        lmda2 : float
            The l2 regularization coefficient.

        """
        self.l1 = lmda1
        self.l2 = lmda2

    def value(self, weights, batch_size):
        return self.l1 * np.absolute(weights) / (1.0 * batch_size) + (self.l2 / 2.0) *\
        np.square(weights) / (1.0 * batch_size)

    def grad(self, weights, batch_size):
        return self.l1 / (1.0 * batch_size) + self.l2 * weights / (1.0 * batch_size)
