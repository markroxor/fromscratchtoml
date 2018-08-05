#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 Mohit Rathore <mrmohitrathoremr@gmail.com>
# Licensed under the GNU General Public License v3.0 - https://www.gnu.org/licenses/gpl-3.0.en.html

from fromscratchtoml import np
import numpy

from fromscratchtoml.neural_network.layers import Layer
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Dropout(Layer):
    """
    The dropout layer actively diminishes the activations of previous layer by a certain
    probability.

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

    def __init__(self, rate=0.5, input_dim=None, seed=None):
        """
        Initialising the layer parameters.

        Parameters
        ----------
        rate : float
            The probability of dropping out activations.
        input_dim : int, int
            If using dropout as the first layer, this parameter is required to dropout inputs.
        seed : int, optional
            The seed value for mantaining reproduciblity of random weights.

        """

        self.rate = rate
        self.mask = None
        self.input_dim = input_dim
        self.seed = seed

    def forward(self, X, train=False):
        """
        Forward pass the output of the previous layer as it is at test time and by masking certain activations
        at train time.

        Parameters
        ----------
        X : numpy.ndarray
            The ouput of the previous layer

        Returns
        -------
        numpy.array : The masked input.
        """
        if len(X.shape) == 1:
            assert(1==2)
            X = np.expand_dims(X, axis=1)

        self.output = X

        if train:
            # NOTE: this will create disperencies when rate is 0. Results when rate==0 and when not using
            # dropout layer at all wont be the same.
            if self.seed is not None:
                np.random.seed(self.seed)

            if self.input_dim:
                self.mask = np.random.rand(self.input_dim) >= self.rate
            else:
                self.mask = np.random.rand(*X.shape) >= self.rate

            self.output *= self.mask / (1. - self.rate)

        return self.output

    def back_propogate(self, dEdO):
        """
        Backpropogate the error, this function adds the share of this layer to the accumulated gradient.

        Parameters
        ----------
        dEdO : numpy.ndarray
            The accumulated gradient used for calculating error gradient with respect to parameters.

        Returns
        -------
        numpy.array : The accumulated gradient.
        """
        return dEdO * self.mask
