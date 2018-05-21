#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 Mohit Rathore <mrmohitrathoremr@gmail.com>
# Licensed under the GNU General Public License v3.0 - https://www.gnu.org/licenses/gpl-3.0.en.html

import unittest

import numpy as np
from fromscratchtoml.neural_network.models import Sequential
from fromscratchtoml.neural_network.layers import Dense, Activation
from fromscratchtoml.neural_network.optimizers import StochasticGradientDescent

from sklearn.model_selection import train_test_split
from fromscratchtoml.toolbox.preprocess import to_onehot
from fromscratchtoml.toolbox.random import Distribution

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TestNN(unittest.TestCase):
    def setUp(self):
        X11 = Distribution.radial_binary(pts=300,
                       mean=[0, 0],
                       st=1,
                       ed=2, seed=20)
        X22 = Distribution.radial_binary(pts=300,
                       mean=[0, 0],
                       st=4,
                       ed=5, seed=20)

        Y11 = np.ones(X11.shape[0])
        Y22 = np.zeros(X11.shape[0])

        X = np.vstack((X11, X22))
        y = np.hstack((Y11, Y22))

        y = to_onehot(y)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=50, random_state=42)

    def test_dense_acts_sgd(self):
        model = Sequential()

        model.add(Dense(10, input_dim=2, seed=1))
        model.add(Activation('sigmoid'))

        model.add(Dense(2, seed=7))
        model.add(Activation('tanh'))

        model.add(Dense(2, seed=5))
        model.add(Activation('softmax'))


        model.add(Dense(2, seed=2))
        model.add(Activation('relu'))

        model.add(Dense(2, seed=3))
        model.add(Activation('leaky_relu'))

        model.add(Dense(2, seed=4))
        model.add(Activation('linear'))


        model.add(Dense(2, seed=6))

        sgd = StochasticGradientDescent(learning_rate=0.05)
        model.compile(optimizer=sgd, loss="mean_squared_error")

        model.fit(self.X_train, self.y_train, epochs=14)

        expected_biases = np.array([[0.08650937, 1.00013189]], dtype=np.float128)
        self.assertTrue(np.allclose(expected_biases, model.layers[-1].biases))

        expected_weights = np.array([[-0.49908263, -0.17316507], [-0.42623203,  0.48448988]], dtype=np.float128)
        self.assertTrue(np.allclose(expected_weights, model.layers[-1].weights))

        predictions = model.predict(self.X_test, one_hot=1)

        self.assertTrue(np.allclose(predictions, self.y_test))
