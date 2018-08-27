#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author - Mohit Rathore <mrmohitrathoremr@gmail.com>
# Licensed under The MIT License - https://opensource.org/licenses/MIT

import unittest

from fromscratchtoml import np
from fromscratchtoml.neural_network.models import Sequential
from fromscratchtoml.neural_network.layers import Dense, Activation
from fromscratchtoml.neural_network.optimizers import StochasticGradientDescent, Adam, Adagrad, RMSprop
from fromscratchtoml.neural_network.regularizers import l2

from sklearn.model_selection import train_test_split
from fromscratchtoml.toolbox.preprocess import to_onehot
from fromscratchtoml.toolbox.random import Distribution

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TestOptim(unittest.TestCase):
    def setUp(self):
        X11 = Distribution.radial_binary(pts=300,
                       mean=[0, 0],
                       st=1,
                       ed=2, seed=20)
        X22 = Distribution.radial_binary(pts=300,
                       mean=[0, 0],
                       st=4,
                       ed=5, seed=10)

        Y11 = np.ones(X11.shape[0])
        Y22 = np.zeros(X11.shape[0])

        X = np.vstack((X11, X22))
        y = np.hstack((Y11, Y22))

        y = to_onehot(y)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=50, random_state=42)

    def test_nag_momentum(self):
        model = Sequential()
        lmda = 0.00001

        model.add(Dense(10, kernel_regularizer=l2(lmda), input_dim=2, seed=1))
        model.add(Activation('sigmoid'))

        model.add(Dense(2, kernel_regularizer=l2(lmda), seed=3))
        model.add(Activation('tanh'))

        model.add(Dense(2, kernel_regularizer=l2(lmda), seed=6))
        model.add(Activation('softmax'))

        opt = StochasticGradientDescent(learning_rate=0.1, momentum=0.8, nesterov=1)
        model.compile(optimizer=opt, loss="mean_squared_error")

        model.fit(self.X_train, self.y_train, epochs=4, batch_size=4)

        expected_biases = np.array([[-0.95917324, -0.32783731]], dtype=np.float64)
        self.assertTrue(np.allclose(expected_biases, model.layers[-2].biases))

        expected_weights = np.array([[1.01491882, -1.26331766], [2.53084507, -0.92286187]], dtype=np.float64)
        self.assertTrue(np.allclose(expected_weights, model.layers[-2].weights))

    def test_adagrad(self):
        model = Sequential()
        lmda = 0.00001

        model.add(Dense(10, kernel_regularizer=l2(lmda), input_dim=2, seed=1))
        model.add(Activation('sigmoid'))

        model.add(Dense(2, kernel_regularizer=l2(lmda), seed=3))
        model.add(Activation('tanh'))

        model.add(Dense(2, kernel_regularizer=l2(lmda), seed=6))
        model.add(Activation('softmax'))

        opt = Adagrad(learning_rate=0.2)
        model.compile(optimizer=opt, loss="mean_squared_error")

        model.fit(self.X_train, self.y_train, epochs=1, batch_size=4)
        print(model.layers[-2].weights)
        print(model.layers[-2].biases)

        expected_biases = np.array([[-0.95917324, -0.32783731]], dtype=np.float64)
        self.assertTrue(np.allclose(expected_biases, model.layers[-2].biases))

        expected_weights = np.array([[0.21512053, -0.46377846], [2.19214312, -0.58248272]], dtype=np.float64)
        self.assertTrue(np.allclose(expected_weights, model.layers[-2].weights))

    def test_rmsprop(self):
        model = Sequential()
        lmda = 0.00001

        model.add(Dense(10, kernel_regularizer=l2(lmda), input_dim=2, seed=1))
        model.add(Activation('sigmoid'))

        model.add(Dense(2, kernel_regularizer=l2(lmda), seed=3))
        model.add(Activation('tanh'))

        model.add(Dense(2, kernel_regularizer=l2(lmda), seed=6))
        model.add(Activation('softmax'))

        opt = RMSprop(learning_rate=0.01, decay=0.9)
        model.compile(optimizer=opt, loss="mean_squared_error")

        model.fit(self.X_train, self.y_train, epochs=4, batch_size=4)

        expected_biases = np.array([[-0.95917324, -0.32783731]], dtype=np.float64)
        self.assertTrue(np.allclose(expected_biases, model.layers[-2].biases))

        expected_weights = np.array([[2.10330602, -2.35170486], [3.82166843, -2.21368523]], dtype=np.float64)
        self.assertTrue(np.allclose(expected_weights, model.layers[-2].weights))

    def test_adam(self):
        model = Sequential()
        lmda = 0.00001

        model.add(Dense(10, kernel_regularizer=l2(lmda), input_dim=2, seed=1))
        model.add(Activation('sigmoid'))

        model.add(Dense(2, kernel_regularizer=l2(lmda), seed=3))
        model.add(Activation('tanh'))

        model.add(Dense(2, kernel_regularizer=l2(lmda), seed=6))
        model.add(Activation('softmax'))

        opt = Adam(learning_rate=0.01, bias_fix=0)
        model.compile(optimizer=opt, loss="mean_squared_error")

        model.fit(self.X_train, self.y_train, epochs=2, batch_size=4)

        expected_biases = np.array([[-0.95917324, -0.32783731]], dtype=np.float64)
        self.assertTrue(np.allclose(expected_biases, model.layers[-2].biases))

        expected_weights = np.array([[0.20729954, -0.45587107], [-1.22463384, 2.83373497]], dtype=np.float64)
        self.assertTrue(np.allclose(expected_weights, model.layers[-2].weights))
