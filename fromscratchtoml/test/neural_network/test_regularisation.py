#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author - Mohit Rathore <mrmohitrathoremr@gmail.com>
# Licensed under The MIT License - https://opensource.org/licenses/MIT

import unittest

from fromscratchtoml import np
from fromscratchtoml.neural_network.models import Sequential
from fromscratchtoml.neural_network.layers import Dense, Activation
from fromscratchtoml.neural_network.optimizers import StochasticGradientDescent
from fromscratchtoml.neural_network.regularizers import l1, l2, l1_l2

from sklearn.model_selection import train_test_split
from fromscratchtoml.toolbox.preprocess import to_onehot
from fromscratchtoml.toolbox.random import Distribution

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TestRegularisation(unittest.TestCase):
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

    def test_l1(self):
        lmda = 0.001

        model = Sequential(verbose=1)
        model.add(Dense(10, kernel_regularizer=l1(lmda), input_dim=2, seed=1))
        model.add(Activation('sigmoid'))

        model.add(Dense(2, kernel_regularizer=l1(lmda), seed=6))
        model.add(Activation('tanh'))

        model.add(Dense(2, kernel_regularizer=l1(lmda), seed=6))
        model.add(Activation('softmax'))

        sgd = StochasticGradientDescent(learning_rate=0.05)
        model.compile(optimizer=sgd, loss="cross_entropy")

        model.fit(self.X_train, self.y_train, epochs=10, batch_size=2)
        print(model.layers[-2].biases)
        print(model.layers[-2].weights)

        expected_biases = np.array([[-0.95917324, -0.32783731]], dtype=np.float64)
        self.assertTrue(np.allclose(expected_biases, model.layers[-2].biases))

        expected_weights = np.array([[0.19999472, -3.19873907], [2.10903048, -3.24881063]], dtype=np.float64)
        self.assertTrue(np.allclose(expected_weights, model.layers[-2].weights))

    def test_l2(self):
        lmda = 0.001

        model = Sequential(verbose=1)
        model.add(Dense(10, kernel_regularizer=l2(lmda), input_dim=2, seed=1))
        model.add(Activation('sigmoid'))

        model.add(Dense(2, kernel_regularizer=l2(lmda), seed=6))
        model.add(Activation('tanh'))

        model.add(Dense(2, kernel_regularizer=l2(lmda), seed=6))
        model.add(Activation('softmax'))

        sgd = StochasticGradientDescent(learning_rate=0.05)
        model.compile(optimizer=sgd, loss="cross_entropy")

        model.fit(self.X_train, self.y_train, epochs=10, batch_size=2)
        print(model.layers[-2].biases)
        print(model.layers[-2].weights)

        expected_biases = np.array([[-0.95917324, -0.32783731]], dtype=np.float64)
        self.assertTrue(np.allclose(expected_biases, model.layers[-2].biases))

        expected_weights = np.array([[1.58872834, -1.65159914], [2.04547398, -1.63848661]], dtype=np.float64)
        self.assertTrue(np.allclose(expected_weights, model.layers[-2].weights))

    def test_l1_l2(self):
        lmda1 = 0.001
        lmda2 = 0.001

        model = Sequential(verbose=1)
        model.add(Dense(10, kernel_regularizer=l1_l2(lmda1, lmda2), input_dim=2, seed=1))
        model.add(Activation('sigmoid'))

        model.add(Dense(2, kernel_regularizer=l1_l2(lmda1, lmda2), seed=6))
        model.add(Activation('tanh'))

        model.add(Dense(2, kernel_regularizer=l1_l2(lmda1, lmda2), seed=6))
        model.add(Activation('softmax'))

        sgd = StochasticGradientDescent(learning_rate=0.05)
        model.compile(optimizer=sgd, loss="cross_entropy")

        model.fit(self.X_train, self.y_train, epochs=9, batch_size=2)
        print(model.layers[-2].biases)
        print(model.layers[-2].weights)

        expected_biases = np.array([[-0.95917324, -0.32783731]], dtype=np.float64)
        self.assertTrue(np.allclose(expected_biases, model.layers[-2].biases))

        expected_weights = np.array([[0.71132812, -2.20343103], [1.44723471, -2.40020303]], dtype=np.float64)
        self.assertTrue(np.allclose(expected_weights, model.layers[-2].weights))
