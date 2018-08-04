#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 Mohit Rathore <mrmohitrathoremr@gmail.com>
# Licensed under the GNU General Public License v3.0 - https://www.gnu.org/licenses/gpl-3.0.en.html

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


class TestNN(unittest.TestCase):
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

    def test_dense_l2_sgd_mse(self):
        model = Sequential()
        lmda = 0.001

        model.add(Dense(10, kernel_regularizer=l2(lmda), input_dim=2, seed=1))
        model.add(Activation('sigmoid'))

        model.add(Dense(2, kernel_regularizer=l2(lmda), seed=7))
        model.add(Activation('tanh'))

        model.add(Dense(2, kernel_regularizer=l2(lmda), seed=2))
        model.add(Activation('relu'))

        model.add(Dense(2, kernel_regularizer=l2(lmda), seed=3))
        model.add(Activation('leaky_relu'))

        model.add(Dense(2, kernel_regularizer=l2(lmda), seed=4))
        model.add(Activation('linear'))

        model.add(Dense(2, kernel_regularizer=l2(lmda), seed=6))
        model.add(Activation('softmax'))

        sgd = StochasticGradientDescent(learning_rate=0.05)
        model.compile(optimizer=sgd, loss="mean_squared_error")

        model.fit(self.X_train, self.y_train, epochs=15, batch_size=4)

        expected_biases = np.array([[0.82234889, -0.30852266]], dtype=np.float128)
        self.assertTrue(np.allclose(expected_biases, model.layers[-2].biases))

        expected_weights = np.array([[-1.21168207, 1.31585794], [-0.10285933, -0.78498722]], dtype=np.float128)
        self.assertTrue(np.allclose(expected_weights, model.layers[-2].weights))

    def test_dense_l1_sgd_cross_entropy(self):
        model = Sequential()
        lmda = 0.001

        model.add(Dense(10, kernel_regularizer=l1(lmda), input_dim=2, seed=1))
        model.add(Activation('sigmoid'))

        model.add(Dense(2, kernel_regularizer=l1(lmda), seed=7))
        model.add(Activation('tanh'))

        model.add(Dense(2, kernel_regularizer=l1(lmda), seed=2))
        model.add(Activation('relu'))

        model.add(Dense(2, kernel_regularizer=l1(lmda), seed=3))
        model.add(Activation('leaky_relu'))

        model.add(Dense(2, kernel_regularizer=l1(lmda), seed=4))
        model.add(Activation('linear'))

        model.add(Dense(2, kernel_regularizer=l1(lmda), seed=6))
        model.add(Activation('softmax'))

        sgd = StochasticGradientDescent(learning_rate=0.05)
        model.compile(optimizer=sgd, loss="cross_entropy")

        model.fit(self.X_train, self.y_train, epochs=10, batch_size=2)

        expected_biases = np.array([[0.01003523, -1.89462709]], dtype=np.float128)
        self.assertTrue(np.allclose(expected_biases, model.layers[-2].biases))

        expected_weights = np.array([[-2.23959131, -0.33495123], [-1.22481013, -3.02053907]], dtype=np.float128)
        self.assertTrue(np.allclose(expected_weights, model.layers[-2].weights))

    def test_dense_l1_l2_sgd_cross_entropy(self):
        model = Sequential()
        lmda1 = 0.001
        lmda2 = 0.001

        model.add(Dense(10, kernel_regularizer=l1_l2(lmda1, lmda2), input_dim=2, seed=1))
        model.add(Activation('sigmoid'))

        model.add(Dense(2, kernel_regularizer=l1_l2(lmda1, lmda2), seed=7))
        model.add(Activation('tanh'))

        model.add(Dense(2, kernel_regularizer=l1_l2(lmda1, lmda2), seed=2))
        model.add(Activation('relu'))

        model.add(Dense(2, kernel_regularizer=l1_l2(lmda1, lmda2), seed=3))
        model.add(Activation('leaky_relu'))

        model.add(Dense(2, kernel_regularizer=l1_l2(lmda1, lmda2), seed=4))
        model.add(Activation('linear'))

        model.add(Dense(2, kernel_regularizer=l1_l2(lmda1, lmda2), seed=6))
        model.add(Activation('softmax'))

        sgd = StochasticGradientDescent(learning_rate=0.05)
        model.compile(optimizer=sgd, loss="cross_entropy")

        model.fit(self.X_train, self.y_train, epochs=10, batch_size=2)

        expected_biases = np.array([[-0.39717598, -0.87858446]], dtype=np.float128)
        self.assertTrue(np.allclose(expected_biases, model.layers[-2].biases))

        expected_weights = np.array([[-2.04528524, 0.59513791], [-0.84020769, -1.03224048]], dtype=np.float128)
        self.assertTrue(np.allclose(expected_weights, model.layers[-2].weights))
