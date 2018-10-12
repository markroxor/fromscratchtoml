#!/usr/bin/env python
# -*- coding: utf-8 -*-
#


import unittest

from fromscratchtoml import np
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


class Gradient_check(object):
    def function(self, h, **kwargs):
        pass

    def compute_relative_error(self, loss, **kwargs):
        # numerical solution
        self.h = 1e-6
        Ep, _ = self.function(self.h, loss, **kwargs)
        En, _ = self.function(-self.h, loss, **kwargs)

        dEdW = (np.sum(Ep - En)) / (2. * self.h)

        # analytic solution
        _, dEdW_ = self.function(0, loss, **kwargs)
        dEdW_ = dEdW_[0][0]

        relative_error = np.abs(dEdW - dEdW_) / max(np.abs(dEdW), np.abs(dEdW_))
        return relative_error


class TestGradient(unittest.TestCase, Gradient_check):
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

    def function(self, h, loss, **kwargs):
        X = kwargs["X"]
        y = kwargs["y"]

        model = Sequential()
        model.add(Dense(2, input_dim=2, seed=1))
        model.add(Activation('relu'))
        model.add(Dense(2, seed=6))
        model.add(Activation('tanh'))
        model.add(Dense(2, seed=2))
        model.add(Activation('linear'))
        model.add(Dense(2, seed=3))
        model.add(Activation('tan'))
        model.add(Dense(2, seed=4))
        model.add(Activation('tanh'))
        model.add(Dense(2, seed=7))
        model.add(Activation('leaky_relu'))
        model.add(Dense(2, seed=5))
        model.add(Activation('sigmoid'))
        model.add(Dense(2, seed=8))
        model.add(Activation('sigmoid'))
        # TODO check sigmoid as well.

        model.layers[0].weights[0][0] += h
        sgd = StochasticGradientDescent(learning_rate=0.01)

        model.compile(optimizer=sgd, loss=loss)

        y_ = model.forwardpass(X)
        E, dE = model.loss(y_, y, return_deriv=True)

        for layer in reversed(model.layers):
            dE = layer.back_propogate(dE)

        return E, model.layers[0].dEdW

    def test_gradient_consistency(self):
        kwargs = {}
        kwargs["X"] = self.X_train
        kwargs["y"] = self.y_train

        print(self.compute_relative_error(loss="mean_squared_error", **kwargs))
        print(self.compute_relative_error(loss="cross_entropy", **kwargs))

        self.assertTrue(self.compute_relative_error(loss="mean_squared_error", **kwargs) < 1e-7)
        self.assertTrue(self.compute_relative_error(loss="cross_entropy", **kwargs) < 1e-7)
