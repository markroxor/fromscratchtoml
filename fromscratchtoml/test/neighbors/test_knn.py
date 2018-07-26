#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 Mohit Rathore <mrmohitrathoremr@gmail.com>
# Licensed under the GNU General Public License v3.0 - https://www.gnu.org/licenses/gpl-3.0.en.html

import unittest
from fromscratchtoml import np

from sklearn.neighbors import KNeighborsClassifier as sk_KNeighborsClassifier
from sklearn import datasets
from sklearn.utils import shuffle

from fromscratchtoml.neighbors import KNeighborsClassifier as fs2ml_KNeighborsClassifier

from fromscratchtoml.toolbox.exceptions import InvalidArgumentError

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TestKNN(unittest.TestCase):
    def setUp(self):
        # sets up a basic input dataset from iris dataset.
        iris = datasets.load_iris()

        X = iris.data[:, :2]
        Y = iris.target[:]
        X, Y = shuffle(X, Y, random_state=10)

        self.Xtrain = X[:120]
        self.Ytrain = Y[:120]
        self.Xtest = X[120:]
        self.Ytest = Y[120:]

    def test_predictions(self):
        sk_knn = sk_KNeighborsClassifier()
        sk_knn.fit(self.Xtrain, self.Ytrain)

        fs2ml_knn = fs2ml_KNeighborsClassifier()
        fs2ml_knn.fit(self.Xtrain, self.Ytrain)

        sk_labels = sk_knn.predict(self.Xtest)
        fs2ml_labels = fs2ml_knn.predict(self.Xtest)

        self.assertTrue(np.allclose(sk_labels, fs2ml_labels))

    def test_invalid_argument_error(self):
        with self.assertRaises(InvalidArgumentError):
            fs2ml_KNeighborsClassifier(n_neighbors=-1)

        with self.assertRaises(InvalidArgumentError):
            fs2ml_KNeighborsClassifier(n_neighbors=1.2)
