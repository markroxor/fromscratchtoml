#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 Mohit Rathore <mrmohitrathoremr@gmail.com>
# Licensed under the GNU General Public License v3.0 - https://www.gnu.org/licenses/gpl-3.0.en.html

import unittest

import torch as ch

from fromscratchtoml.models import svm
from fromscratchtoml.test.toolbox import _tempfile, _test_data_path, torch_equal
from fromscratchtoml.toolbox.exceptions import ModelNotFittedError
from fromscratchtoml.toolbox.random import Distribution

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TestNN(unittest.TestCase):
    def setUp(self):
        # sets up a basic input dataset which implements a XOR gate.
        self.X = ch.Tensor([[8.0, 7], [4, 10], [9, 7], [7, 10], [9, 6], [4, 8], [10, 10],
                            [2, 7], [8, 3], [7, 5], [4, 4], [4, 6], [1, 3], [2, 5]])
        self.y = ch.Tensor([1, 1, 1, 1, 1, 1, 1,
                            -1, -1, -1, -1, -1, -1, -1])

    def model_equal(self, m1, m2):
        # compares two fs2ml.nn models by comparing their weights and biases
        self.assertTrue(m1.b, m1.b)

        for s1, s2 in zip(m1.support_vectors, m2.support_vectors):
            self.assertTrue(ch.equal(s1, s2))

        self.assertTrue(ch.equal(m1.support_vectors_y, m2.support_vectors_y))
        self.assertTrue(ch.equal(m1.effective_lagrange_multipliers, m2.effective_lagrange_multipliers))

    def test_consistency(self):
        # tests for model's load save consistency.
        old_model = svm.SVC()
        old_model.fit(self.X, self.y)

        fname = _tempfile("model.fs2ml")
        old_model.save_model(fname)

        new_model = svm.SVC()
        new_model.load_model(fname)
        self.model_equal(old_model, new_model)

    def test_persistence(self):
        # ensure backward compatiblity and persistence of the model.
        model = svm.SVC()
        model.fit(self.X, self.y)

        saved_model = svm.SVC()
        saved_model.load_model(_test_data_path("suc_svm.fs2ml"))

        self.model_equal(model, saved_model)

    def test_not_fitted(self):
        # ensure that ModelNotFittedError is raised when predict is called
        # before predict.
        model = svm.SVC()
        with self.assertRaises(ModelNotFittedError):
            model.predict(self.X)

    def test_linear_kernel(self):
        # test when y is a list of integers (as in torch's dataloader implementation) our
        # model is still sane.
        X1_train = Distribution.linear(pts=100, mean=[8, 10], covr=[[1.5, 1], [1, 1.5]], seed=100)
        X2_train = Distribution.linear(pts=100, mean=[9, 5], covr=[[1.5, 1], [1, 1.5]], seed=100)

        Y1_train = ch.ones(X1_train.size()[0])
        Y2_train = -ch.ones(X2_train.size()[0])
        X_train = ch.cat((X1_train, X2_train))
        y_train = ch.cat((Y1_train, Y2_train))

        X1_test = Distribution.linear(pts=10, mean=[8, 10], covr=[[1.5, 1], [1, 1.5]], seed=100)
        X2_test = Distribution.linear(pts=10, mean=[9, 5], covr=[[1.5, 1], [1, 1.5]], seed=100)
        X_test = ch.cat((X1_test, X2_test))

        clf_lin = svm.SVC(kernel='linear')
        clf_lin.fit(X_train, y_train)

        _, projections = clf_lin.predict(X_test, return_projection=True)

        expected_projection = ch.Tensor([5.2845, 2.8847, 3.8985, 2.4527, 4.2714,
                                        4.6425, 5.1706, 3.3409, 5.3939, 2.7791,
                                        -2.9095, -5.3093, -4.2954, -5.7412, -3.9226,
                                        -3.5514, -3.0234, -4.8531, -2.8000, -5.4149])

        self.assertTrue(torch_equal(projections, expected_projection))

    def test_poly_kernel(self):
        X1 = Distribution.linear(pts=50, mean=[8, 20], covr=[[1.5, 1], [1, 2]], seed=100)
        X2 = Distribution.linear(pts=50, mean=[8, 15], covr=[[1.5, -1], [-1, 2]], seed=100)

        X3 = Distribution.linear(pts=50, mean=[15, 20], covr=[[1.5, 1], [1, 2]], seed=100)
        X4 = Distribution.linear(pts=50, mean=[15, 15], covr=[[1.5, -1], [-1, 2]], seed=100)

        X1 = ch.cat((X1, X2))
        X2 = ch.cat((X3, X4))

        Y1 = ch.ones(X1.size()[0])
        Y2 = -ch.ones(X2.size()[0])

        X_train = ch.cat((X1, X2))
        y_train = ch.cat((Y1, Y2))

        clf = svm.SVC(kernel='polynomial', const=1, degree=2)
        clf.fit(X_train, y_train)

        X1 = Distribution.linear(pts=5, mean=[8, 20], covr=[[1.5, 1], [1, 2]], seed=100)
        X2 = Distribution.linear(pts=5, mean=[8, 15], covr=[[1.5, -1], [-1, 2]], seed=100)

        X3 = Distribution.linear(pts=5, mean=[15, 20], covr=[[1.5, 1], [1, 2]], seed=100)
        X4 = Distribution.linear(pts=5, mean=[15, 15], covr=[[1.5, -1], [-1, 2]], seed=100)

        X1 = ch.cat((X1, X2))
        X2 = ch.cat((X3, X4))
        X_test = ch.cat((X1, X2))

        _, projections = clf.predict(X_test, return_projection=True)
        expected_projection = ch.Tensor([1.9282, 4.1054, 4.4496, 2.8150, 3.3379,
                                         1.5935, 4.2374, 3.6997, 3.8549, 2.8403,
                                         -6.7379, -2.9163, -2.5978, -4.8333, -4.4217,
                                         -5.2334, -2.2745, -3.0599, -2.4423, -3.8900])

        self.assertTrue(torch_equal(projections, expected_projection))

    def test_radial_kernel(self):
        X1 = Distribution.radial_binary(pts=100, mean=[0, 0], start=1, end=2, seed=100)
        X2 = Distribution.radial_binary(pts=100, mean=[0, 0], start=4, end=5, seed=100)

        Y1 = ch.ones(X1.size()[0])
        Y2 = -ch.ones(X1.size()[0])

        X_train = ch.cat((X1, X2))
        y_train = ch.cat((Y1, Y2))

        clf = svm.SVC(kernel='rbf', gamma=10)
        clf.fit(X_train, y_train)

        X1 = Distribution.radial_binary(pts=10, mean=[0, 0], start=1, end=2, seed=100)
        X2 = Distribution.radial_binary(pts=10, mean=[0, 0], start=4, end=5, seed=100)

        X_test = ch.cat((X1, X2))

        _, projections = clf.predict(X_test, return_projection=True)

        expected_projection = ch.Tensor([1.2587, 1.3259, 1.4985, 1.1960, 1.4524,
                                         1.0512, 1.4300, 1.4184, 1.1026, 1.1007,
                                         -1.7036, -1.5044, -1.0048, -1.8327, -1.0906,
                                         -2.2424, -1.2318, -1.2278, -2.1293, -2.0913])

        self.assertTrue(torch_equal(projections, expected_projection))
