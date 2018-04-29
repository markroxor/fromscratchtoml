#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 Mohit Rathore <mrmohitrathoremr@gmail.com>
# Licensed under the GNU General Public License v3.0

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
        # Linearly separable data.
        self.X = ch.Tensor([[8.0, 7], [4, 10], [9, 7], [7, 10], [9, 6], [4, 8],
                            [10, 10], [2, 7], [8, 3], [7, 5], [4, 4], [4, 6],
                            [1, 3], [2, 5]])
        self.y = ch.Tensor([1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1])

    def model_equal(self, m1, m2):
        # compares two svc models by comparing their support vectors and
        # lagrange multipliers
        self.assertTrue(m1.b, m1.b)

        for s1, s2 in zip(m1.support_vectors, m2.support_vectors):
            self.assertTrue(ch.equal(s1, s2))

        self.assertTrue(ch.equal(m1.support_vectors_y, m2.support_vectors_y))
        self.assertTrue(ch.equal(m1.support_lagrange_multipliers,
                                 m2.support_lagrange_multipliers))

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
        # NOTE save the model using py36 or else pickle will error out.
        saved_model.load_model(_test_data_path("suc_svm.fs2ml"))

        self.model_equal(model, saved_model)

    def test_not_fitted(self):
        # ensure that ModelNotFittedError is raised when predict is called
        # before predict.
        model = svm.SVC()
        with self.assertRaises(ModelNotFittedError):
            model.predict(self.X)

    def test_linear_kernel(self):
        # Tests linear kernel of svc.
        X1 = Distribution.linear(pts=100, mean=[8, 10],
                                 covr=[[1.5, 1], [1, 1.5]], seed=100)
        X2 = Distribution.linear(pts=100, mean=[9, 5],
                                 covr=[[1.5, 1], [1, 1.5]], seed=100)

        Y1 = ch.ones(X1.size()[0])
        Y2 = -ch.ones(X2.size()[0])
        X_train = ch.cat((X1, X2))
        y_train = ch.cat((Y1, Y2))

        clf_lin = svm.SVC(kernel='linear')
        clf_lin.fit(X_train, y_train)

        X1 = Distribution.linear(pts=10, mean=[8, 10],
                                 covr=[[1.5, 1], [1, 1.5]], seed=100)
        X2 = Distribution.linear(pts=10, mean=[9, 5],
                                 covr=[[1.5, 1], [1, 1.5]], seed=100)

        Y1 = ch.ones(X1.size()[0])
        Y2 = -ch.ones(X2.size()[0])

        X_test = ch.cat((X1, X2))
        y_test = ch.cat((Y1, Y2))

        predictions, projections = clf_lin.predict(X_test,
                                                   return_projection=True)

        expected_projections = ch.Tensor([5.2845, 2.8847, 3.8985, 2.4527, 4.2714,
                                        4.6425, 5.1706, 3.3409, 5.3939, 2.7791,
                                        -2.9095, -5.3093, -4.2954, -5.7412,
                                        -3.9226, -3.5514, -3.0234, -4.8531,
                                        -2.8000, -5.4149])

        self.assertTrue(torch_equal(projections, expected_projections))
        self.assertTrue(torch_equal(predictions, y_test))

    def test_poly_kernel(self):
        # Tests polynomial kernel of svc.
        X1 = Distribution.linear(pts=50, mean=[8, 20],
                                 covr=[[1.5, 1], [1, 2]], seed=100)
        X2 = Distribution.linear(pts=50, mean=[8, 15],
                                 covr=[[1.5, -1], [-1, 2]], seed=100)

        X3 = Distribution.linear(pts=50, mean=[15, 20],
                                 covr=[[1.5, 1], [1, 2]], seed=100)
        X4 = Distribution.linear(pts=50, mean=[15, 15],
                                 covr=[[1.5, -1], [-1, 2]], seed=100)

        X1 = ch.cat((X1, X2))
        X2 = ch.cat((X3, X4))

        Y1 = ch.ones(X1.size()[0])
        Y2 = -ch.ones(X2.size()[0])

        X_train = ch.cat((X1, X2))
        y_train = ch.cat((Y1, Y2))

        clf = svm.SVC(kernel='polynomial', const=1, degree=2)
        clf.fit(X_train, y_train)

        X1 = Distribution.linear(pts=5, mean=[8, 20],
                                 covr=[[1.5, 1], [1, 2]], seed=100)
        X2 = Distribution.linear(pts=5, mean=[8, 15],
                                 covr=[[1.5, -1], [-1, 2]], seed=100)

        X3 = Distribution.linear(pts=5, mean=[15, 20],
                                 covr=[[1.5, 1], [1, 2]], seed=100)
        X4 = Distribution.linear(pts=5, mean=[15, 15],
                                 covr=[[1.5, -1], [-1, 2]], seed=100)

        X1 = ch.cat((X1, X2))
        X2 = ch.cat((X3, X4))

        Y1 = ch.ones(X1.size()[0])
        Y2 = -ch.ones(X2.size()[0])

        X_test = ch.cat((X1, X2))
        y_test = ch.cat((Y1, Y2))

        predictions, projections = clf.predict(X_test, return_projection=True)
        expected_projections = ch.Tensor([1.9282, 4.1054, 4.4496, 2.8150, 3.3379,
                                         1.5935, 4.2374, 3.6997, 3.8549, 2.8403,
                                         -6.7379, -2.9163, -2.5978, -4.8333,
                                         -4.4217, -5.2334, -2.2745, -3.0599,
                                         -2.4423, -3.8900])

        self.assertTrue(torch_equal(projections, expected_projections))
        self.assertTrue(torch_equal(predictions, y_test))

    def test_rbf_kernel(self):
        # Tests RBF kernel of svc.
        X1 = Distribution.radial_binary(pts=100, mean=[0, 0], start=1, end=2,
                                        seed=100)
        X2 = Distribution.radial_binary(pts=100, mean=[0, 0], start=4, end=5,
                                        seed=100)

        Y1 = ch.ones(X1.size()[0])
        Y2 = -ch.ones(X1.size()[0])

        X_train = ch.cat((X1, X2))
        y_train = ch.cat((Y1, Y2))

        clf = svm.SVC(kernel='rbf', gamma=10)
        clf.fit(X_train, y_train)

        X1 = Distribution.radial_binary(pts=10, mean=[0, 0], start=1, end=2,
                                        seed=100)
        X2 = Distribution.radial_binary(pts=10, mean=[0, 0], start=4, end=5,
                                        seed=100)

        Y1 = ch.ones(X1.size()[0])
        Y2 = -ch.ones(X2.size()[0])

        X_test = ch.cat((X1, X2))
        y_test = ch.cat((Y1, Y2))

        predictions, projections = clf.predict(X_test, return_projection=True)

        expected_projections = ch.Tensor([1.2631, 1.3302, 1.5028, 1.2003, 1.4568,
                                         1.0555, 1.4343, 1.4228, 1.1070, 1.1050,
                                         -1.6992, -1.5001, -1.0005, -1.8284,
                                         -1.0863, -2.2380, -1.2274, -1.2235,
                                         -2.1250, -2.0870])

        self.assertTrue(torch_equal(projections, expected_projections))
        self.assertTrue(torch_equal(predictions, y_test))

    def test_multiclass(self):
        X1 = Distribution.radial_binary(pts=100, mean=[0, 0], start=1, end=2,
                                        seed=100)
        X2 = Distribution.radial_binary(pts=100, mean=[0, 0], start=4, end=5,
                                        seed=100)
        X3 = Distribution.radial_binary(pts=100, mean=[0, 0], start=6, end=7,
                                        seed=100)
        X4 = Distribution.radial_binary(pts=100, mean=[0, 0], start=8, end=9,
                                        seed=100)

        Y1 = -ch.ones(X1.size()[0])
        Y2 = ch.ones(X2.size()[0])
        Y3 = 2 * ch.ones(X3.size()[0])
        Y4 = 3000 * ch.ones(X4.size()[0])

        X_train = ch.cat((X1, X2, X3, X4))
        y_train = ch.cat((Y1, Y2, Y3, Y4))

        clf = svm.SVC(kernel='rbf', gamma=10)
        clf.fit(X_train, y_train)

        X1 = Distribution.radial_binary(pts=10, mean=[0, 0], start=1, end=2,
                                        seed=100)
        X2 = Distribution.radial_binary(pts=10, mean=[0, 0], start=4, end=5,
                                        seed=100)
        X3 = Distribution.radial_binary(pts=10, mean=[0, 0], start=6, end=7,
                                        seed=100)
        X4 = Distribution.radial_binary(pts=10, mean=[0, 0], start=8, end=9,
                                        seed=100)

        X_test = ch.cat((X1, X2, X3, X4))

        _, projections = clf.predict(X_test, return_projection=True)

        expected_projections = ch.Tensor([1.2607, 1.3279, 1.5005, 1.1980, 1.4544,
                               1.0532, 1.4320, 1.4205, 1.1046, 1.1027,
                               1.1062, 1.0377, 0.8997, 1.0960, 1.0411,
                               1.0289, 0.9724, 0.9620, 1.0358, 1.0679,
                               1.1600, 1.0981, 0.0609, 1.1698, 1.1084,
                               1.1597, 0.9454, 0.9914, 1.1755, 1.1143,
                               2.0801, 1.7532, 0.0329, 2.2153, 1.0000,
                               2.6483, 1.3649, 1.3261, 2.5342, 2.5862])

        self.assertTrue(torch_equal(projections, expected_projections))
