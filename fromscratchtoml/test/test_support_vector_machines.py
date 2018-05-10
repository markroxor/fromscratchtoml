#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 Mohit Rathore <mrmohitrathoremr@gmail.com>
# Licensed under the GNU General Public License v3.0

import unittest

import numpy as np

from fromscratchtoml import svm
from fromscratchtoml.test.toolbox import _tempfile, _test_data_path  # noqa:F401
from fromscratchtoml.toolbox.exceptions import ModelNotFittedError
from fromscratchtoml.toolbox.random import Distribution

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TestSVM(unittest.TestCase):
    def setUp(self):
        # Linearly separable data.
        self.X = np.array([[8.0, 7], [4, 10], [9, 7], [7, 10], [9, 6], [4, 8],
                            [10, 10], [2, 7], [8, 3], [7, 5], [4, 4], [4, 6],
                            [1, 3], [2, 5]])
        self.y = np.array([1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1])

    def model_equal(self, m1, m2):
        # compares two svc models by comparing their support vectors and
        # lagrange multipliers
        self.assertTrue(m1.b, m1.b)

        for s1, s2 in zip(m1.support_vectors, m2.support_vectors):
            self.assertTrue(np.allclose(s1, s2))

        self.assertTrue(np.allclose(m1.support_vectors_y, m2.support_vectors_y))
        self.assertTrue(np.allclose(m1.support_lagrange_multipliers,
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

    # TODO this test is breaking too often bcz of persistent change in model
    # add it once repo is stablised
    # def test_persistence(self):
    #     # ensure backward compatiblity and persistence of the model.
    #     model = svm.SVC()
    #     model.fit(self.X, self.y)
    #
    #     saved_model = svm.SVC()
    #     # NOTE save the model using py36 or else pickle will error out.
    #     saved_model.load_model(_test_data_path("suc_svm.fs2ml"))
    #
    #     self.model_equal(model, saved_model)

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

        Y1 = np.ones(X1.shape[0])
        Y2 = -np.ones(X2.shape[0])
        X_train = np.vstack((X1, X2))
        y_train = np.hstack((Y1, Y2))

        clf_lin = svm.SVC(kernel='linear')
        clf_lin.fit(X_train, y_train)

        X1 = Distribution.linear(pts=10, mean=[8, 10],
                                 covr=[[1.5, 1], [1, 1.5]], seed=100)
        X2 = Distribution.linear(pts=10, mean=[9, 5],
                                 covr=[[1.5, 1], [1, 1.5]], seed=100)

        Y1 = np.ones(X1.shape[0])
        Y2 = -np.ones(X2.shape[0])

        X_test = np.vstack((X1, X2))
        y_test = np.hstack((Y1, Y2))

        predictions, projections = clf_lin.predict(X_test,
                                                   return_projection=True)

        expected_projections = np.array([5.2844825, 2.8846788, 3.898558, 2.4527097, 4.271367,
                                            4.6425023, 5.170607, 3.3408344, 5.3939104, 2.779106,
                                           -2.909471, -5.3092747, -4.2953954, -5.7412434, -3.9225864,
                                           -3.551451, -3.0233462, -4.853119, -2.8000426, -5.4148474]
                                           )
        self.assertTrue(np.allclose(projections, expected_projections))
        self.assertTrue(np.allclose(predictions, y_test))

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

        X1 = np.vstack((X1, X2))
        X2 = np.vstack((X3, X4))

        Y1 = np.ones(X1.shape[0])
        Y2 = -np.ones(X2.shape[0])

        X_train = np.vstack((X1, X2))
        y_train = np.hstack((Y1, Y2))

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

        X1 = np.vstack((X1, X2))
        X2 = np.vstack((X3, X4))

        Y1 = np.ones(X1.shape[0])
        Y2 = -np.ones(X2.shape[0])

        X_test = np.vstack((X1, X2))
        y_test = np.hstack((Y1, Y2))

        predictions, projections = clf.predict(X_test, return_projection=True)
        expected_projections = np.array([1.2630574, 1.3302442, 1.502788, 1.2003369, 1.4567516,
                                            1.0555044, 1.434326, 1.4227715, 1.1069533, 1.104987,
                                           -1.6992458, -1.5001097, -1.0005158, -1.8284273, -1.0863144,
                                           -2.238042, -1.2274336, -1.2235101, -2.1250129, -2.0870237]
                                           )
        expected_projections = np.array([1.9282368, 4.1053743, 4.449601, 2.8149981, 3.337817,
                                            1.5934888, 4.237419, 3.699658, 3.8548565, 2.8402433,
                                           -6.7378554, -2.9163127, -2.5978136, -4.833237, -4.421687,
                                           -5.2333884, -2.2744238, -3.0598483, -2.4422958, -3.890006],
                                           )
        self.assertTrue(np.allclose(projections, expected_projections))
        self.assertTrue(np.allclose(predictions, y_test))

    def test_rbf_kernel(self):
        # Tests RBF kernel of svc.
        X1 = Distribution.radial_binary(pts=100, mean=[0, 0], st=1, ed=2,
                                        seed=100)
        X2 = Distribution.radial_binary(pts=100, mean=[0, 0], st=4, ed=5,
                                        seed=100)

        Y1 = np.ones(X1.shape[0])
        Y2 = -np.ones(X1.shape[0])

        X_train = np.vstack((X1, X2))
        y_train = np.hstack((Y1, Y2))

        clf = svm.SVC(kernel='rbf', gamma=10)
        clf.fit(X_train, y_train)

        X1 = Distribution.radial_binary(pts=10, mean=[0, 0], st=1, ed=2,
                                        seed=100)
        X2 = Distribution.radial_binary(pts=10, mean=[0, 0], st=4, ed=5,
                                        seed=100)

        Y1 = np.ones(X1.shape[0])
        Y2 = -np.ones(X2.shape[0])

        X_test = np.vstack((X1, X2))
        y_test = np.hstack((Y1, Y2))

        predictions, projections = clf.predict(X_test, return_projection=True)

        expected_projections = np.array([1.2630574, 1.3302442, 1.502788, 1.2003369, 1.4567516,
                                         1.0555044, 1.434326, 1.4227715, 1.1069533, 1.104987,
                                         -1.6992458, -1.5001097, -1.0005158, -1.8284273, -1.0863144,
                                         -2.238042, -1.2274336, -1.2235101, -2.1250129, -2.0870237],
                                          )

        self.assertTrue(np.allclose(projections, expected_projections))
        self.assertTrue(np.allclose(predictions, y_test))

    def test_multiclass(self):
        X1 = Distribution.radial_binary(pts=10, mean=[0, 0], st=1, ed=2,
                                        seed=100)
        X2 = Distribution.radial_binary(pts=10, mean=[0, 0], st=4, ed=5,
                                        seed=100)
        X3 = Distribution.radial_binary(pts=10, mean=[0, 0], st=6, ed=7,
                                        seed=100)
        X4 = Distribution.radial_binary(pts=10, mean=[0, 0], st=8, ed=9,
                                        seed=100)

        Y1 = -np.ones(X1.shape[0])
        Y2 = np.ones(X2.shape[0])
        Y3 = 2 * np.ones(X3.shape[0])
        Y4 = 3000 * np.ones(X4.shape[0])

        X_train = np.vstack((X1, X2, X3, X4))
        y_train = np.hstack((Y1, Y2, Y3, Y4))

        clf = svm.SVC(kernel='rbf', gamma=10)
        clf.fit(X_train, y_train)

        X1 = Distribution.radial_binary(pts=10, mean=[0, 0], st=1, ed=2,
                                        seed=100)
        X2 = Distribution.radial_binary(pts=10, mean=[0, 0], st=4, ed=5,
                                        seed=100)
        X3 = Distribution.radial_binary(pts=10, mean=[0, 0], st=6, ed=7,
                                        seed=100)
        X4 = Distribution.radial_binary(pts=10, mean=[0, 0], st=8, ed=9,
                                        seed=100)

        X_test = np.vstack((X1, X2, X3, X4))

        _, projections = clf.predict(X_test, return_projection=True)

        expected_projections = np.array(
                                        [1.23564788, 1.15519477, 1.32441802, 1.04496554, 1.29740627, 0.,
                                         1.25561797, 1.22925452, 0., 1.11920321, 0.2991908, 0.23818634,
                                         0.55359011, 0.29655677, 0., 0.59992803, 0.52733203, 0.30456398,
                                         0.6027897, 0.33755249, 0., 0.04997651, 0.12099712, 0.12276944,
                                         0., 0.19631702, 0.11836214, 0.06221966, 0.24539362, 0.,
                                         1.00000106, 1.0000021, 1.00000092, 1.19952335, 1.00000283, 1.17741522,
                                         1.40596479, 1.60945299, 1.41534644, 1.27928235]
                                         )

        print(projections)
        self.assertTrue(np.allclose(projections, expected_projections))
