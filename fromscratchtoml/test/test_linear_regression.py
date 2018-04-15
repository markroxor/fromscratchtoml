#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Licensed under the GNU General Public License v3.0 - https://www.gnu.org/licenses/gpl-3.0.en.html

import unittest

import torch as ch
import fromscratchtoml as omg

from fromscratchtoml.test.toolbox import _tempfile, _test_data_path

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TestLinearRegression(unittest.TestCase):
    def setUp(self):
        # sets up a basic input dataset which implements a XOR gate.
        self.x = ch.Tensor([[1.0], [2.0], [3.0]])
        self.y = ch.Tensor([[2.0], [4.0], [6.0]])

    def model_equal(self, m1, m2):
        # compares two omg.nn models by comparing their weights and biases
        self.assertTrue(ch.equal(m1.alpha, m2.alpha))
        self.assertTrue(ch.equal(m1.beta, m2.beta))

    def test_consistency(self):
        # tests for model's load save consistency.
        old_model = omg.linear_regression.LinearRegressionClassifier()
        old_model.fit(self.x, self.y, optimizer='analytical')

        fname = _tempfile("model.omg")
        old_model.save_model(fname)

        new_model = omg.linear_regression.LinearRegressionClassifier()
        new_model.load_model(fname)

        self.model_equal(old_model, new_model)

    def test_persistence(self):
        # ensure backward compatiblity and persistence of the model.
        new_model = omg.linear_regression.LinearRegressionClassifier()
        new_model.fit(self.x, self.y, optimizer='analytical')

        saved_model = omg.linear_regression.LinearRegressionClassifier()
        saved_model.load_model(_test_data_path("lr_analytical_model.ch"))

        self.model_equal(new_model, saved_model)

    def test_inconsistency(self):
        # ensure that NotImplementedError is raised when fit is not called.
        model = omg.linear_regression.LinearRegressionClassifier()
        with self.assertRaises(NotImplementedError):
            model.predict(self.x)
