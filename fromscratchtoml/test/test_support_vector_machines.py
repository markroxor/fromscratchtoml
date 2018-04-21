#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 Mohit Rathore <mrmohitrathoremr@gmail.com>
# Copyright (C) 2017 Dikshant Gupta <dikshant2210@gmail.com>
# Licensed under the GNU General Public License v3.0 - https://www.gnu.org/licenses/gpl-3.0.en.html

import unittest

import torch as ch
from fromscratchtoml.models import svm

from fromscratchtoml.test.toolbox import _tempfile, _test_data_path

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
        self.assertTrue(ch.equal(m1.w, m1.w))
        self.assertTrue(m1.b, m1.b)

        for s1, s2 in zip(m1.support_vectors, m2.support_vectors):
            self.assertTrue(ch.equal(s1, s2))

        self.assertTrue(ch.equal(m1.support_vectors_y, m2.support_vectors_y))

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

    def test_inconsistency(self):
        # ensure that NotImplementedError is raised when the netowrk architecture
        # is not defined.
        # TODO
        pass

    def test_model_sanity(self):
        # test when y is a list of integers (as in torch's dataloader implementation) our
        # model is still sane.
        # TODO
        pass
