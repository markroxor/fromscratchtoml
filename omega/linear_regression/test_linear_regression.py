#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Licensed under the GNU General Public License v3.0 - https://www.gnu.org/licenses/gpl-3.0.en.html

import unittest

import torch as ch
import omega as omg
import torch.utils.data

from omega.test.toolbox import _tempfile, _test_data_path

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TestNN(unittest.TestCase):
    def setUp(self):
        # sets up a basic input dataset which implements a XOR gate.
        x = ch.Tensor([24, 50, 15])
        y = ch.Tensor([21.54945196, 47.46446305, 17.21865634])
        self.train_data = torch.utils.data.TensorDataset(x, y)

    def model_equal(self, m1, m2):
        # compares two omg.nn models by comparing their weights and biases
        for wt1, wt2 in zip(m1.layerwise_weights, m2.layerwise_weights):
            self.assertTrue(torch.equal(wt1, wt2))

        for b1, b2 in zip(m1.layerwise_biases, m2.layerwise_biases):
            self.assertTrue(torch.equal(b1, b2))

    def test_consistency(self):
        # tests for model's load save consistency.
        old_nw = omg.nn.NetworkMesh([2, 5, 2], seed=100)
        old_nw.SGD(train_data=self.train_data, epochs=15, batch_size=4, lr=3)

        fname = _tempfile("model.omg")
        old_nw.save_model(fname)

        new_nw = omg.nn.NetworkMesh()
        new_nw.load_model(fname)
        self.model_equal(old_nw, new_nw)

    def test_persistence(self):
        # ensure backward compatiblity and persistence of the model.
        model = omg.nn.NetworkMesh([2, 5, 2], seed=100)
        model.SGD(train_data=self.train_data, epochs=15, batch_size=4, lr=3)

        saved_model = omg.nn.NetworkMesh()
        saved_model.load_model(_test_data_path("xor_15_4_3_100.ch"))

        self.model_equal(model, saved_model)

    def test_inconsistency(self):
        # ensure that NotImplementedError is raised when the netowrk architecture
        # is not defined.
        model = omg.nn.NetworkMesh()
        with self.assertRaises(NotImplementedError):
            model.SGD(train_data=self.train_data, epochs=15, batch_size=4, lr=3)

    def test_model_sanity(self):
        # test when y is a list of integers (as in torch's dataloader implementation) our
        # model is still sane.
        X = ch.Tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
        Y = [0, 1, 1, 0]
        self.train_data_new = [(x, y) for x, y in zip(X, Y)]
        model = omg.nn.NetworkMesh([2, 5, 2], seed=100)
        model.SGD(train_data=self.train_data_new, epochs=15, batch_size=4, lr=3, test_data=self.train_data_new)
