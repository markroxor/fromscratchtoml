#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 Mohit Rathore <mrmohitrathoremr@gmail.com>
# Copyright (C) 2017 Dikshant Gupta <dikshant2210@gmail.com>
# Licensed under the GNU General Public License v3.0 - https://www.gnu.org/licenses/gpl-3.0.en.html

import unittest

import torch as ch
import omega as omg
import torch.utils.data

from omega.test.toolbox import _tempfile, _test_data_path


class TestNN(unittest.TestCase):
    def setUp(self):
        x = ch.Tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = ch.Tensor([[1, 0], [0, 1], [0, 1], [1, 0]])
        self.train_data = torch.utils.data.TensorDataset(x, y)

    def model_equal(self, m1, m2):
        for wt1, wt2 in zip(m1.layerwise_weights, m2.layerwise_weights):
            self.assertTrue(torch.equal(wt1, wt2))

        for b1, b2 in zip(m1.layerwise_biases, m2.layerwise_biases):
            self.assertTrue(torch.equal(b1, b2))

    def test_persistence(self):
        old_nw = omg.nn.NetworkMesh([2, 5, 2], seed=100)
        old_nw.SGD(train_data=self.train_data, epochs=15, batch_size=4, eta=3)

        fname = _tempfile("model.omg")
        old_nw.save_model(fname)

        new_nw = omg.nn.NetworkMesh()
        new_nw.load_model(fname)
        self.model_equal(old_nw, new_nw)

    def test_persistence1(self):
        # backward compatiblity
        model = omg.nn.NetworkMesh([2, 5, 2], seed=100)
        model.SGD(train_data=self.train_data, epochs=15, batch_size=4, eta=3)

        saved_model = omg.nn.NetworkMesh()
        saved_model.load_model(_test_data_path("xor_15_4_3_100.ch"))

        self.model_equal(model, saved_model)

        # raise error when architecture not defined
        model = omg.nn.NetworkMesh()
        with self.assertRaises(NotImplementedError):
            model.SGD(train_data=self.train_data, epochs=15, batch_size=4, eta=3)

        # test y as number TODO working?
        x = ch.Tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = ch.Tensor([[0], [1], [1], [0]])
        self.train_data_new = torch.utils.data.TensorDataset(x, y)
        model = omg.nn.NetworkMesh([2, 5, 2], seed=100)
        model.SGD(train_data=self.train_data_new, epochs=15, batch_size=4, eta=3)

        # evaluation consistency
        model = omg.nn.NetworkMesh([2, 5, 2], seed=100)
        model.SGD(train_data=self.train_data_new, epochs=15, batch_size=4, eta=3, test_data=self.train_data)
