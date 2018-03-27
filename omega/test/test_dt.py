#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 Mohit Rathore <mrmohitrathoremr@gmail.com>
# Copyright (C) 2017 Dikshant Gupta <dikshant2210@gmail.com>
# Licensed under the GNU General Public License v3.0 - https://www.gnu.org/licenses/gpl-3.0.en.html

import unittest

import torch
from omega.DecisionTree import DecisionTreeClassifier

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TestDecisionTree(unittest.TestCase):
    def setUp(self):
        self.x = torch.Tensor([[2.771244718, 1.784783929],
                               [1.728571309, 1.169761413],
                               [3.678319846, 2.81281357],
                               [3.961043357, 2.61995032],
                               [2.999208922, 2.209014212],
                               [7.497545867, 3.162953546],
                               [9.00220326, 3.339047188],
                               [7.444542326, 0.476683375],
                               [10.12493903, 3.234550982],
                               [6.642287351, 3.319983761]])
        self.y = torch.Tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    def model_equal(self, observed, expected):
        self.assertEqual(observed, expected)

    def test_model_depth_2(self):
        dt = DecisionTreeClassifier(2, 1)
        root = dt.fit(self.x, self.y)

        nodes = [root]
        expected = [6.64229, 2.77124, 7.49755]
        observed = list()
        while True:
            if isinstance(nodes[0], dict):
                observed.append(float("{0:.5f}".format(nodes[0]['value'])))
                nodes.append(nodes[0]['left'])
                nodes.append(nodes[0]['right'])

            nodes = nodes[1:]
            if not nodes:
                break
        self.model_equal(observed, expected)

    def test_model_depth_3(self):
        dt = DecisionTreeClassifier(3, 1)
        root = dt.fit(self.x, self.y)
        nodes = [root]
        expected = [6.64229, 2.77124, 7.49755, 1.72857, 2.77124, 7.44454, 7.49755]
        observed = list()
        while True:
            if isinstance(nodes[0], dict):
                observed.append(float("{0:.5f}".format(nodes[0]['value'])))
                nodes.append(nodes[0]['left'])
                nodes.append(nodes[0]['right'])

            nodes = nodes[1:]
            if not nodes:
                break
        self.model_equal(observed, expected)

    def test_predict(self):
        dt = DecisionTreeClassifier(2, 1)
        dt.fit(self.x, self.y)

        x1 = torch.Tensor([1.728571309, 1.169761413])
        y1 = torch.Tensor([0])

        x2 = torch.Tensor([10.12493903, 3.234550982])
        y2 = torch.Tensor([1])

        self.assertEqual(y1[0], dt.predict(x1))
        self.assertEqual(y2[0], dt.predict(x2))

    def test_numpy_input(self):
        dt = DecisionTreeClassifier(2, 1)
        root = dt.fit(self.x.numpy(), self.y.numpy())

        nodes = [root]
        expected = [6.64229, 2.77124, 7.49755]
        observed = list()
        while True:
            if isinstance(nodes[0], dict):
                observed.append(float("{0:.5f}".format(nodes[0]['value'])))
                nodes.append(nodes[0]['left'])
                nodes.append(nodes[0]['right'])

            nodes = nodes[1:]
            if not nodes:
                break
        self.model_equal(observed, expected)

    def test_min_size(self):
        dt = DecisionTreeClassifier(2, 6)
        root = dt.fit(self.x, self.y)

        nodes = [root]
        expected = [6.64229]
        observed = list()
        while True:
            if isinstance(nodes[0], dict):
                observed.append(float("{0:.5f}".format(nodes[0]['value'])))
                nodes.append(nodes[0]['left'])
                nodes.append(nodes[0]['right'])

            nodes = nodes[1:]
            if not nodes:
                break
        self.model_equal(observed, expected)
