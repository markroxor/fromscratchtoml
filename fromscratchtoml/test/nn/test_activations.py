#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 Mohit Rathore <mrmohitrathoremr@gmail.com>
# Licensed under the GNU General Public License v3.0 - https://www.gnu.org/licenses/gpl-3.0.en.html

import unittest

import numpy as np
from fromscratchtoml.models.nn import Activations

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TestNN(unittest.TestCase):
    def setUp(self):
        self.X = np.linspace(-5, 5, 20)

    def test_linear(self):
        Y = Activations.linear(self.X)

        expected_Y = np.array([-5., -4.47368421, -3.94736842, -3.42105263, -2.89473684,
                               -2.36842105, -1.84210526, -1.31578947, -0.78947368, -0.26315789,
                               0.26315789, 0.78947368, 1.31578947, 1.84210526, 2.36842105,
                               2.89473684, 3.42105263, 3.94736842, 4.47368421, 5.])
        self.assertTrue(np.allclose(Y, expected_Y))

    def test_tanh(self):
        Y = Activations.tanh(self.X)

        expected_Y = np.array([-0.9999092, -0.99973988, -0.99925488, -0.99786657, -0.99389948,
                               -0.98261979, -0.95099682, -0.865733, -0.65811078, -0.25724684,
                               0.25724684, 0.65811078, 0.865733, 0.95099682, 0.98261979,
                               0.99389948, 0.99786657, 0.99925488, 0.99973988, 0.9999092])
        self.assertTrue(np.allclose(Y, expected_Y))

    def test_sigmoid(self):
        Y = Activations.sigmoid(self.X)

        expected_Y = np.array([0.00669285, 0.01127661, 0.0189398, 0.03164396, 0.05241435,
                               0.08561266, 0.1368025, 0.21151967, 0.31228169, 0.43458759,
                               0.56541241, 0.68771831, 0.78848033, 0.8631975, 0.91438734,
                               0.94758565, 0.96835604, 0.9810602, 0.98872339, 0.99330715])
        self.assertTrue(np.allclose(Y, expected_Y))

    def test_relu(self):
        Y = Activations.relu(self.X)

        expected_Y = np.array([0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
                               0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
                               0.26315789, 0.78947368, 1.31578947, 1.84210526, 2.36842105,
                               2.89473684, 3.42105263, 3.94736842, 4.47368421, 5.00000000])
        self.assertTrue(np.allclose(Y, expected_Y))

    def test_leaky_relu(self):
        Y = Activations.leaky_relu(self.X)
        expected_Y = np.array([-1.50000000, -1.34210526, -1.18421053, -1.02631579, -0.86842105,
                              -0.71052632, -0.55263158, -0.39473684, -0.23684211, -0.07894737,
                              0.26315789, 0.78947368, 1.31578947, 1.84210526, 2.36842105,
                              2.89473684, 3.42105263, 3.94736842, 4.47368421, 5.00000000])

        self.assertTrue(np.allclose(Y, expected_Y))

    def test_step(self):
        Y = Activations.step(self.X)
        expected_Y = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        self.assertTrue(np.allclose(Y, expected_Y))

    def test_softmax(self):
        Y = Activations.softmax(self.X)
        expected_Y = np.array([0.00001858, 0.00003145, 0.00005323, 0.00009011, 0.00015252,
                               0.00025817, 0.00043700, 0.00073971, 0.00125209, 0.00211939,
                               0.00358746, 0.00607243, 0.01027872, 0.01739862, 0.02945038,
                               0.04985021, 0.08438068, 0.14282987, 0.24176593, 0.40923346])
        self.assertTrue(np.allclose(Y, expected_Y))
