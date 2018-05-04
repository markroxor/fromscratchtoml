#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU General Public License v3.0 - https://www.gnu.org/licenses/gpl-3.0.en.html

import unittest

from fromscratchtoml.decomposition import Decomposition
import numpy as np

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TestDecisionTree(unittest.TestCase):
    def setUp(self):
        n_features = 5
        n_dim = 3
        self.n_comp = 3

        self.X = np.zeros([n_features, n_dim])
        for i in range(n_features):
            self.X[i, 2] = i
            self.X[i, 1] = 5 * i + 2
        #     X[i,1] = pow(i,2)
            self.X[i, 0] = (5 - 2 * self.X[i, 0] - 3 * self.X[i, 1]) / 2

    def test_cov_matrix(self):
        cov_matrix = Decomposition.cov_matrix(self.X, self.X)
        expected_cov_matrix = [[140.625, -93.75, -18.75],
                               [-93.75, 62.5, 12.5],
                               [-18.75, 12.5, 2.5]]
        self.assertTrue(np.allclose(cov_matrix, expected_cov_matrix))

    def test_eigen(self):
        eigen_vals, eigen_vecs = Decomposition.eigens(self.X)
        expected_eigen_vals = [2.05625000e+02, -2.84217094e-14, -3.48372238e-16]
        expected_eigen_vecs = [[0.82697677, -0.56223609, 0.02881115],
                               [-0.55131785, -0.81091744, -0.15430393],
                               [-0.11026357, -0.16218349, 0.98760327]]
        self.assertTrue(np.allclose(eigen_vals, expected_eigen_vals))
        self.assertTrue(np.allclose(eigen_vecs, expected_eigen_vecs))

    def test_pca(self):
        principal_components, rescaled_X = Decomposition.pca(self.X, num_components=3, return_scaled=True)
        expected_rescaled_X = [[1.81383571e+01, 2.22044605e-16, 3.33066907e-16],
                                [9.06917857e+00, 1.11022302e-16, 1.66533454e-16],
                                [0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                                [-9.06917857e+00, -1.11022302e-16, -1.66533454e-16],
                                [-1.81383571e+01, -2.22044605e-16, -3.33066907e-16]]
        self.assertTrue(np.allclose(rescaled_X, expected_rescaled_X))

        expected_principal_components = [[0.82697677, 0.02881115, -0.56223609],
                               [-0.55131785, -0.15430393, -0.81091744],
                               [-0.11026357, 0.98760327, -0.16218349]]
        self.assertTrue(np.allclose(principal_components, expected_principal_components))

        reconstructed_x = np.dot(rescaled_X, principal_components.T) + self.X.mean(axis=0)
        self.assertTrue(np.allclose(reconstructed_x, self.X))
