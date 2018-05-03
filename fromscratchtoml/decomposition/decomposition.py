#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Licensed under the GNU General Public License v3.0 - https://www.gnu.org/licenses/gpl-3.0.en.html

import numpy as np

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Decomposition(object):

    @staticmethod
    def cov_matrix(X, Y):
        x_norm = (X.mean(0) - X)
        y_norm = (Y.mean(0) - Y)
        cov_matrix = np.dot(x_norm.T, y_norm) / (x_norm.shape[0] - 1)
        return cov_matrix

    @staticmethod
    def eigens(X):
        cov_matrix = Decomposition.cov_matrix(X, X)
        eigen_vals, eigen_vecs = np.linalg.eig(cov_matrix)

        return eigen_vals, eigen_vecs

    @staticmethod
    def pca(X, num_components=None):
        if num_components is None:
            num_components = X.shape[1]
        eigen_vals, eigen_vecs = Decomposition.eigens(X)
        U, S, eigen_vecs = np.linalg.svd(X - X.mean(0))

        sort_index = np.argsort(-eigen_vals)
        eigen_vals = eigen_vals[sort_index]
        eigen_vecs = eigen_vecs[sort_index]

        rescaled_x = np.dot(X, eigen_vecs)

        return rescaled_x, eigen_vals[:num_components], eigen_vecs[:num_components]
