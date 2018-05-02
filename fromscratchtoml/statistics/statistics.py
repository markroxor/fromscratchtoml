#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Licensed under the GNU General Public License v3.0 - https://www.gnu.org/licenses/gpl-3.0.en.html

import torch as ch
import numpy as np

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Statistics(object):

    @staticmethod
    def cov_matrix(X, Y):
        x_norm = (X.mean(dim=0) - X)
        y_norm = (Y.mean(dim=0) - Y)
        cov_matrix = ch.mm(x_norm.t(), y_norm) / (x_norm.size()[0] - 1)
        return cov_matrix

    @staticmethod
    def eigens(X):
        cov_matrix = Statistics.cov_matrix(X, X)
        # since covariance matrix is symmetric we can use linalg.eigh for efficient
        # computing
        eigen_vals, eigen_vecs = np.linalg.eig(cov_matrix.numpy())

        return ch.Tensor(eigen_vals), ch.Tensor(eigen_vecs)

    @staticmethod
    def pca(X, num_components=2):
        eigen_vals, eigen_vecs = Statistics.eigens(X)

        sort_index = np.argsort(-eigen_vals.numpy())
        eigen_vals = eigen_vals[sort_index]
        eigen_vecs = eigen_vecs[:, sort_index].t()

        rescaled_x = ch.mm(X, eigen_vecs[:, :num_components].t())

        return rescaled_x, eigen_vals[:num_components], eigen_vecs[:, :num_components]
