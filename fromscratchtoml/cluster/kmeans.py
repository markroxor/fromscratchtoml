#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 Mohit Rathore <mrmohitrathoremr@gmail.com>
# Licensed under the GNU General Public License v3.0 - https://www.gnu.org/licenses/gpl-3.0.en.html

import numpy as np

from ..base import BaseModel
from fromscratchtoml.toolbox.exceptions import InvalidArgumentError

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class KMeans(BaseModel):

    def __init__(self, n_clusters=2, max_iter=500, seed=100):
        if not isinstance(n_clusters, int) or n_clusters <= 0:
            raise InvalidArgumentError("Expected n_clusters to be a positive int "
                                       "but got type {} and value {}".format(type(n_clusters), n_clusters))

        if not isinstance(max_iter, int) or max_iter <= 0:
            raise InvalidArgumentError("Expected n_clusters to be a positive int "
                                       "but got type {} and value {}".format(type(max_iter), max_iter))

        np.random.seed(seed)
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def __get_new_centers(self, centers):
        converged = True

        for label in range(self.n_clusters):
            indices = np.where(self.labels == label)
            pts_in_cluster = self.X[indices]
            new_center = pts_in_cluster.mean(0)

            if not np.allclose(centers[label], new_center):
                centers[label] = new_center
                converged = False

        return centers, converged

    def fit(self, X):
        self.X = X

        center_ids = np.random.randint(self.X.shape[0], size=self.n_clusters)
        centers = self.X[center_ids]
        self.labels = -np.ones(self.X.shape[0], dtype=np.float64)

        converged = False
        for i in range(self.max_iter):
            if converged:
                break

            converged = True

            for j, x in enumerate(self.X):
                min_dist_center = np.linalg.norm(x - centers[0])
                label = 0

                for i, center in enumerate(centers):
                    if min_dist_center > np.linalg.norm(x - center):
                        min_dist_center = np.linalg.norm(x - center)
                        label = i
                self.labels[j] = label

            centers, converged = self.__get_new_centers(centers)

        return self

    def fit_predict(self, X):
        self.fit(X)
        return self.labels
