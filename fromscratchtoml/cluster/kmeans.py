#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author - Mohit Rathore <mrmohitrathoremr@gmail.com>
# Licensed under The MIT License - https://opensource.org/licenses/MIT

from fromscratchtoml import np

from ..base import BaseModel
from ..toolbox.exceptions import InvalidArgumentError

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class KMeans(BaseModel):
    """Implements the kmeans unsupervised clustering algorithm.

    Examples
    --------
    >>> from fromscratchtoml import np
    >>> from fromscratchtoml.cluster import KMeans as KMeans
    >>> from fromscratchtoml.toolbox.random import Distribution
    >>> X1 = Distribution.linear(pts=500, covr=[[1.2, -1],[-1, 1]], mean=[0, 0])
    >>> X2 = Distribution.linear(pts=500, covr=[[1.2, -1],[-1, 1]], mean=[-1, -2])
    >>> X3 = Distribution.linear(pts=500, covr=[[1.2, -1],[-1, 1]], mean=[6, -3])
    >>> X = np.vstack([X1, X2, X3])
    >>> KMeans(n_clusters=3).fit_predict(X)
    array([1., 0., 1., ..., 2., 2., 2.])

    Parameters
    ----------
    n_clusters : int
        The number of clusters in which data will be clustered.
    max_iter : int, optional
        The upper limit of the number of iterations to perform before converging.
    seed: int, optional
        Numpy's random seed.
    """

    def __init__(self, n_clusters, max_iter=500, seed=None):
        if not isinstance(n_clusters, int) or n_clusters <= 0:
            raise InvalidArgumentError("Expected n_clusters to be a positive int "
                                       "but got type {} and value {}".format(type(n_clusters), n_clusters))

        if not isinstance(max_iter, int) or max_iter <= 0:
            raise InvalidArgumentError("Expected n_clusters to be a positive int "
                                       "but got type {} and value {}".format(type(max_iter), max_iter))

        if seed and (not isinstance(seed, int) or seed <= 0):
            raise InvalidArgumentError("Expected n_clusters to be a positive int "
                                       "but got type {} and value {}".format(type(seed), seed))

        if seed:
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
        """Fits the kmeans unsupervised clustering algorithm.

        Parameters
        ----------
        X : numpy.ndarray
            The training features.

        """
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
        """Fits and predicts using the knn unsupervised clustering algorithm.

        Parameters
        ----------
        X : numpy.ndarray
            The training features.

        Returns
        -------
        labels : numpy.ndarray
            The class label corresponding to each data point.
        """
        self.fit(X)
        return self.labels
