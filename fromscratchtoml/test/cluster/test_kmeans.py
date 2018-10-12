#!/usr/bin/env python
# -*- coding: utf-8 -*-
#



import unittest
from fromscratchtoml import np

from sklearn.cluster import KMeans as skl_KMeans
from fromscratchtoml.cluster import KMeans as fs2ml_KMeans

from fromscratchtoml.toolbox.exceptions import InvalidArgumentError
from fromscratchtoml.toolbox.random import Distribution

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TestKMeans(unittest.TestCase):
    def setUp(self):
        self.n_clusters = 3
        X1 = Distribution.linear(pts=10, covr=[[1, -1], [-1, 1]], mean=[0, 0])
        X2 = Distribution.linear(pts=10, covr=[[1, -1], [-1, 1]], mean=[0, -6])
        X3 = Distribution.linear(pts=10, covr=[[1, -1], [-1, 1]], mean=[6, -3])

        self.X = np.vstack([X1, X2, X3])

    def test_predictions(self):
        skl_km = skl_KMeans(n_clusters=self.n_clusters, random_state=5)
        skl_km.fit(self.X)
        skl_labels = sorted(np.array(skl_km.labels_, dtype=np.int64))

        fs2ml_km = fs2ml_KMeans(n_clusters=self.n_clusters, seed=5)
        fs2ml_labels = fs2ml_km.fit_predict(self.X)
        fs2ml_labels = sorted(np.array(fs2ml_labels, dtype=np.int64))

        self.assertTrue(np.allclose(fs2ml_labels, skl_labels))

    def test_invalid_argument_error(self):
        with self.assertRaises(InvalidArgumentError):
            fs2ml_KMeans(n_clusters=-3, max_iter=100, seed=2)

        with self.assertRaises(InvalidArgumentError):
            fs2ml_KMeans(n_clusters=3, max_iter=-100, seed=2)

        with self.assertRaises(InvalidArgumentError):
            fs2ml_KMeans(n_clusters=3, max_iter=100, seed=-2)

        with self.assertRaises(InvalidArgumentError):
            fs2ml_KMeans(n_clusters=3.3, max_iter=100, seed=2)

        with self.assertRaises(InvalidArgumentError):
            fs2ml_KMeans(n_clusters=3, max_iter=10.2, seed=2)

        with self.assertRaises(InvalidArgumentError):
            fs2ml_KMeans(n_clusters=3, max_iter=100, seed=2.2)
