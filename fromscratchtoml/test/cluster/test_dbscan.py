#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 Mohit Rathore <mrmohitrathoremr@gmail.com>
# Licensed under the GNU General Public License v3.0 - https://www.gnu.org/licenses/gpl-3.0.en.html

import unittest
import numpy as np

from sklearn.datasets import load_iris

from sklearn.cluster import DBSCAN as skl_DBSCAN
from fromscratchtoml.cluster import DBSCAN as fs2ml_DBSCAN

from fromscratchtoml.toolbox.exceptions import InvalidArgumentError

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TestNN(unittest.TestCase):
    def setUp(self):
        # sets up a basic input dataset from iris dataset.
        self.eps = 0.5
        self.min_points = 5

        self.X = load_iris().data

    def test_predictions(self):
        skl_db = skl_DBSCAN(self.eps, self.min_points)
        skl_db.fit(self.X)

        fs2ml_db = fs2ml_DBSCAN(self.eps, self.min_points)
        fs2ml_db.fit(self.X)
        self.assertTrue(np.allclose(np.array(fs2ml_db.clan, dtype=np.int64), np.array(skl_db.labels_, dtype=np.int64)))

    def test_invalid_argument_error(self):
        with self.assertRaises(InvalidArgumentError):
            fs2ml_DBSCAN(eps=-1, min_neigh=1)

        with self.assertRaises(InvalidArgumentError):
            fs2ml_DBSCAN(eps=1, min_neigh=-1)

        with self.assertRaises(InvalidArgumentError):
            fs2ml_DBSCAN(eps=1, min_neigh=1.2)
