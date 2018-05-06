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


class DBSCAN(BaseModel):
    """This implements dbscan unsupervised clustering algorithm.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from fromscratchtoml.cluster import DBSCAN
    >>> eps = 0.5
    >>> min_points = 5
    >>> iris = load_iris()
    >>> X = iris.data
    >>> db = DBSCAN(eps, min_points)
    >>> db.fit(X)
    >>> labels = db.clan
    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0,
           0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, -1, 1, 1, 1, 1, 1,
           1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1,
           1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, -1,
           -1, 1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1,
           1, -1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          dtype=object)

    """

    def __init__(self, eps, min_neigh):
        """Initializing the dbscan class parameters.

        Parameters
        ----------
        eps : float, int
            The eucledian range within which the next point should lie.
        min_neigh : int
            The minimum number of points required to form a cluster.

        """
        if not (isinstance(eps, float) or isinstance(eps, int)) or eps < 0:
            raise InvalidArgumentError("Expected eps to be a positive float or int "
                                       "but got type {} and value {}".format(type(eps), eps))

        if not isinstance(min_neigh, int) or min_neigh < 0:
            raise InvalidArgumentError("Expected min_neigh to be positive int "
                                       "but got type {} and value {}".format(type(min_neigh), min_neigh))

        self.eps = eps
        self.min_walking_dist = eps
        self.min_neigh = min_neigh

    def __get_neighbours(self, villager_id):
        """Get all the surrounding neighbors in range of eps.

        Parameters
        ----------
        villager_id : int
            The index of the point in our feature vector.

        Returns
        -------
        neighbor_ids : list
            The index of all the surrounding neighbors in range eps.
        """

        villager = self.village[villager_id]
        neighbor_ids = []

        for potential_neighbor_id in range(self.n_houses):
            potential_neighbor = self.village[potential_neighbor_id]

            # finds the euclidian distance between the points.
            walking_dist = np.linalg.norm(villager - potential_neighbor)

            if walking_dist <= self.min_walking_dist:
                neighbor_ids.append(potential_neighbor_id)

        return neighbor_ids

    def fit(self, X):
        """Fits the dbscan unsupervised clustering algorithm.

        Parameters
        ----------
        X : numpy.ndarray
            The training features.

        """
        self.village = X
        self.n_houses = len(self.village)

        # When a villager is not assigned a clan his clan is None
        self.clan = np.array([None] * self.n_houses)
        current_clan_id = 0

        for villager_id in range(self.n_houses):
            # if the villager is not assigned a clan
            if self.clan[villager_id] is None:
                # get all his neighbors, fitting the criteria in __init__
                neighbor_ids = self.__get_neighbours(villager_id)

                # if he is an isolated villager he will be assigned -1 clan
                # AKA the isolated clan.
                if len(neighbor_ids) < self.min_neigh:
                    self.clan[villager_id] = -1
                    continue

                # else he and his neighbors will be assigned the same clan
                for neighbor_id in neighbor_ids:
                    self.clan[neighbor_id] = current_clan_id

                for neighbor_id in neighbor_ids:
                    # these neighbors will try to convince their neighbors
                    # to join there clan.
                    neighbors_neighbors_ids = self.__get_neighbours(neighbor_id)

                    # if their number is more than the required threshold they
                    # are allowed to join the clan.
                    # Only those villagers are allowed to join the clan who are
                    # not already a part of any clan or are part of the isolated
                    # clan.
                    if len(neighbors_neighbors_ids) >= self.min_neigh:
                        for neighbors_neighbor_id in neighbors_neighbors_ids:
                            # if they have not been allocated a clan before
                            # they also have the priviledge to recruit more
                            # villagers.
                            if self.clan[neighbors_neighbor_id] is None:
                                neighbor_ids.append(neighbors_neighbor_id)
                                self.clan[neighbors_neighbor_id] = current_clan_id

                            # isolated ones have already been given the chance to
                            # recruit more members, but they are indeed isloated.
                            elif self.clan[neighbors_neighbor_id] == -1:
                                self.clan[neighbors_neighbor_id] = current_clan_id

                # When a new clan is formed we get a new clan ID.
                current_clan_id += 1
        return self

    def fit_predict(self, X):
        """Fits and predicts using the dbscan unsupervised clustering algorithm.

        Parameters
        ----------
        X : numpy.ndarray
            The training features.

        Returns
        -------
        clan : numpy.ndarray
            The class label corresponding to each data point.
        """
        self.fit(X)
        return self.clan
