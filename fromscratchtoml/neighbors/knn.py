#!/usr/bin/env python
# -*- coding: utf-8 -*-
#



from fromscratchtoml import np
from collections import Counter

from ..base import BaseModel
from ..toolbox.exceptions import InvalidArgumentError

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class KNeighborsClassifier(BaseModel):
    """This implements k nearest neighbors supervised classification algorithm.

    Parameters
    ----------
    n_neighbors : int
        The closest number of neighbors which will decide the class of the point.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from fromscratchtoml.neighbors import KNeighborsClassifier
    >>> iris = load_iris()
    >>> iris = datasets.load_iris()
    >>> X = iris.data[:, :2]
    >>> Y = iris.target[:]
    >>> X, Y = shuffle(X, Y)
    >>> Xtrain = X[:120]
    >>> Ytrain = Y[:120]
    >>> Xtest = X[120:]
    >>> Ytest = Y[120:]
    >>> knn = KNeighborsClassifier()
    >>> knn.fit(Xtrain, Ytrain)
    >>> knn.predict(Xtest)
    array([2, 0, 1, 1, 0, 0, 2, 2, 0, 2, 0, 0, 2, 1, 1, 0, 2, 2, 1, 2, 0, 0,
           1, 0, 2, 2, 0, 1, 1, 0])

    """

    def __init__(self, n_neighbors=5):
        if not isinstance(n_neighbors, int) or n_neighbors < 0:
            raise InvalidArgumentError("Expected min_neigh to be positive int "
                                       "but got type {} and value {}".format(type(n_neighbors), n_neighbors))

        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        """Fits k nearest neighbors supervised classification algorithm.

        Parameters
        ----------
        X : numpy.ndarray
            The training features.
        y : numpy.ndarray
            The training targets.

        """
        self.X = X
        self.y = y
        return self

    def predict(self, X_test):
        """Fits and predicts using the dbscan unsupervised clustering algorithm.

        Parameters
        ----------
        X_test : numpy.ndarray
            The testing features.

        Returns
        -------
        y_target : numpy.ndarray
            The class label corresponding to each testing feature.
        """

        y_target = np.zeros(len(X_test), dtype=np.int64)

        for i, x_test in enumerate(X_test):
            # get its euclidean distance from each feature in training set.
            dist = np.array([np.linalg.norm(x_test - x_train) for x_train in self.X])

            # get the id of top n_neighbors closest neighbours
            sorted_index = dist.argsort()[:self.n_neighbors]

            # get the votes of these neighbors
            k_nearest_neighbor_votes = self.y[sorted_index]

            # get the mode of all the votes to get the final prediction
            votes = Counter(k_nearest_neighbor_votes).most_common()
            winner = votes[0][0]

            y_target[i] = winner

        return y_target
