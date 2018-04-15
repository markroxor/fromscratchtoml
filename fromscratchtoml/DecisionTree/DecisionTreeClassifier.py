#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Licensed under the GNU General Public License v3.0 - https://www.gnu.org/licenses/gpl-3.0.en.html

import torch as ch
import numpy as np


class DecisionTreeClassifier(object):
    """Objects of this class are decision tree based classifiers.

    Parameters
    ----------
    max_depth : int, optional, default value 100
        Represents maximum allowed depth of the binary tree.
    min_size : int, optional, default value 2
        Represents minimum number of data points associated with each node.
    split_metric : str, optional, default value gini_index
        possible values - gini_index, entropy_loss(under development)
        Represents the metrics to compare each split point.

    Examples
    --------
    >>> from fromscratchtoml.DecisionTree import DecisionTreeClassifier
    >>> import torch as ch
    >>> x = ch.Tensor([[2.771244718, 1.784783929], [1.728571309, 1.169761413], [7.444542326, 0.476683375]])
    >>> y = ch.Tensor([0, 0 ,1])
    >>> dt = DecisionTreeClassifier()
    >>> dt.fit(x, y)
    >>> dt.predict(ch.Tensor([10.12493903, 3.234550982]))
    1.0
    """

    def __init__(
             self,
             max_depth=100,
             min_size=2,
             split_metric='gini_index'):
        """Initializes the parameters of a decision tree and create root node
        the decision as an empty dictionary.

        Parameters
        ----------
        max_depth : int, optional, default value 100
            Represents maximum allowed depth of the binary tree.
        min_size : int, optional, default value 2
            Represents minimum number of data points associated with each node.
        split_metric : str, optional, default value gini_index
            possible values - gini_index, entropy_loss(under development)
            Represents the metrics to compare each split point.
        """
        self.__use_cuda = ch.cuda.is_available()
        self.__max_depth = max_depth
        self.__min_size = min_size
        self.__split_metric = split_metric
        self.root = {}

    def __gini_index(self, left, right, classes):
        """Calculates the value of gini index for each split.

        Parameters
        ----------
        left :  ch.Tensor
            a fraction of the original data
        right : ch.Tensor
            a fraction of the original data
        classes : list
            list of all possible values to target labels.
        Returns
        -------
        gini : float
            gini index of the split
        """
        num_instances = sum([len(left[1]), len(right[1])])
        gini = 0.0
        for group in [left, right]:
            y = group[1]
            size = len(y) * 1.0
            if size == 0:
                continue
            score = 0.0
            for cl in classes:
                proportion = sum(y == cl) / size
                score += proportion ** 2
            gini += (1.0 - score) * (size / num_instances)
        return gini

    def __test_split(self, feature_index, value, x, y):
        """Splits the dataset into two parts on the basis of a value.

        Parameters
        ----------
        feature_index :  int
            index of the attribute on which to split
        value : float
            attribute value on which to split
        x : ch.tensor
            training data
        y : ch.tensor
            training labels

        Returns
        -------
        left : torch.Tensor
        right : torch.Tensor
            Two parts of the dataset x and y after split
        """
        mask = ch.nonzero(ch.lt(x[:, feature_index], value))
        if mask.size():
            mask = mask.view(mask.size()[0])
            left = (ch.index_select(x, 0, mask), ch.index_select(y, 0, mask))
        else:
            left = (ch.DoubleTensor(), ch.DoubleTensor())

        mask = ch.nonzero(1 - ch.lt(x[:, feature_index], value))
        if mask.size():
            mask = mask.view(mask.size()[0])
            right = (ch.index_select(x, 0, mask), ch.index_select(y, 0, mask))
        else:
            right = (ch.DoubleTensor(), ch.DoubleTensor())
        return left, right

    def fit(self, x, y):
        """Trains the classifier on the training data and forms the decision tree

        Parameters
        ----------
        x : ch.tensor
            training data
        y : ch.tensor
            training labels

        Returns
        -------
        root : dict
            A dictionary containing the tree.
        """
        if isinstance(x, np.ndarray):
            x = ch.from_numpy(x)
        if isinstance(y, np.ndarray):
            y = ch.from_numpy(y)
        if self.__use_cuda:
            x = x.cuda()
            y = y.cuda()
        self.root = self.__build_tree(x, y)
        return self.root

    def __build_tree(self, x, y):
        """Initiates the tree building process by calling for the first split.

        Parameters
        ----------
        x : ch.tensor
            training data
        y : ch.tensor
            training labels

        Returns
        -------
        root : dict
            A dictionary containing the tree.
        """
        root = self.__get_best_split(x, y)
        self.__split(root, 1)
        return root

    def __get_best_split(self, x, y):
        """Finds the best split given the training data and labels

        Parameters
        ----------
        x : ch.tensor
            training data
        y : ch.tensor
            training labels

        Returns
        -------
        dict : dict
            A node for the tree with best split value given the data.
        """
        b_index, b_value, b_score, b_groups = 999, 999, 999, None

        classes = set(y)

        for feature_index in range(x.size()[1]):
            for row in x:
                groups = self.__test_split(feature_index, row[feature_index], x, y)
                score = self.__gini_index(groups[0], groups[1], classes)
                if score < b_score:
                    b_index, b_value, b_score, b_groups = feature_index, row[feature_index], score, groups
        return {'index': b_index, 'value': b_value, 'groups': b_groups}

    def __to_terminal(self, group):
        """Determines the label of a terminal node.

        Parameters
        ----------
        group : list
            list contains two elements, data and labels of the rows associated with the split.

        Returns
        -------
        res : float
            target label of a terminal node.
        """
        res = ch.mode(group[1])[0][0]
        return res

    def __split(self, node, depth):
        """Build the subtree given a node, keeps track of maximum depth of the tree.

        Parameters
        ----------
        node : dict
            node for which the subtree is built
        depth : int
            depth of the node
        """
        left, right = node['groups']
        del(node['groups'])
        if not left[1].size() or not right[1].size():
            group = left if left[1].size() else right
            node['left'] = node['right'] = self.__to_terminal(group)
            return

        if depth >= self.__max_depth:
            node['left'] = self.__to_terminal(left)
            node['right'] = self.__to_terminal(right)
            return

        if left[1].size()[0] < self.__min_size:
            node['left'] = self.__to_terminal(left)
        else:
            node['left'] = self.__get_best_split(left[0], left[1])
            self.__split(node['left'], depth + 1)

        if right[1].size()[0] < self.__min_size:
            node['right'] = self.__to_terminal(right)
        else:
            node['right'] = self.__get_best_split(right[0], right[1])
            self.__split(node['right'], depth + 1)

    def predict(self, row):
        """Given a row as data, predicts the class of the dataset.

        Parameters
        ----------
        row : torch.Tensor
            row similar to the training data

        Returns
        -------
        node : float
            Predicted class of the input data.
        """
        node = self.root
        while True:
            if isinstance(node, dict):
                if row[node['index']] < node['value']:
                    node = node['left']
                else:
                    node = node['right']
            else:
                return node

    def save_model(self, file_path):
        """This function saves the model in a file for loading it in future.

        Parameters
        ----------
        file_path : str
            The path to file where the model should be saved.

        """
        ch.save(self.__dict__, file_path)
        return

    def load_model(self, file_path):
        """This function loads the saved model from a file.

        Parameters
        ----------
        file_path : str
            The path of file from where the model should be retrieved.

        """
        self.__dict__ = ch.load(file_path)
        return
