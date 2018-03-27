#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 Mohit Rathore <mrmohitrathoremr@gmail.com>
# Copyright (C) 2017 Dikshant Gupta <dikshant2210@gmail.com>
# Licensed under the GNU General Public License v3.0 - https://www.gnu.org/licenses/gpl-3.0.en.html

import torch
import numpy as np


class DecisionTreeClassifier:
    """Objects of this class are decision tree based classifiers.

    Examples
    --------
    >>> from omega.DecisionTree import DecisionTreeClassifier
    >>> import torch
    >>> x = torch.Tensor([[2.771244718, 1.784783929], [1.728571309, 1.169761413], [7.444542326, 0.476683375])
    >>> y = torch.Tensor([0, 0 ,1])
    >>> dt = DecisionTreeClassifier()
    >>> dt.fit(x, y)
    >>> dt.predict(torch.Tensor([10.12493903, 3.234550982]))
    1.0
    """

    def __init__(self,
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
        self.__use_cuda = torch.cuda.is_available()
        self.__max_depth = max_depth
        self.__min_size = min_size
        self.__split_metric = split_metric
        self.__root = {}

    def __gini_index(self, left, right, classes):
        """Calculates the value of gini index for each split.

        Parameters
        ----------
        left :  a fraction of the original data
        right : a fraction of the original data
        classes : list of integers
            list of all possible values to target labels.
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

    def __test_split(self, index, value, x, y):
        """Splits the dataset into two parts on the basis of a value.

        Parameters
        ----------
        index :  index of the attribute on which to split
        value : attribute value on which to split
        x : training data
        y : training labels
        """
        mask = torch.nonzero(torch.lt(x[:, index], value))
        if mask.size():
            mask = mask.view(mask.size()[0])
            left = (torch.index_select(x, 0, mask), torch.index_select(y, 0, mask))
        else:
            left = (torch.DoubleTensor(), torch.DoubleTensor())

        mask = torch.nonzero(1 - torch.lt(x[:, index], value))
        if mask.size():
            mask = mask.view(mask.size()[0])
            right = (torch.index_select(x, 0, mask), torch.index_select(y, 0, mask))
        else:
            right = (torch.DoubleTensor(), torch.DoubleTensor())
        return left, right

    def fit(self, x, y):
        """Trains the classifier on the training data and forms the decision tree

        Parameters
        ----------
        x : training data
        y : training labels

        Returns
        -------
        A dictionary containing the tree.
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y)
        if self.__use_cuda:
            x = x.cuda()
            y = y.cuda()
        self.__root = self.__build_tree(x, y)
        return self.__root

    def __build_tree(self, x, y):
        """Initiates the tree building process by calling for the first split.

        Parameters
        ----------
        x : training data
        y : training labels

        Returns
        -------
        A dictionary containing the tree.
        """
        root = self.__get_split(x, y)
        self.__split(root, 1)
        return root

    def __get_split(self, x, y):
        """Finds the best split given the training data and labels

        Parameters
        ----------
        x : training data
        y : training labels

        Returns
        -------
        A node for the tree with best split value given the data.
        """
        b_index, b_value, b_score, b_groups = 999, 999, 999, None

        classes = list()
        for val in y:
            if val not in classes:
                classes.append(val)

        for index in range(x.size()[1]):
            for row in x:
                groups = self.__test_split(index, row[index], x, y)
                score = self.__gini_index(groups[0], groups[1], classes)
                if score < b_score:
                    b_index, b_value, b_score, b_groups = index, row[index], score, groups
        return {'index': b_index, 'value': b_value, 'groups': b_groups}

    def __to_terminal(self, group):
        """Determines the label of a terminal node.

        Parameters
        ----------
        group : list
            list contains two elements, data and labels of the rows associated with the split.

        Returns
        -------
        target label of a terminal node.
        """
        res = torch.mode(group[1])[0][0]
        return res

    def __split(self, node, depth):
        """Build the subtree given a node, keeps track of maximum depth of the tree.

        Parameters
        ----------
        node : node for which the subtree is built
        depth : depth of the node
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
            node['left'] = self.__get_split(left[0], left[1])
            self.__split(node['left'], depth + 1)

        if right[1].size()[0] < self.__min_size:
            node['right'] = self.__to_terminal(right)
        else:
            node['right'] = self.__get_split(right[0], right[1])
            self.__split(node['right'], depth + 1)

    def predict(self, row):
        """Given a row as data, predicts the class of the dataset.

        Parameters
        ----------
        row : row similar to the training data

        Returns
        -------
        Predicted class of the input data.
        """
        node = self.__root
        while True:
            if isinstance(node, dict):
                if row[node['index']] < node['value']:
                    node = node['left']
                else:
                    node = node['right']
            else:
                return node
