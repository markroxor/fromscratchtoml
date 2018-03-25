import torch
import numpy as np


class DecisionTreeClassifier:
    def __init__(self,
                 max_depth=100,
                 min_size=100,
                 split_metric='gini_index'):
        self.use_cuda = torch.cuda.is_available()
        self.max_depth = max_depth
        self.min_size = min_size
        self.split_metric = split_metric
        self.root = {}

    def __gini_index(self, left, right, classes):
        num_instances = sum([len(left[1]), len(right[1])])
        gini = 0.0
        for group in [left, right]:
            y = group[1].numpy()
            size = len(y)
            if size == 0:
                continue
            score = 0.0
            for cl in classes:
                proportion = np.count_nonzero(y == cl) / size
                score += proportion ** 2
            gini += (1.0 - score) * (size / num_instances)
        return gini

    def __entropy_loss(self):
        pass

    def __test_split(self, index, value, x, y):
        mask = torch.nonzero(torch.gt(x[:, index], value))
        if mask.size():
            mask = mask.view(mask.size()[0])
            right = (torch.index_select(x, 0, mask), torch.index_select(y, 0, mask))
        else:
            right = (torch.DoubleTensor(), torch.DoubleTensor())

        mask = torch.nonzero(1 - torch.gt(x[:, index], value))
        if mask.size():
            mask = mask.view(mask.size()[0])
            left = (torch.index_select(x, 0, mask), torch.index_select(y, 0, mask))
        else:
            left = (torch.DoubleTensor(), torch.DoubleTensor())
        return left, right

    def fit(self, input, target):
        x = torch.from_numpy(input)
        y = torch.from_numpy(target)
        if self.use_cuda:
            x = x.cuda()
            y = y.cuda()
        self.root = self.__build_tree(x, y)
        return self.root

    def __build_tree(self, x, y):
        root = self.__get_split(x, y)
        self.__split(root, 1)
        return root

    def __get_split(self, x, y):
        b_index, b_value, b_score, b_groups = 999, 999, 999, None
        for index in range(x.size()[1]):
            for row in x:
                groups = self.__test_split(index, row[index], x, y)
                score = self.__gini_index(groups[0], groups[1], np.unique(y.numpy()))
                print('X%d < %.3f Gini=%.3f' % ((index + 1), row[index], score))
                if score < b_score:
                    b_index, b_value, b_score, b_groups = index, row[index], score, groups
        return {'index': b_index, 'value': b_value, 'groups': b_groups}

    def __to_terminal(self, group):
        outcomes = group[1].numpy()
        values, counts = np.unique(outcomes, return_counts=True)
        ind = np.argmax(counts)
        return values[ind]

    def __split(self, node, depth):
        left, right = node['groups']
        del(node['groups'])
        if not left or not right:
            node['left'] = node['right'] = self.__to_terminal(left + right)
            return

        if depth >= self.max_depth:
            node['left'] = self.__to_terminal(left)
            node['right'] = self.__to_terminal(right)
            return

        if left[1].size()[0] < self.min_size:
            node['left'] = self.__to_terminal(left)
        else:
            node['left'] = self.__get_split(left[0], left[1])
            self.__split(node['left'], depth + 1)

        if right[1].size()[0] < self.min_size:
            node['right'] = self.__to_terminal(right)
        else:
            node['right'] = self.__get_split(right[0], right[1])
            self.__split(node['right'], depth + 1)

    def print_tree(self, node, depth=0):
        if isinstance(node, dict):
            print('%s[X%d < %.3f]' % (depth * 4 * ' ', (node['index'] + 1), node['value']))
            self.print_tree(node['left'], depth + 1)
            self.print_tree(node['right'], depth + 1)
        else:
            print('%s[%s]' % (depth * 4 * ' ', node))
