import numpy as np
from scipy.stats.stats import pearsonr
from ..base import BaseModel

class DTModel(object, BaseModel):

    def __init__(self, leaf_size=1, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose

    def _build_tree(self, dataX, dataY):
        # shape of data - n rows * m cols
        [samples_n, features_n] = dataX.shape

        # if n < leaf_size or unique data in Y then return leaf node
        leaf = np.array([-1, dataY.mean(), np.nan, np.nan])
        if samples_n <= self.leaf_size or len(np.unique(dataY)) == 1:
            return leaf

        # find the feature correlation with label
        feature_corrs = list()
        for feature_i in range(features_n):
            abs_cor = abs(pearsonr(dataX[:, feature_i], dataY)[0])
            abs_cor = abs_cor if not np.isnan(abs_cor) else 0.0
            feature_corrs.append((feature_i, abs_cor))
        
        # sort the feature correlation tuple by abs corr value
        feature_corrs = sorted(feature_corrs, key = lambda x: x[1] , reverse = True)

        # select the best feature available to split
        feature_cor_max_i = -1
        for (feature_i, abs_cor) in feature_corrs:            
            #split by current max feature
            split_val = np.median(dataX[:, feature_i])
            left_index = dataX[:, feature_i] <= split_val
            right_index = dataX[:, feature_i] > split_val
            
            # if split gives two non-empty groups
            if len(np.unique(left_index)) > 1:
                feature_cor_max_i = feature_i
                break
        
        # if no feature with split found then return leaf
        if feature_cor_max_i == -1:
            return leaf

        # build left tree and right tree based on indices
        left_tree = self._build_tree(dataX[left_index], dataY[left_index])
        right_tree = self._build_tree(dataX[right_index], dataY[right_index])

        # calculate left tree shape
        left_tree_shape = left_tree.shape[0] if left_tree.ndim > 1 else 1

        # create root node
        root = np.array([feature_cor_max_i, split_val, 1, left_tree_shape + 1])

        # Append root with left tree and right tree and then return
        return np.vstack((root, left_tree, right_tree))

    def fit(self, dataX, dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        self.tree = self._build_tree(dataX, dataY)


    def _search_tree(self, dataX):
        samples_n, features_n = dataX.shape
        dataY = np.zeros([samples_n])
        tree = self.tree
        for i, x in enumerate(dataX):
            leaf_idx = 0
            while int(tree[leaf_idx][0]) != -1: # while tree not at leaf index
                feature_idx, split_val, left_idx, right_idx = tree[leaf_idx]
                feature_idx = int(feature_idx)
                left_idx = int(left_idx)
                right_idx = int(right_idx)
                
                if  x[feature_idx] <= split_val: # go left tree
                    leaf_idx += left_idx
                else: #go right tree
                    leaf_idx += right_idx
            dataY[i] = tree[leaf_idx][1]
        return dataY

    def test(self, points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        return self._search_tree(points)