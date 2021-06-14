from utils.exceptions import NotFittedError, InvalidArgError
import numpy as np

def entropy(buckets, total = None):
    if total is None: total = buckets.sum()
    if total == 0: return 1
    probs = buckets / total
    probs *= np.log2(probs, out = np.zeros_like(probs), where=(probs != 0))
    return -probs.sum()

def gini(buckets, total = None):
    if total is None: total = buckets.sum()
    if total == 0: return 1
    return 1 - np.square(buckets / total).sum()

CRITERIA_FUNCS = {'gini': gini, 'entropy': entropy }

class DecisionTree:
    '''
        DISCLAIMER: Not very optimised, fast implementation would likely be in CPython
    '''

    class Node:
        def __init__(self, X, y, buckets, predicted_class, n_samples, impurity):
            self.X = X
            self.y = y
            self.buckets = buckets # 1-dimensional np.ndarray with #n_classes counts for each class
            self.predicted_class = predicted_class
            self.n_samples = n_samples
            self.impurity = impurity
            self.feature_index = -1
            self.thres = -1
            self.left = None
            self.right = None
        
        def debug(self, tabs=0):
            if self.left: print('  '*tabs, 'Inner[index=', self.feature_index, ', thres=', self.thres, ', n_samples=', self.n_samples, ']')
            else: print('  '*tabs, 'Leaf[n_samples=', self.n_samples, ']')
            if self.left:
                self.left.debug(tabs+1)
                self.right.debug(tabs+1)


    def __init__(self, criterion = "gini", max_depth = None, min_samples_split = 2, min_impurity_decrease = 0.0):
        self.n_leaf_nodes = 0
        self.criterion = criterion

        if max_depth is not None and max_depth <= 0:
            raise InvalidArgError("max_depth must be a positive integer.")
        self.max_depth = max_depth

        if min_samples_split < 2: 
            raise InvalidArgError("min_samples_split must be at least 2.")
        self.min_samples_split = min_samples_split

        if min_impurity_decrease < 0.0:
            raise InvalidArgError("min_impurity_decrease must be positive.")
        self.min_impurity_decrease = min_impurity_decrease


    def fit(self, X, y, sample_weight = None):
        self.criterion = CRITERIA_FUNCS[self.criterion];

        if sample_weight is None:
            self.weighted_n_samples = len(X)
        elif isinstance(sample_weight, np.ndarray):
            self.weighted_n_samples = sample_weight.sum()
        else:
            raise InvalidArgError("Expect numpy array for sample_weight")

        self.n_classes = len(np.unique(y))
        self.sample_weight = sample_weight
        self.samples = X

        if isinstance(self.min_samples_split, float):
            self.min_samples_split = self.min_samples_split * self.n_samples

        self.tree_ = self._make_tree(X, y, 0, np.arange(X.shape[0]))

    def predict(self, X):
        if self.tree_ is None: 
            raise NotFittedError("Model not fitted yet. Call'fit' on the data before invoking 'predict'")
        
        n_samples = X.shape[0]
        preds = np.zeros((n_samples, ), dtype=int)
        for i in range(n_samples):
            cur_node = self.tree_
            while cur_node.left:
                if X[i, cur_node.feature_index] < cur_node.thres:
                    cur_node = cur_node.left
                else:
                    cur_node = cur_node.right
            preds[i] = cur_node.predicted_class
        return preds
    
    def debug(self):
        if self.tree_ is None: 
            raise NotFittedError("Estimator not fitted yet. Call'fit' on the data before invoking 'predict'")
        self.tree_.debug()

    def _make_tree(self, X, y, depth, indices):
        '''
            Constructs decision tree recursively and stores it in self.tree_
        '''
        buckets = np.zeros(self.n_classes)
        for i in range(self.n_classes):
            if self.sample_weight is not None:
                buckets[i] = np.sum(self.sample_weight[np.nonzero(y == i)])
            else:
                buckets[i] = np.sum(y==i)

        impurity = self.criterion(buckets)
        root = self.Node(X, y, buckets, np.argmax(buckets), y.size, impurity)

        if self.max_depth and depth < self.max_depth:
            idx, threshold = self._split_least_impure(root, indices)
            if idx is not None:
                root.thres = threshold
                root.feature_index = idx
                mask = X[:, idx] < threshold
                X_left, X_right, y_left, y_right = X[mask], X[~mask], y[mask], y[~mask]
                left_indices = indices[mask]
                right_indices = indices[~mask]
                root.left = self._make_tree(X_left, y_left, depth + 1, left_indices)
                root.right = self._make_tree(X_right, y_right, depth + 1, right_indices)

        return root

    def _split_least_impure(self, root, indices):
        '''
            Splits x and y into 2 subsets each such that the sum of the impurities of the left and right subsets is minimised
            Returns tuple containing the index of the feature and threhsold to be used for partitioning
        '''
        X, y = root.X, root.y
        m,n = X.shape

        if m <= 1 or m < self.min_samples_split:
            return None, None
           
        self.weighted_n_current = m
        if self.sample_weight is not None:
            self.weighted_n_current = (np.take(self.sample_weight, indices)).sum()

        counts = root.buckets
        lowest_impurity = root.impurity
        lowest_left_impurity, lowest_right_impurity = None, None
        best_thres, best_fi = None, None
        w = 1.0

        for idx in range(n):
            thresholds, classes = zip(*sorted(zip(X[:, idx], y))) # m log m (times n for each feature_index)
            left_bucket = np.zeros(self.n_classes)
            right_bucket = counts.copy()
            weighted_n_left = 0

            for i in range(1,m):
                if self.sample_weight is not None:
                    w = self.sample_weight[indices[i]]

                prev_c = classes[i-1]
                left_bucket[prev_c] += w
                right_bucket[prev_c] -= w

                if thresholds[i] == thresholds[i-1]:
                    continue

                weighted_n_left += w
                weighted_n_right = self.weighted_n_current - weighted_n_left
                left_impurity = self.criterion(left_bucket, weighted_n_left)
                right_impurity = self.criterion(right_bucket, weighted_n_right)

                impurity = (weighted_n_left * left_impurity + weighted_n_right * right_impurity) / self.weighted_n_current

                if impurity < lowest_impurity:
                    lowest_impurity, lowest_left_impurity, lowest_right_impurity = impurity, left_impurity, right_impurity
                    best_n_left = weighted_n_left 
                    best_fi = idx
                    best_thres = (thresholds[i] + thresholds[i-1]) / 2


        if best_fi is not None and self.min_impurity_decrease > 0.0:
            best_n_right = self.weighted_n_current - best_n_left
            weighted_imp_dec = (self.weighted_n_current / self.weighted_n_samples) *  \
                               (root.impurity - (best_n_left / self.weighted_n_current * lowest_left_impurity) - \
                               (best_n_right / self.weighted_n_current * lowest_right_impurity))

            if weighted_imp_dec < self.min_impurity_decrease:
                return None, None # Don't split if improvement is below threshold

        return (best_fi, best_thres)

