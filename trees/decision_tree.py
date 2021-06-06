from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

def gini_impurity(buckets):
    total = buckets.sum()
    if total == 0: return 1
    return 1 - np.square(buckets / total).sum()

class NotFittedError(Exception):
    def __init__(self, msg):
        super.__init__(msg)
    
class DecisionTree:
    class Node:
        def __init__(self, X, y, buckets, gini):
            self.X = X
            self.y = y
            self.buckets = buckets # 1-dimensional np.ndarray with #num_classes counts for each class
            self.n_samples = np.sum(self.buckets) 
            self.gini = gini
            self.left = None
            self.right = None
            self.feature_index = None
            self.thres = None

        @property
        def proba_(self):
            '''
                Returns the class probabilities for this node
            '''
            return self.buckets / self.n_samples

        def is_leaf(self):
            return self.left == None and self.right == None


    def __init__(self, max_depth):
        # self.criterion = criterion
        # self.splitter = splitter
        self.max_depth = max_depth
        self.prev_gini_cost = np.inf

    def fit(self, X, y):
        self.num_classes = len(np.unique(y))
        root_buckets = self.make_buckets(y)
        root_gini = gini_impurity(root_buckets)
        root = self.Node(X, y, root_buckets, root_gini)
        self.tree = self.make_tree(root, 0)

    def predict(self, X):
        if self.tree is None: 
            raise NotFittedError("Model not fitted yet. Call'fit' on the data before invoking 'predict'")
        
        n_samples = X.shape[0]
        preds = np.zeros((n_samples, 1))
        for i in range(n_samples):
            cur_node = self.tree
            while not cur_node.is_leaf():
                if X[i, cur_node.feature_index] < cur_node.thres:
                    cur_node = cur_node.left
                else:
                    cur_node = cur_node.right
            preds[i] = np.argmax(cur_node.proba_)
        return preds

    def make_tree(self, root, depth):
        '''
            Constructs decision tree recursively and stores it in self.tree
        '''
        if root is None:
            return root
        if depth == self.max_depth:
            return root

        l_node, r_node = self.split_least_gini(root)
        root.left = self.make_tree(l_node, depth + 1)
        root.right = self.make_tree(r_node, depth + 1)
        return root

    def make_buckets(self, arr):
        '''
            Takes an np.ndarray and turns it into discrete buckets with no.of occurrences of each class in each one
        '''
        _, counts = np.unique(arr, return_counts = True)
        buckets = np.resize(counts, self.num_classes)
        return buckets

    def split_least_gini(self, root):
        '''
            Splits x and y into 2 subsets each such that the sum of the Gini impurity of the left and right subsets is minimised,
            and returns the tuple containing instances of the Node class for the left and right subsets respectively.
        '''
        X, y = root.X, root.y
        m,n = X.shape
        best_gini = np.inf
        best_thres = -1
        best_fi = -1
        best_left_bucket, best_right_bucket = None, None

        for feature_index in range(n):
            X_sorted_tuple, y_sorted_tuple = zip(*sorted(zip(X[:, feature_index], y))) # m log m (times n for each feature_index)
            X_sorted, y_sorted= np.array(X_sorted_tuple), np.array(y_sorted_tuple)
            m_left = -1
            for i, threshold in enumerate(X_sorted):
                if i == 0:
                    # we have not initialised them yet
                    mask = X_sorted < threshold
                    y_left = y_sorted[mask]
                    m_left = len(y_left)
                    left_bucket = self.make_buckets(y_left)
                    y_right = y_sorted[~mask]
                    right_bucket = self.make_buckets(y_right)
                else:
                    # do incremental updates 
                    if threshold == X_sorted[i-1]:
                        continue
                    previous_class = y_sorted[i-1]
                    left_bucket[previous_class] += 1
                    right_bucket[previous_class] -= 1
                    m_left += 1

                gini_l = gini_impurity(left_bucket)
                gini_r = gini_impurity(right_bucket)
                gini_cost = (m_left / m)  * gini_l + ((m-m_left)/m) * gini_r

                if gini_cost < best_gini:
                    best_gini_cost, best_gini_l, best_gini_r = gini_cost, gini_l, gini_r
                    best_thres = threshold
                    best_fi = feature_index
                    best_left_bucket = left_bucket
                    best_right_bucket = right_bucket

        # First check if we have an improvement in cost from parent
        if best_gini_cost >= self.prev_gini_cost:
            return None, None;

        # Set new gini cost
        self.prev_gini_cost = best_gini_cost

        # Found the best values for the root
        root.feature_index = best_fi
        root.thres = best_thres

        # Now return the child nodes
        best_mask = X[:, best_fi] < best_thres
        return self.Node(X[best_mask], y[best_mask], best_left_bucket, best_gini_l), \
               self.Node(X[~best_mask], y[~best_mask], best_right_bucket, best_gini_r)

