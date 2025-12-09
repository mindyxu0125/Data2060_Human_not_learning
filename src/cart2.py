import numpy as np

def node_score_gini_from_counts(counts):
    '''
    Compute Gini impurity from class counts directly.

    Parameters
    ----------
    counts : 1D numpy array of ints
        counts[k] = number of samples of class k in the node.

    Returns
    -------
    float
        Gini impurity G = 1 - sum_k p_k^2,
        but computed in a counts-based way:
            G = 1 - sum_k (counts[k]^2) / n^2
    '''
    n = int(counts.sum())
    if n == 0:
        return 0.0
    sum_sq = float((counts.astype(np.float64) ** 2).sum())
    return 1.0 - sum_sq / (n * n)


def node_score_entropy_from_counts(counts):
    '''
    Compute Entropy impurity from class counts directly.

    Parameters
    ----------
    counts : 1D numpy array of ints
        counts[k] = number of samples of class k in the node.

    Returns
    -------
    float
        Entropy H = - sum_k p_k log_2(p_k),
        but computed via counts:

            Let n = sum_k counts[k].
            H = log2(n) - (1/n) * sum_k counts[k] * log2(counts[k]).

        For counts[k] = 0, the term counts[k]*log2(counts[k]) is taken as 0.
    '''
    n = int(counts.sum())
    if n == 0:
        return 0.0

    counts = counts.astype(np.float64)
    mask = counts > 0
    if not np.any(mask):
        return 0.0

    c = counts[mask]
    return (np.log2(n) - (c * np.log2(c)).sum() / n)


def _class_counts(y, n_classes):
    '''
    Count how many examples of each class appear in y.
    '''
    return np.bincount(y, minlength=n_classes)


def _to_probs(counts):
    '''
    Convert class counts to probabilities (for prediction at leaves).
    '''
    total = counts.sum()
    if total == 0:
        return np.zeros_like(counts, dtype=float)
    return counts.astype(float) / float(total)


class Node:
    '''
    Helper structure representing a single node in the CART tree.
    '''

    def __init__(self, depth, class_counts):
        self.depth = depth
        self.is_leaf = True
        self.feature_index = None
        self.threshold = None
        self.left = None
        self.right = None

        self.class_counts = class_counts.astype(int)
        self.n_samples = int(class_counts.sum())

        self.proba = _to_probs(self.class_counts)
        self.prediction = int(np.argmax(self.proba))


class DecisionTreeCART:
    '''
    CART (Classification and Regression Tree) classifier implemented from scratch.

    This version:
      - For each feature at a node, sorts samples by that feature and scans all
        adjacent distinct-value positions as candidate thresholds.
      - Computes Gini / Entropy impurity directly from class counts.
    '''

    def __init__(self,
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 impurity='gini',
                 random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = max(min_samples_split, 2)
        self.min_samples_leaf = max(min_samples_leaf, 1)
        self.impurity_name = impurity
        self.random_state = random_state
        self.n_classes_ = None
        self.n_features_ = None
        self.root_ = None

        self.rng_ = np.random.RandomState(random_state)

    def fit(self, X, y):
        '''
        Train the CART classifier on labeled data.
        '''
        X = np.asarray(X, dtype=np.float64, order='C')
        y = np.asarray(y, dtype=np.intp)

        n_samples, n_features = X.shape
        self.n_features_ = n_features
        self.n_classes_ = int(np.max(y)) + 1

        indices = np.arange(n_samples)
        self.root_ = self._build_tree(X, y, indices, depth=0)
        return self

    def predict(self, X):
        '''
        Predict class labels for a matrix of input samples.
        '''
        X = np.asarray(X, dtype=np.float64)
        preds = [self._predict_one(x, self.root_) for x in X]
        return np.array(preds, dtype=int)

    def predict_proba(self, X):
        '''
        Predict class probabilities for each sample.
        '''
        X = np.asarray(X, dtype=np.float64)
        probas = [self._predict_proba_one(x, self.root_) for x in X]
        return np.vstack(probas)

    def accuracy(self, X, y):
        '''
        Compute classification accuracy on a dataset.
        '''
        y = np.asarray(y, dtype=int)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def loss(self, X, y):
        '''
        Misclassification loss = 1 - accuracy
        '''
        return 1.0 - self.accuracy(X, y)

    # --------------------------------------------------------
    # Tree construction
    # --------------------------------------------------------

    def _build_tree(self, X, y, indices, depth):
        '''
        Recursively grow the tree.
        '''
        y_node = y[indices]
        counts = _class_counts(y_node, self.n_classes_)
        node = Node(depth=depth, class_counts=counts)

        if self._is_terminal(node, indices, y_node):
            return node

        best_feature, best_threshold, best_gain = self._best_split(X, y_node, indices)

        if best_feature is None or best_gain <= 0.0:
            return node

        node.is_leaf = False
        node.feature_index = best_feature
        node.threshold = float(best_threshold)

        feature_values = X[indices, best_feature]
        left_mask = feature_values <= node.threshold
        right_mask = ~left_mask

        left_indices = indices[left_mask]
        right_indices = indices[right_mask]

        node.left = self._build_tree(X, y, left_indices, depth + 1)
        node.right = self._build_tree(X, y, right_indices, depth + 1)

        return node

    def _is_terminal(self, node, indices, y_node):
        '''
        Check whether a node should stop splitting.
        '''
        n_samples = indices.size

        if n_samples == 0:
            return True

        if np.unique(y_node).size == 1:
            return True

        if self.max_depth is not None and node.depth >= self.max_depth:
            return True

        if n_samples < self.min_samples_split:
            return True

        return False

    def _best_split(self, X, y_node, indices):
        '''
        Find the best (feature, threshold) split using sorted order and
        cumulative class counts.

        For each feature:
          - sort the samples in this node by X[:, f]
          - scan i = 1..n-1; whenever fv_sorted[i] != fv_sorted[i-1],
            consider a threshold between them
          - track left/right class counts incrementally and compute impurity
        '''
        n_node_samples = indices.size
        if n_node_samples == 0:
            return None, None, 0.0

        parent_counts = _class_counts(y_node, self.n_classes_)
        parent_impurity = self._impurity(parent_counts)

        best_gain = -np.inf
        best_feature = -1
        best_threshold = None

        if self.random_state is not None:
            feature_indices = self.rng_.permutation(self.n_features_)
        else:
            feature_indices = np.arange(self.n_features_)

        for f in feature_indices:
            feature_values = X[indices, f]

            # All values equal -> no split possible
            if np.all(feature_values == feature_values[0]):
                continue

            # Stable sort by feature value
            order = np.argsort(feature_values, kind="mergesort")
            fv_sorted = feature_values[order]
            y_sorted = y_node[order]

            left_counts = np.zeros(self.n_classes_, dtype=int)
            right_counts = parent_counts.copy()

            for i in range(1, n_node_samples):
                c = y_sorted[i - 1]
                left_counts[c] += 1
                right_counts[c] -= 1

                # Cannot split between equal feature values
                if fv_sorted[i] == fv_sorted[i - 1]:
                    continue

                n_left = i
                n_right = n_node_samples - i

                if (n_left < self.min_samples_leaf) or (n_right < self.min_samples_leaf):
                    continue

                impur_left = self._impurity(left_counts)
                impur_right = self._impurity(right_counts)

                gain = parent_impurity - (
                    (float(n_left) / float(n_node_samples)) * impur_left +
                    (float(n_right) / float(n_node_samples)) * impur_right
                )

                if gain > best_gain:
                    best_gain = gain
                    best_threshold = 0.5 * (fv_sorted[i] + fv_sorted[i - 1])
                    best_feature = f

        if best_feature == -1:
            return None, None, 0.0

        return best_feature, best_threshold, best_gain

    def _impurity(self, counts):
        '''
        Compute impurity from class counts, according to chosen criterion.
        '''
        if self.impurity_name == 'gini':
            return node_score_gini_from_counts(counts)
        elif self.impurity_name == 'entropy':
            return node_score_entropy_from_counts(counts)
        else:
            raise ValueError(
                f"Unknown impurity '{self.impurity_name}', use 'gini' or 'entropy'."
            )

    def _predict_one(self, x, node):
        '''
        Predict class label for a single sample by traversing the tree.
        '''
        while not node.is_leaf:
            f = node.feature_index
            t = node.threshold
            if x[f] <= t:
                node = node.left
            else:
                node = node.right
        return node.prediction

    def _predict_proba_one(self, x, node):
        '''
        Predict class probabilities for a single sample.
        '''
        while not node.is_leaf:
            f = node.feature_index
            t = node.threshold
            if x[f] <= t:
                node = node.left
            else:
                node = node.right
        return node.proba