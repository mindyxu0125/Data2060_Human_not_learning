import numpy as np
from collections import Counter

def gini_impurity(y: np.ndarray) -> float:
    n = len(y)
    if n == 0:
        return 0.0
    counts = Counter(y)
    return 1.0 - sum((c / n) ** 2 for c in counts.values())

class Node:
    def __init__(
        self,
        depth = 0,
        index_split_on = -1,   
        threshold = None,
        isleaf = False,
        label=None,
        left = None,
        right = None
    ):
        self.depth = depth
        self.index_split_on = index_split_on
        self.threshold = threshold
        self.isleaf = isleaf
        self.label = label
        self.left = left
        self.right = right
        self.info = {}        

    def _set_info(self, gain, num_samples):
        self.info["gain"] = gain
        self.info["num_samples"] = num_samples

class DecisionTreeCART:
    """
    tree = DecisionTreeCART(max_depth=10, min_samples_split=5)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    tree.print_tree()

    Supports:
        - binary + multiclass
        - continuous features
        - categorical features (after one-hot encoding)
    """

    def __init__(self, max_depth=40, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.n_features = None
        self.majority_class = None

    def fit(self, X, y, X_val = None, y_val = None):
        """
        Build a CART decision tree.
        X: (N, d)
        y: (N,)
        """
        self.n_features = X.shape[1]

        # global majority class
        counts = Counter(y)
        self.majority_class = counts.most_common(1)[0][0]

        # build root
        self.root = Node(depth=0, label=self.majority_class)

        # recursive building
        indices = list(range(self.n_features))
        self._split_recurs(self.root, X, y, indices)

        # pruning if validation provided
        if X_val is not None and y_val is not None:
            self._prune_recurs(self.root, X_val, y_val)

    def predict(self, X):
        preds = []
        for row in X:
            preds.append(self._predict_recurs(self.root, row))
        return np.array(preds)

    def accuracy(self, X, y):
        return 1 - self.loss(X, y)

    def loss(self, X, y):
        y_pred = self.predict(X)
        return float(np.mean(y_pred != y))


    ##################

    def _predict_recurs(self, node, row):
        if node.isleaf or node.index_split_on < 0:
            return node.label

        j = node.index_split_on
        t = node.threshold

        if row[j] <= t:
            return self._predict_recurs(node.left, row)
        else:
            return self._predict_recurs(node.right, row)

    def _is_terminal(self, node, X, y, indices):
        if len(y) == 0:
            return True, self.majority_class

        counts = Counter(y)
        maj_class = counts.most_common(1)[0][0]

        if len(counts) == 1:
            return True, maj_class

        # reach max depth
        if node.depth >= self.max_depth:
            return True, maj_class

        # too few samples
        if len(y) < self.min_samples_split:
            return True, maj_class

        # no features to split
        if len(indices) == 0:
            return True, maj_class

        return False, maj_class

    def _split_recurs(self, node, X, y, indices):
        should_stop, label = self._is_terminal(node, X, y, indices)
        node.label = label
        node.isleaf = should_stop

        if should_stop:
            return
        
        best_gain = -float("inf")
        best_j, best_t = None, None

        for j in indices:
            gain_j, t_j = self._best_split_on_feature(X, y, j)
            if gain_j is not None and gain_j > best_gain:
                best_gain = gain_j
                best_j = j
                best_t = t_j

        if best_j is None:
            node.isleaf = True
            return

        node.index_split_on = best_j
        node.threshold = best_t
        node._set_info(best_gain, len(y))

        left_mask = X[:, best_j] <= best_t
        right_mask = ~left_mask

        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]

        if len(y_left) == 0 or len(y_right) == 0:
            node.isleaf = True
            return

        node.left = Node(depth=node.depth + 1, label=label)
        node.right = Node(depth=node.depth + 1, label=label)

        # CART can reuse features → do not remove best_j from indices
        self._split_recurs(node.left, X_left, y_left, indices)
        self._split_recurs(node.right, X_right, y_right, indices)

    def _best_split_on_feature(self, X, y, j):
        xs = X[:, j]
        ys = y

        if xs.max() == xs.min():
            return None, None

        idx_sorted = np.argsort(xs)
        xs_sorted = xs[idx_sorted]
        ys_sorted = ys[idx_sorted]

        n = len(xs_sorted)
        impurity_parent = gini_impurity(ys_sorted)

        candidates = []
        for k in range(1, n):
            if xs_sorted[k] != xs_sorted[k - 1]:
                t = 0.5 * (xs_sorted[k] + xs_sorted[k - 1])
                candidates.append(t)

        best_gain = -float("inf")
        best_t = None

        for t in candidates:
            left_mask = xs_sorted <= t
            right_mask = ~left_mask

            y_left = ys_sorted[left_mask]
            y_right = ys_sorted[right_mask]

            if len(y_left) == 0 or len(y_right) == 0:
                continue

            imp_left = gini_impurity(y_left)
            imp_right = gini_impurity(y_right)

            w_left = len(y_left) / n
            w_right = len(y_right) / n

            gain = impurity_parent - (w_left * imp_left + w_right * imp_right)

            if gain > best_gain:
                best_gain = gain
                best_t = t

        return best_gain, best_t

    # [TODO] Tree Pruning 
    # this is a simple implementation of post-pruning using validation set
    # need to be adjusted for cart specifics
    def _prune_recurs(self, node, X_val, y_val):

        if node is None or node.isleaf:
            return

        if node.left is not None:
            self._prune_recurs(node.left, X_val, y_val)
        if node.right is not None:
            self._prune_recurs(node.right, X_val, y_val)

        if node.left and node.right and node.left.isleaf and node.right.isleaf:

            loss_before = self.loss(X_val, y_val)

            backup = {
                "left": node.left,
                "right": node.right,
                "index_split_on": node.index_split_on,
                "threshold": node.threshold,
                "isleaf": node.isleaf,
                "label": node.label,
                "info": dict(node.info),
            }

            node.isleaf = True
            node.left = None
            node.right = None
            node.index_split_on = -1
            node.threshold = None
            node._set_info(0.0, backup["info"].get("num_samples", 0))

            loss_after = self.loss(X_val, y_val)

            if loss_after > loss_before:
                # rollback
                node.left = backup["left"]
                node.right = backup["right"]
                node.index_split_on = backup["index_split_on"]
                node.threshold = backup["threshold"]
                node.isleaf = backup["isleaf"]
                node.label = backup["label"]
                node.info = backup["info"]

    # ==================================================

    def print_tree(self):
        if self.root is None:
            print("Tree is empty")
            return

        print("--- CART TREE ---")
        def _print_subtree(node, indent=""):
            if node.isleaf:
                return indent + f"Leaf(label={node.label}, samples={node.info.get('num_samples','?')})"
            s = indent + f"[Feature {node.index_split_on} <= {node.threshold:.4f}] gain={node.info.get('gain',0):.4f}\n"
            left = _print_subtree(node.left, indent + "  ")
            right = _print_subtree(node.right, indent + "  ")
            return s + "\n" + left + "\n" + right

        print(_print_subtree(self.root))
        print("--- END ---")
