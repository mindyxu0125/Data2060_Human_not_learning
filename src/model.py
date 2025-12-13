import numpy as np


def node_score_gini_from_counts(counts):
    '''
    Compute Gini impurity from class counts directly.
    '''
    n = int(counts.sum())
    if n == 0:
        return 0.0
    sum_sq = float((counts ** 2).sum())
    return 1.0 - sum_sq / (n * n)

def node_score_entropy_from_counts(counts):
    '''
    Compute Entropy impurity from class counts directly:
        H = log2(n) - (1/n) * sum_k c_k * log2(c_k)
    '''
    n = int(counts.sum())
    if n == 0:
        return 0.0

    mask = counts > 0
    if not np.any(mask):
        return 0.0

    c = counts[mask]
    return float(np.log2(n) - (c * np.log2(c)).sum() / n)

## Tree Structrue to mimic sklearn's DecisionTreeClassifier tree storage
class _Tree:
    """
    Simple tree structure storing node info in parallel lists, then
    converted to numpy arrays via finalize().
    """

    def __init__(self, n_classes):
        self.n_classes = n_classes

        self.children_left = []
        self.children_right = []
        self.feature = []
        self.threshold = []
        self.impurity = []
        self.n_node_samples = []
        self.value = []   # class counts per node (1D arrays length n_classes)

    def add_node(self, feature, threshold, impurity,
                 n_node_samples, counts, left=-1, right=-1):
        """
        Append a node and return its node_id (index).
        feature = -1 means leaf.
        """
        node_id = len(self.feature)
        self.children_left.append(int(left))
        self.children_right.append(int(right))
        self.feature.append(int(feature))
        self.threshold.append(float(threshold))
        self.impurity.append(float(impurity))
        self.n_node_samples.append(int(n_node_samples))
        self.value.append(np.asarray(counts, dtype=np.int64))
        return node_id

    def finalize(self):
        """
        Convert internal Python lists to numpy arrays.
        """
        self.children_left = np.asarray(self.children_left, dtype=np.int32)
        self.children_right = np.asarray(self.children_right, dtype=np.int32)
        self.feature = np.asarray(self.feature, dtype=np.int32)
        self.threshold = np.asarray(self.threshold, dtype=np.float64)
        self.impurity = np.asarray(self.impurity, dtype=np.float64)
        self.n_node_samples = np.asarray(self.n_node_samples, dtype=np.int64)
        self.value = np.stack(self.value, axis=0)  # shape (n_nodes, n_classes)
        self.node_count = self.feature.shape[0]




class CARTClassifier:
    """
    Numpy-only CART decision tree classifier with:
    - criterion: "gini" or "entropy"
    - max_depth: maximum depth of the tree (or None)
    - min_sample_split: minimum samples required to split
    - alpha: cost-complexity pruning parameter (0 = no pruning)
    - random_state: used ONLY to randomly permute feature order at each split
    """

    def __init__(self,
                 criterion="gini",
                 max_depth=None,
                 min_sample_split=2,
                 alpha=0.0,
                 random_state=None):
        
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.alpha = float(alpha)
        self.random_state = random_state

        if criterion == "gini":
            self._impurity_from_counts = node_score_gini_from_counts
        else:
            self._impurity_from_counts = node_score_entropy_from_counts

    def fit(self, X, y):
        """
        Build the full tree, then apply cost-complexity pruning with alpha.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)

        self.n_samples_, self.n_features_in_ = X.shape

        # Handle class labels (0..K-1 or need remap)
        classes = np.unique(y)
        if not np.array_equal(classes, np.arange(classes.size)):
            # remap to 0..K-1
            self._class_mapping_ = {c: i for i, c in enumerate(classes)}
            y_enc = np.array([self._class_mapping_[c] for c in y], dtype=np.int64)
            self.classes_ = classes
        else:
            self._class_mapping_ = None
            y_enc = y
            self.classes_ = classes

        self.n_classes_ = self.classes_.size

        # Build full (unpruned) tree
        self.tree_ = _Tree(n_classes=self.n_classes_)
        indices = np.arange(self.n_samples_, dtype=np.int64)
        self._build_tree(X, y_enc, indices, depth=0)
        self.tree_.finalize()

        # Cost-complexity pruning with given alpha
        if self.alpha > 0.0:
            self._prune_tree()

        return self

    # Internal helpers
    def _class_counts(self, y_subset):
        return np.bincount(y_subset, minlength=self.n_classes_)

    def _build_tree(self, X, y, indices, depth):
        """
        Recursively build the tree using greedy splitting.
        Returns node_id of the root of this subtree.
        """
        y_node = y[indices]
        counts = self._class_counts(y_node)
        n_node_samples = indices.size
        impurity = self._impurity_from_counts(counts)

        # Stopping criteria
        # 1) Pure node
        if impurity == 0.0:
            return self.tree_.add_node(
                feature=-1,
                threshold=-1.0,
                impurity=impurity,
                n_node_samples=n_node_samples,
                counts=counts,
                left=-1,
                right=-1,
            )

        # 2) Too few samples
        if n_node_samples < self.min_sample_split:
            return self.tree_.add_node(
                feature=-1,
                threshold=-1.0,
                impurity=impurity,
                n_node_samples=n_node_samples,
                counts=counts,
                left=-1,
                right=-1,
            )

        # 3) Depth limit
        if self.max_depth is not None and depth >= self.max_depth:
            return self.tree_.add_node(
                feature=-1,
                threshold=-1.0,
                impurity=impurity,
                n_node_samples=n_node_samples,
                counts=counts,
                left=-1,
                right=-1,
            )

        # Find best split
        best_feature = None
        best_threshold = None
        best_loss = np.inf

        # Mimic Sklearn's Randomness: randomly permute feature order at each split\
        np.random.RandomState(self.random_state)
        feature_indices = np.random.permutation(self.n_features_in_)

        for j in feature_indices:
            x_j = X[indices, j]
            uniq = np.unique(x_j)
            if uniq.size <= 1:
                continue

            thresholds = (uniq[:-1] + uniq[1:]) * 0.5

            for t in thresholds:
                left_mask = x_j <= t
                right_mask = ~left_mask
                if not left_mask.any() or not right_mask.any():
                    continue

                y_left = y_node[left_mask]
                y_right = y_node[right_mask]

                counts_left = self._class_counts(y_left)
                counts_right = self._class_counts(y_right)

                nL = counts_left.sum()
                nR = counts_right.sum()

                imp_left = self._impurity_from_counts(counts_left)
                imp_right = self._impurity_from_counts(counts_right)

                # Weighted impurity * sample counts (no need to divide by N)
                loss = nL * imp_left + nR * imp_right

                if loss <= best_loss:
                    best_loss = loss
                    best_feature = j
                    best_threshold = t

        # 4) No valid split found -> leaf
        if best_feature is None:
            return self.tree_.add_node(
                feature=-1,
                threshold=-1.0,
                impurity=impurity,
                n_node_samples=n_node_samples,
                counts=counts,
                left=-1,
                right=-1,
            )

        # Create internal node
        node_id = self.tree_.add_node(
            feature=best_feature,
            threshold=best_threshold,
            impurity=impurity,
            n_node_samples=n_node_samples,
            counts=counts,
            left=-1,
            right=-1,
        )

        # Partition samples
        x_best = X[indices, best_feature]
        left_mask = x_best <= best_threshold
        right_mask = ~left_mask
        idx_left = indices[left_mask]
        idx_right = indices[right_mask]

        # Recursively build children
        left_child = self._build_tree(X, y, idx_left, depth + 1)
        right_child = self._build_tree(X, y, idx_right, depth + 1)

        # Patch children pointers
        self.tree_.children_left[node_id] = left_child
        self.tree_.children_right[node_id] = right_child

        return node_id

    # Cost-complexity pruning
    def _compute_subtree_stats(self, node_id):
        """
        Compute:
            R_subtree = sum impurity(leaf) * n_samples(leaf)
            n_leaves  = number of leaves
        for the subtree rooted at node_id.
        """
        left = self.tree_.children_left[node_id]
        right = self.tree_.children_right[node_id]

        if left == -1 and right == -1:
            R = self.tree_.impurity[node_id] * self.tree_.n_node_samples[node_id]
            return R, 1

        R_l, L_l = self._compute_subtree_stats(left)
        R_r, L_r = self._compute_subtree_stats(right)
        return R_l + R_r, L_l + L_r

    def _prune_tree(self):
        """
        Apply cost-complexity pruning with the given alpha.
        Greedy weakest-link strategy:
            repeatedly prune the node t with smallest g(t)
            as long as g(t) <= alpha.
        """
        alpha = self.alpha
        if alpha <= 0.0:
            return

        while True:
            n_nodes = self.tree_.node_count
            R_subtree = np.zeros(n_nodes, dtype=np.float64)
            n_leaves = np.zeros(n_nodes, dtype=np.int64)

            def dfs(node_id):
                left = self.tree_.children_left[node_id]
                right = self.tree_.children_right[node_id]
                if left == -1 and right == -1:
                    R = self.tree_.impurity[node_id] * self.tree_.n_node_samples[node_id]
                    R_subtree[node_id] = R
                    n_leaves[node_id] = 1
                    return R, 1
                R_l, L_l = dfs(left)
                R_r, L_r = dfs(right)
                R_subtree[node_id] = R_l + R_r
                n_leaves[node_id] = L_l + L_r
                return R_l + R_r, L_l + L_r

            dfs(0)

            g = np.full(n_nodes, np.inf, dtype=np.float64)
            for node_id in range(n_nodes):
                left = self.tree_.children_left[node_id]
                right = self.tree_.children_right[node_id]
                if left == -1 and right == -1:
                    continue  # leaf
                if n_leaves[node_id] <= 1:
                    continue

                R_leaf = self.tree_.impurity[node_id] * self.tree_.n_node_samples[node_id]
                R_T = R_subtree[node_id]
                denom = n_leaves[node_id] - 1
                if denom <= 0:
                    continue

                g[node_id] = (R_leaf - R_T) / denom

            min_g = g.min()
            if (not np.isfinite(min_g)) or (min_g > alpha):
                break

            node_to_prune = int(np.argmin(g))
            self.tree_.children_left[node_to_prune] = -1
            self.tree_.children_right[node_to_prune] = -1

    # Prediction
    def _predict_one_proba(self, x):
        """
        Traverse the tree for a single sample x and return class probabilities.
        """
        node = 0
        while True:
            feature = self.tree_.feature[node]
            if feature == -1:
                counts = self.tree_.value[node]
                total = counts.sum()
                if total == 0:
                    return np.ones(self.n_classes_) / self.n_classes_
                return counts / total

            thr = self.tree_.threshold[node]
            if x[feature] <= thr:
                node = self.tree_.children_left[node]
            else:
                node = self.tree_.children_right[node]

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        n_samples = X.shape[0]
        proba = np.zeros((n_samples, self.n_classes_), dtype=np.float64)
        for i in range(n_samples):
            proba[i] = self._predict_one_proba(X[i])
        return proba

    def predict(self, X):
        proba = self.predict_proba(X)
        class_indices = np.argmax(proba, axis=1)
        return self.classes_[class_indices]