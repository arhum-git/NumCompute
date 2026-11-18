"""NumPy decision-tree classifier with deterministic, buffered streaming retrains.

The tree is fitted from scratch whenever its buffer reaches ``buffer_size``.
It does not mutate existing splits per observation (it is not a Hoeffding tree).
"""

from dataclasses import dataclass
import numpy as np

from ._validation import numeric_2d, labels_1d, check_features, validate_positive_int


@dataclass
class _Node:
    """An internal split or a leaf with a deterministic majority-class index."""

    prediction: int
    distribution: object = None
    feature: int = -1
    threshold: float = 0.0
    missing_left: bool = True
    left: object = None
    right: object = None


class DecisionTreeClassifier:
    """CART-style classifier using Gini/entropy and periodic buffer retraining.

    Parameters
    ----------
    max_depth : int or None
        Maximum number of split levels, or no limit.
    min_samples_split : int
        Minimum training observations required to consider splitting a node.
    max_features : None, int, float, 'sqrt' or 'log2'
        Number of randomly selected features checked at each split.
    buffer_size : int
        Retrain after at least this many newly buffered observations. All
        observations in that buffer (including any overshoot) are consumed.
    criterion : {'gini', 'entropy'}
        Impurity measure for choosing a split.
    random_state : int or None
        Seed for feature subsampling.

    Notes
    -----
    NaNs are permitted in X. Each split learns whether missing values route
    left or right; ties favour left. Inf is rejected. Labels are non-missing.
    Ties in class frequency select the first class returned by np.unique.
    Fitting costs approximately O(d * n log n) at each visited node when all
    features are inspected; prediction costs O(n * tree_depth) and its result
    requires O(n) memory. The buffer uses O(buffer_size * (d + 1)) storage.
    """

    def __init__(self, max_depth=5, min_samples_split=2, max_features=None,
                 buffer_size=500, criterion="gini", random_state=42):
        if max_depth is not None:
            validate_positive_int(max_depth, "max_depth")
        validate_positive_int(min_samples_split, "min_samples_split")
        validate_positive_int(buffer_size, "buffer_size")
        if criterion not in ("gini", "entropy"):
            raise ValueError("criterion must be 'gini' or 'entropy'")
        if max_features is not None:
            if isinstance(max_features, str):
                if max_features not in ("sqrt", "log2"):
                    raise ValueError("max_features must be 'sqrt' or 'log2'")
            elif isinstance(max_features, bool):
                raise ValueError("max_features cannot be boolean")
            elif isinstance(max_features, (int, np.integer)):
                validate_positive_int(max_features, "max_features")
            elif isinstance(max_features, (float, np.floating)):
                if not 0 < max_features <= 1:
                    raise ValueError("float max_features must be in (0, 1]")
            else:
                raise ValueError("Invalid max_features setting")
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.max_features = max_features
        self.buffer_size = int(buffer_size)
        self.criterion = criterion
        self.random_state = random_state
        self.root_ = None
        self.classes_ = None
        self.n_features_in_ = None
        self._buffer_X = []
        self._buffer_y = []
        self.n_buffered_samples_ = 0
        self.n_retrains_ = 0
        self._rng = np.random.default_rng(random_state)

    def _impurity(self, counts):
        """Vectorised impurity for a row or a matrix of class count vectors."""
        counts = np.asarray(counts, dtype=float)
        total = counts.sum(axis=-1, keepdims=True)
        p = np.divide(counts, total, out=np.zeros_like(counts), where=total > 0)
        if self.criterion == "gini":
            return 1.0 - np.sum(p * p, axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(p > 0, p * np.log2(np.where(p > 0, p, 1)), 0.0)
        return -np.sum(terms, axis=-1)

    def _selected_features(self, n_features):
        k = self.max_features
        if k is None:
            return np.arange(n_features)
        if k == "sqrt":
            k = max(1, int(np.sqrt(n_features)))
        elif k == "log2":
            k = max(1, int(np.log2(n_features)))
        elif isinstance(k, (float, np.floating)):
            k = max(1, int(np.ceil(k * n_features)))
        if k > n_features:
            raise ValueError("max_features cannot exceed the input feature count")
        if k == n_features:
            return np.arange(n_features)
        return np.sort(self._rng.choice(n_features, size=k, replace=False))

    def _best_split(self, X, y_idx, parent_impurity):
        """Search every allowable threshold, with vectorised prefix counts.

        NaNs choose the better child (weighted impurity), deterministically.
        Sorting one feature is O(n log n), and cumulative counts are O(n*k)
        for k classes. A Python loop over *features*, not over observations,
        keeps the implementation easy to audit.
        """
        n, d = X.shape
        k = len(self.classes_)
        total_counts = np.bincount(y_idx, minlength=k)
        best = None
        best_gain = 1e-12  # avoid numerically insignificant splits
        for feature in self._selected_features(d):
            values = X[:, feature]
            finite = ~np.isnan(values)
            ids = np.flatnonzero(finite)
            if len(ids) < 2:
                continue
            sorted_ids = ids[np.argsort(values[ids], kind="stable")]
            sorted_values = values[sorted_ids]
            positions = np.flatnonzero(sorted_values[1:] > sorted_values[:-1])
            if len(positions) == 0:
                continue
            codes = y_idx[sorted_ids]
            cumulative = np.cumsum(np.eye(k, dtype=np.int64)[codes], axis=0)
            left_base = cumulative[positions]
            present_counts = cumulative[-1]
            missing_counts = total_counts - present_counts
            right_base = present_counts - left_base
            n_missing = n - len(ids)

            # Evaluate NaN-left and NaN-right for every threshold at once.
            left_l = left_base + missing_counts
            right_l = right_base
            left_r = left_base
            right_r = right_base + missing_counts
            left_n_l = (positions + 1) + n_missing
            right_n_l = n - left_n_l
            left_n_r = positions + 1
            right_n_r = n - left_n_r
            weight_l = (left_n_l * self._impurity(left_l) +
                        right_n_l * self._impurity(right_l)) / n
            weight_r = (left_n_r * self._impurity(left_r) +
                        right_n_r * self._impurity(right_r)) / n
            use_left = weight_l <= weight_r
            best_weight = np.minimum(weight_l, weight_r)
            best_pos = int(np.argmin(best_weight))
            gain = float(parent_impurity - best_weight[best_pos])
            if gain > best_gain:
                left_val = sorted_values[positions[best_pos]]
                right_val = sorted_values[positions[best_pos] + 1]
                # This formula limits cancellation/overflow in the midpoint.
                threshold = float((left_val / 2.0 + right_val / 2.0)
                                  if np.signbit(left_val) != np.signbit(right_val)
                                  else (left_val + (right_val - left_val) / 2.0))
                best = (int(feature), threshold, bool(use_left[best_pos]))
                best_gain = gain
        return best

    def _build(self, X, y_idx, depth):
        counts = np.bincount(y_idx, minlength=len(self.classes_))
        node = _Node(prediction=int(np.argmax(counts)),
                     distribution=counts.astype(float) / counts.sum())
        pure = np.count_nonzero(counts) == 1
        if (pure or len(y_idx) < self.min_samples_split or
                (self.max_depth is not None and depth >= self.max_depth)):
            return node
        split = self._best_split(X, y_idx, float(self._impurity(counts)))
        if split is None:
            return node
        feature, threshold, missing_left = split
        values = X[:, feature]
        nan = np.isnan(values)
        left = (values <= threshold) | (nan if missing_left else False)
        if not missing_left:
            left = (values <= threshold) & ~nan
        if not left.any() or left.all():
            return node
        node.feature = feature
        node.threshold = threshold
        node.missing_left = missing_left
        node.left = self._build(X[left], y_idx[left], depth + 1)
        node.right = self._build(X[~left], y_idx[~left], depth + 1)
        return node

    def fit(self, X, y):
        """Train a fresh tree on all (n,d) samples; clear pending stream data.

        Returns self. Raises ValueError for invalid shapes, labels, or values.
        Time is at most O(d*n^2*log(n)) for unbalanced recursion; tree
        storage is O(number of nodes), plus sorting/count arrays per node.
        """
        X = numeric_2d(X)
        y = labels_1d(y, X.shape[0])
        self.classes_, codes = np.unique(y, return_inverse=True)
        self.n_features_in_ = X.shape[1]
        self.root_ = self._build(X, codes, 0)
        self._buffer_X.clear()
        self._buffer_y.clear()
        self.n_buffered_samples_ = 0
        self.n_retrains_ += 1
        return self

    def partial_fit(self, X_chunk, y_chunk):
        """Append a chunk; retrain on its buffer when at least buffer_size.

        Unconsumed samples are retained between calls. A chunk may overshoot
        the threshold; in that case all pending samples train a *single* tree,
        then the buffer is cleared. Prediction keeps the last fitted tree.
        """
        X = numeric_2d(X_chunk)
        y = labels_1d(y_chunk, len(X))
        if self.n_features_in_ is not None:
            check_features(X, self.n_features_in_)
        elif self._buffer_X:
            check_features(X, self._buffer_X[0].shape[1])
        self._buffer_X.append(X.copy())
        self._buffer_y.append(y.copy())
        self.n_buffered_samples_ += len(X)
        if self.n_buffered_samples_ >= self.buffer_size:
            self.fit(np.concatenate(self._buffer_X), np.concatenate(self._buffer_y))
        return self

    def predict(self, X):
        """Predict shape (n,) labels using the latest completed retrain.

        Raises RuntimeError before the first successful fit/retrain.
        Missing values follow their learned routes. Prediction traverses
        at most O(n * tree_depth) nodes and allocates O(n) output storage.
        """
        if self.root_ is None:
            raise RuntimeError("Decision tree has not been trained yet")
        X = numeric_2d(X, empty=True)
        check_features(X, self.n_features_in_)
        pred = np.empty(len(X), dtype=self.classes_.dtype)
        if not len(X):
            return pred
        # Partition arrays of rows by node; no loop over individual samples.
        pending = [(self.root_, np.arange(len(X), dtype=int))]
        while pending:
            node, indices = pending.pop()
            if node.feature < 0:
                pred[indices] = self.classes_[node.prediction]
                continue
            values = X[indices, node.feature]
            valid = ~np.isnan(values)
            left = (valid & (values <= node.threshold)) | (~valid & node.missing_left)
            if np.any(left):
                pending.append((node.left, indices[left]))
            if np.any(~left):
                pending.append((node.right, indices[~left]))
        return pred

    def predict_proba(self, X):
        """Return class probabilities shape (n_samples, n_classes) from leaf frequencies.

        Probabilities are empirical training label frequencies in the reached
        leaf, not calibration estimates. NaNs follow learned missing routes.
        """
        if self.root_ is None:
            raise RuntimeError("Decision tree has not been trained yet")
        X = numeric_2d(X, empty=True)
        check_features(X, self.n_features_in_)
        result = np.empty((len(X), len(self.classes_)), dtype=float)
        pending = [(self.root_, np.arange(len(X), dtype=int))]
        while pending:
            node, ids = pending.pop()
            if node.feature < 0:
                result[ids] = node.distribution
                continue
            value = X[ids, node.feature]
            left = (np.isfinite(value) & (value <= node.threshold)) | (np.isnan(value) & node.missing_left)
            if left.any():
                pending.append((node.left, ids[left]))
            if (~left).any():
                pending.append((node.right, ids[~left]))
        return result

    @property
    def buffer_memory_bytes(self):
        """Bytes of NumPy arrays currently held in the pending buffer."""
        return sum(a.nbytes for a in self._buffer_X + self._buffer_y)
