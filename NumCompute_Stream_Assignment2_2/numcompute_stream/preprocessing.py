"""Chunk-wise preprocessing for NumCompute-Stream.

Numeric transformers ignore missing entries while accumulating statistics and
preserve NaNs until an imputer handles them. Categorical metadata updates may
use Python dictionaries; numeric operations are vectorised with NumPy.
"""

import numpy as np
from ._validation import numeric_2d, categorical_2d, check_features


class StandardScaler:
    """Online population z-score scaler with pooled variance (O(n*d) per chunk).

    Parameters: with_mean, with_std -- booleans controlling centering/scaling.
    After partial_fit: ``mean_``, ``var_``, ``scale_``, ``count_`` arrays (d,).
    NaNs are ignored during fitting and preserved during transform. Columns
    with no observations use mean=0 and scale=1; constant columns use scale=1.
    Memory beyond input/output arrays: O(d). ``fit`` resets prior state.
    """

    def __init__(self, with_mean=True, with_std=True):
        self.with_mean = bool(with_mean)
        self.with_std = bool(with_std)
        self.reset()

    def reset(self):
        self.n_features_in_ = None
        self.count_ = None
        self.mean_ = None
        self.var_ = None
        self.scale_ = None
        self._M2 = None
        return self

    def partial_fit(self, X, y=None):
        """Merge each (n,d) chunk into running mean and population variance."""
        X = numeric_2d(X)
        if self.n_features_in_ is None:
            self.n_features_in_ = X.shape[1]
            self.count_ = np.zeros(X.shape[1], dtype=np.int64)
            self.mean_ = np.zeros(X.shape[1])
            self._M2 = np.zeros(X.shape[1])
        check_features(X, self.n_features_in_)
        mask = ~np.isnan(X)
        n_b = mask.sum(axis=0)
        sum_b = np.sum(X, axis=0, where=mask, initial=0.0)
        mu_b = np.divide(sum_b, n_b, out=np.zeros_like(sum_b), where=n_b > 0)
        diff = np.where(mask, X - mu_b, 0.0)
        M2_b = np.einsum("ij,ij->j", diff, diff)
        total = self.count_ + n_b
        delta = mu_b - self.mean_
        w = np.divide(n_b, total, out=np.zeros_like(mu_b), where=total > 0)
        self.mean_ += delta * w
        cross = np.divide(self.count_ * n_b, total, out=np.zeros_like(mu_b), where=total > 0)
        self._M2 += M2_b + delta * delta * cross
        self.count_ = total
        self.var_ = np.divide(self._M2, total, out=np.zeros_like(mu_b), where=total > 0)
        self.scale_ = np.sqrt(np.maximum(self.var_, 0.0))
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def fit(self, X, y=None):
        """Reset and learn statistics for the full (n,d) array; returns self."""
        return self.reset().partial_fit(X)

    def transform(self, X):
        """Return transformed float (n,d) matrix; NaNs preserved. O(n*d)."""
        X = numeric_2d(X, empty=True)
        check_features(X, self.n_features_in_)
        result = X.copy()
        if self.with_mean:
            result -= self.mean_
        if self.with_std:
            result /= self.scale_
        return result

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


class MinMaxScaler:
    """Online min-max scaler. Stores running extremes in O(d) memory.

    With ``feature_range=(low, high)``, constant features map to ``low``.
    Missing values are ignored during fitting and preserved by transform.
    ``fit`` resets historical extremes.
    """

    def __init__(self, feature_range=(0.0, 1.0)):
        if len(feature_range) != 2 or not np.all(np.isfinite(feature_range)) or feature_range[0] >= feature_range[1]:
            raise ValueError("feature_range must have finite low < high")
        self.feature_range = tuple(map(float, feature_range))
        self.reset()

    def reset(self):
        self.n_features_in_ = None
        self.data_min_ = None
        self.data_max_ = None
        return self

    def partial_fit(self, X, y=None):
        """Update running min/max from one chunk, ignoring NaNs; O(n*d)."""
        X = numeric_2d(X)
        if self.n_features_in_ is None:
            self.n_features_in_ = X.shape[1]
            self.data_min_ = np.full(X.shape[1], np.inf)
            self.data_max_ = np.full(X.shape[1], -np.inf)
        check_features(X, self.n_features_in_)
        valid = ~np.isnan(X)
        self.data_min_ = np.minimum(self.data_min_, np.min(np.where(valid, X, np.inf), axis=0))
        self.data_max_ = np.maximum(self.data_max_, np.max(np.where(valid, X, -np.inf), axis=0))
        return self

    def fit(self, X, y=None):
        return self.reset().partial_fit(X)

    def transform(self, X):
        """Return scaled (n,d) float array. Unobserved columns remain NaN."""
        X = numeric_2d(X, empty=True)
        check_features(X, self.n_features_in_)
        low, high = self.feature_range
        span = self.data_max_ - self.data_min_
        result = np.full_like(X, np.nan)
        valid = np.isfinite(span) & (span > 0)
        if valid.any():
            result[:, valid] = ((X[:, valid] - self.data_min_[valid]) /
                                span[valid] * (high - low) + low)
        const = np.isfinite(span) & (span == 0)
        if const.any():
            result[:, const] = np.where(np.isnan(X[:, const]), np.nan, low)
        return result

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


class SimpleImputer:
    """Incremental NaN imputation using mean, median or a constant.

    Exact streaming median requires retaining nonmissing observations for each
    feature (O(total seen samples*d) memory). Mean/constant use only O(d)
    summary memory. All-missing features use ``fill_value`` (default zero).
    """

    def __init__(self, strategy="mean", fill_value=0.0):
        if strategy not in ("mean", "median", "constant"):
            raise ValueError("strategy must be 'mean', 'median' or 'constant'")
        if not np.isfinite(fill_value):
            raise ValueError("fill_value must be finite")
        self.strategy = strategy
        self.fill_value = float(fill_value)
        self.reset()

    def reset(self):
        self.n_features_in_ = None
        self.count_ = None
        self._sums = None
        self._values = None
        self.statistics_ = None
        return self

    def partial_fit(self, X, y=None):
        """Update imputation summaries from a new chunk; return self."""
        X = numeric_2d(X)
        if self.n_features_in_ is None:
            self.n_features_in_ = X.shape[1]
            self.count_ = np.zeros(X.shape[1], dtype=np.int64)
            self._sums = np.zeros(X.shape[1])
            self._values = [[] for _ in range(X.shape[1])] if self.strategy == "median" else None
        check_features(X, self.n_features_in_)
        mask = ~np.isnan(X)
        self.count_ += mask.sum(axis=0)
        if self.strategy == "mean":
            self._sums += np.sum(X, axis=0, where=mask, initial=0.0)
            self.statistics_ = np.divide(self._sums, self.count_, out=np.full(X.shape[1], self.fill_value), where=self.count_ > 0)
        elif self.strategy == "median":
            for j in range(X.shape[1]):
                self._values[j].append(X[mask[:, j], j].copy())
            self.statistics_ = np.array([
                np.median(np.concatenate(v)) if any(part.size for part in v) else self.fill_value
                for v in self._values
            ])
        else:
            self.statistics_ = np.full(X.shape[1], self.fill_value)
        return self

    def fit(self, X, y=None):
        return self.reset().partial_fit(X)

    def transform(self, X):
        """Fill missing entries using current estimates; output shape (n,d)."""
        X = numeric_2d(X, empty=True)
        check_features(X, self.n_features_in_)
        return np.where(np.isnan(X), self.statistics_, X)

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


class OneHotEncoder:
    """Append categories found in each chunk; produce a dense encoded matrix.

    Each categorical feature owns an independent ordered category dictionary.
    New categories are appended (not re-sorted), preserving earlier column
    indices. Output feature count can grow after ``partial_fit``. Pipelines
    must retain a snapshot of the encoder paired with their fitted model.

    Parameters: handle_unknown={'error','ignore'}; unseen categories at predict
    time either raise or produce all-zero columns. NaN and None are forbidden.
    """

    def __init__(self, handle_unknown="error", sparse_output=False):
        if handle_unknown not in ("error", "ignore"):
            raise ValueError("handle_unknown must be 'error' or 'ignore'")
        if sparse_output:
            raise ValueError("Only dense output is supported (NumPy only)")
        self.handle_unknown = handle_unknown
        self.sparse_output = sparse_output
        self.reset()

    @staticmethod
    def _key(value):
        if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)):
            raise ValueError("Missing categories must be imputed before encoding")
        if isinstance(value, np.generic):
            value = value.item()
        try:
            hash(value)
        except TypeError as exc:
            raise ValueError("Category values must be hashable") from exc
        return value

    def reset(self):
        self.n_features_in_ = None
        self.categories_ = None
        self._maps = None
        self.n_output_features_ = 0
        return self

    def partial_fit(self, X, y=None):
        """Discover categories in new (n,d) input; metadata O(n*d)."""
        X = categorical_2d(X)
        if self.n_features_in_ is None:
            self.n_features_in_ = X.shape[1]
            self.categories_ = [[] for _ in range(X.shape[1])]
            self._maps = [{} for _ in range(X.shape[1])]
        check_features(X, self.n_features_in_)
        for j in range(X.shape[1]):
            for value in X[:, j]:
                key = self._key(value)
                if key not in self._maps[j]:
                    self._maps[j][key] = len(self.categories_[j])
                    self.categories_[j].append(key)
        self.n_output_features_ = sum(map(len, self.categories_))
        return self

    def fit(self, X, y=None):
        return self.reset().partial_fit(X)

    def transform(self, X):
        """Encode categorical (n,d) matrix; dense output O(n*sum(categories))."""
        X = categorical_2d(X, empty=True)
        check_features(X, self.n_features_in_)
        result = np.zeros((len(X), self.n_output_features_), dtype=float)
        offset = 0
        for j, mapping in enumerate(self._maps):
            # Missing categories are invalid even in handle_unknown='ignore'.
            for value in X[:, j]:
                self._key(value)
            # Equality is vectorised over samples and known categories.
            # A dense encoder already needs O(samples * categories) memory.
            categories = np.asarray(self.categories_[j], dtype=object)
            match = (X[:, j, None] == categories[None, :])
            found = np.any(match, axis=1)
            if self.handle_unknown == "error" and not found.all():
                raise ValueError(f"Unknown or missing category in column {j}")
            if found.any():
                valid = np.flatnonzero(found)
                indices = np.argmax(match, axis=1)
                result[valid, offset + indices[valid]] = 1.0
            offset += len(mapping)
        return result

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)
