"""Incremental descriptive statistics: pooled Welford moments and chunk histograms.

Exact quantiles over an unbounded stream cannot be computed with bounded memory
in general. This implementation stores history for exact quantiles by default;
set ``history_size`` to a finite number for bounded rolling quantiles.
"""

from collections import deque
import numpy as np
from ._validation import numeric_2d, check_features, validate_positive_int


class StreamingStats:
    """Update mean/variance, histogram and quantiles from arbitrary chunks.

    Parameters
    ----------
    bins : int
        Histogram bin count for each feature.
    histogram_range : tuple(float,float) or None
        If provided, bin counts accumulate online without history; observations
        outside the range are excluded. If omitted, histograms use retained
        history (and therefore may need O(N*d) memory).
    history_size : int or None
        Number of most recent rows stored for quantiles / dynamic histograms;
        None retains all observations. The moments always use *all* observations.

    ``update_stats(X_chunk)`` and ``update(X_chunk)`` are interchangeable.
    Mean and variance are population estimates ignoring NaNs. All-NaN columns
    have mean/variance NaN; min and max use NaN. Cost O(n*d) per chunk, plus
    O(history_size*d) for retained history and exact quantile calculations.
    """

    def __init__(self, bins=10, histogram_range=None, history_size=None):
        self.bins = validate_positive_int(bins, "bins")
        if histogram_range is not None:
            low, high = histogram_range
            if not np.isfinite([low, high]).all() or low >= high:
                raise ValueError("histogram_range requires finite low < high")
            self.histogram_range = (float(low), float(high))
        else:
            self.histogram_range = None
        self.history_size = (None if history_size is None else
                             validate_positive_int(history_size, "history_size"))
        self.reset()

    def reset(self):
        self.n_features_in_ = None
        self.count_ = None
        self.mean_ = None
        self._M2 = None
        self.min_ = None
        self.max_ = None
        self._history = deque()
        self._history_rows = 0
        self._hist_counts = None
        return self

    def update_stats(self, X_chunk):
        """Merge a numeric (n,d) chunk and return self, using O(n*d) arithmetic."""
        X = numeric_2d(X_chunk)
        if self.n_features_in_ is None:
            d = X.shape[1]
            self.n_features_in_ = d
            self.count_ = np.zeros(d, dtype=np.int64)
            self.mean_ = np.zeros(d)
            self._M2 = np.zeros(d)
            self.min_ = np.full(d, np.inf)
            self.max_ = np.full(d, -np.inf)
            if self.histogram_range is not None:
                self._hist_counts = np.zeros((d, self.bins), dtype=np.int64)
        check_features(X, self.n_features_in_)
        mask = ~np.isnan(X)
        n_b = mask.sum(axis=0)
        batch_mu = np.divide(np.sum(X, axis=0, where=mask, initial=0.0), n_b,
                             out=np.zeros(X.shape[1]), where=n_b > 0)
        dev = np.where(mask, X - batch_mu, 0.0)
        M2_b = np.einsum("ij,ij->j", dev, dev)
        n = self.count_ + n_b
        delta = batch_mu - self.mean_
        self.mean_ += delta * np.divide(n_b, n, out=np.zeros(X.shape[1]), where=n > 0)
        cross = np.divide(self.count_ * n_b, n, out=np.zeros(X.shape[1]), where=n > 0)
        self._M2 += M2_b + delta * delta * cross
        self.count_ = n
        self.min_ = np.minimum(self.min_, np.min(np.where(mask, X, np.inf), axis=0))
        self.max_ = np.maximum(self.max_, np.max(np.where(mask, X, -np.inf), axis=0))

        if self._hist_counts is not None:
            for j in range(X.shape[1]):
                counts, _ = np.histogram(X[mask[:, j], j], bins=self.bins,
                                         range=self.histogram_range)
                self._hist_counts[j] += counts

        self._history.append(X.copy())
        self._history_rows += len(X)
        if self.history_size is not None:
            # Bound retained observations while keeping the all-time moments.
            while self._history_rows > self.history_size:
                remove = min(self._history_rows - self.history_size, len(self._history[0]))
                if remove == len(self._history[0]):
                    self._history.popleft()
                else:
                    self._history[0] = self._history[0][remove:]
                self._history_rows -= remove
        return self

    update = update_stats
    partial_fit = update_stats

    def fit(self, X):
        """Reset and compute summaries from one batch."""
        return self.reset().update_stats(X)

    def _history_array(self):
        return np.concatenate(self._history) if self._history else np.empty((0, self.n_features_in_ or 0))

    def quantile(self, q=(0.25, 0.5, 0.75)):
        """Quantiles (shape len(q),d) from retained history; O(history*d)."""
        if self.n_features_in_ is None:
            raise RuntimeError("No data have been observed")
        q = np.asarray(q, dtype=float)
        if q.ndim == 0:
            q = q.reshape(1)
        if q.ndim != 1 or not np.isfinite(q).all() or np.any((q < 0) | (q > 1)):
            raise ValueError("q must contain quantiles between 0 and 1")
        H = self._history_array()
        answer = np.full((len(q), self.n_features_in_), np.nan)
        for j in range(self.n_features_in_):
            col = H[:, j]
            col = col[~np.isnan(col)]
            if len(col):
                answer[:, j] = np.quantile(col, q)
        return answer

    def histogram(self):
        """Return per-feature counts shape (d,bins) and common/individual edges."""
        if self.n_features_in_ is None:
            raise RuntimeError("No data have been observed")
        if self._hist_counts is not None:
            edges = np.linspace(*self.histogram_range, self.bins + 1)
            return self._hist_counts.copy(), np.tile(edges, (self.n_features_in_, 1))
        H = self._history_array()
        counts = np.zeros((self.n_features_in_, self.bins), dtype=np.int64)
        edges = np.zeros((self.n_features_in_, self.bins + 1))
        for j in range(self.n_features_in_):
            col = H[:, j]
            col = col[~np.isnan(col)]
            if len(col):
                counts[j], edges[j] = np.histogram(col, bins=self.bins)
            else:
                edges[j] = np.linspace(0.0, 1.0, self.bins + 1)
        return counts, edges

    def result(self):
        """Return copies of moments and quantiles; population variance is M2/n."""
        if self.n_features_in_ is None:
            raise RuntimeError("No data have been observed")
        mean = np.where(self.count_ > 0, self.mean_, np.nan)
        var = np.divide(self._M2, self.count_, out=np.full(len(self.count_), np.nan), where=self.count_ > 0)
        return {"count": self.count_.copy(), "mean": mean, "variance": var,
                "std": np.sqrt(np.maximum(var, 0)),
                "min": np.where(self.count_ > 0, self.min_, np.nan),
                "max": np.where(self.count_ > 0, self.max_, np.nan),
                "quantiles": self.quantile()}
