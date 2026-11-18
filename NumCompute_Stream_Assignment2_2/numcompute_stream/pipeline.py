"""Streaming Pipeline with raw-data buffering and frozen prediction snapshots.

Preprocessors learn from every chunk. The final tree is refitted once enough
raw observations have arrived; the buffer is *transformed at retrain time* to
avoid mixing stale scaler parameters or categorical feature widths. Between
retraining events, predictions use a snapshot of the transformers paired with
the existing tree, so changing category vocabularies do not break prediction.
"""

from copy import deepcopy
import numpy as np

from ._validation import labels_1d, validate_positive_int


class Pipeline:
    """A named streaming chain of transformers ending with an estimator.

    Every intermediate stage must implement ``partial_fit`` and ``transform``;
    the last must implement ``fit``, ``partial_fit`` and ``predict``. Names must
    be unique nonempty strings. ``partial_fit(X_chunk,y_chunk)`` accumulates raw
    chunks and refits the final model after its buffer_size threshold. For a
    non-buffered estimator, each chunk is sent to estimator.partial_fit.
    """

    def __init__(self, steps):
        if not isinstance(steps, (list, tuple)) or not steps:
            raise ValueError("Pipeline requires at least one named step")
        names = []
        for entry in steps:
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                raise ValueError("Each pipeline step must be (name, component)")
            name, step = entry
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Pipeline step names must be nonempty strings")
            if name in names:
                raise ValueError(f"Duplicate pipeline step name: {name}")
            names.append(name)
        for name, step in steps[:-1]:
            for method in ("partial_fit", "transform", "fit"):
                if not callable(getattr(step, method, None)):
                    raise TypeError(f"Transformer '{name}' requires {method}()")
        name, final = steps[-1]
        for method in ("fit", "partial_fit", "predict"):
            if not callable(getattr(final, method, None)):
                raise TypeError(f"Final estimator '{name}' requires {method}()")
        self.steps = list(steps)
        self._predict_transformers = None
        self._pending_X = []
        self._pending_y = []
        self.n_buffered_samples_ = 0
        self.n_features_in_ = None

    @property
    def estimator(self):
        return self.steps[-1][1]

    def _validate_chunk(self, X, y=None):
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[0] < 1 or X.shape[1] < 1:
            raise ValueError("X must be a nonempty 2D array")
        if self.n_features_in_ is not None and self.n_features_in_ != X.shape[1]:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")
        if y is not None:
            y = labels_1d(y, len(X))
        return X, y

    def _live_transform(self, X):
        for name, transformer in self.steps[:-1]:
            X = transformer.transform(X)
            if np.asarray(X).ndim != 2:
                raise ValueError(f"Transformer '{name}' did not return a 2D array")
        return X

    def fit(self, X, y):
        """Fit preprocessing and tree from scratch; discard any pending buffer."""
        X, y = self._validate_chunk(X, y)
        self.n_features_in_ = X.shape[1]
        curr = X
        for name, transformer in self.steps[:-1]:
            transformer.fit(curr)
            curr = transformer.transform(curr)
        self.estimator.fit(curr, y)
        self._predict_transformers = deepcopy(self.steps[:-1])
        self._pending_X.clear()
        self._pending_y.clear()
        self.n_buffered_samples_ = 0
        return self

    def partial_fit(self, X_chunk, y_chunk):
        """Update preprocessing then periodically refit from the raw buffer.

        A batch that overshoots ``buffer_size`` trains a single model using
        all buffered data, then clears its buffer. ``predict`` always uses the
        last model's frozen transformer snapshot until the next retrain.
        """
        X, y = self._validate_chunk(X_chunk, y_chunk)
        self.n_features_in_ = X.shape[1]
        # Update each transform on the incoming chunk, not on historical data.
        current = X
        for _, transformer in self.steps[:-1]:
            transformer.partial_fit(current)
            current = transformer.transform(current)
        size = getattr(self.estimator, "buffer_size", None)
        if size is None:
            self.estimator.partial_fit(current, y)
            self._predict_transformers = deepcopy(self.steps[:-1])
            return self
        validate_positive_int(size, "buffer_size")
        self._pending_X.append(X.copy())
        self._pending_y.append(y.copy())
        self.n_buffered_samples_ += len(X)
        if self.n_buffered_samples_ >= size:
            # Only transform the raw buffer when the tree is about to retrain.
            full_X = self._live_transform(np.concatenate(self._pending_X))
            full_y = np.concatenate(self._pending_y)
            # If one-hot categories expanded, the old tree has a different
            # feature width. A fresh fit (not partial_fit on the old schema)
            # is required; the pipeline's raw buffer threshold still applies.
            old_width = getattr(self.estimator, "n_features_in_", None)
            if old_width is not None and old_width != full_X.shape[1]:
                self.estimator.fit(full_X, full_y)
            else:
                self.estimator.partial_fit(full_X, full_y)
            if getattr(self.estimator, "root_", None) is None and hasattr(self.estimator, "root_"):
                raise RuntimeError("Estimator did not retrain at the buffer threshold")
            self._predict_transformers = deepcopy(self.steps[:-1])
            self._pending_X.clear()
            self._pending_y.clear()
            self.n_buffered_samples_ = 0
        return self

    def predict(self, X):
        """Predict using the latest trained model + frozen preprocessing."""
        if self._predict_transformers is None:
            raise RuntimeError("Pipeline has not trained a model yet")
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("Prediction input feature count does not match training")
        for _, transformer in self._predict_transformers:
            X = transformer.transform(X)
        return self.estimator.predict(X)

    def predict_proba(self, X):
        """Return estimated probabilities when the final estimator supports it."""
        if not callable(getattr(self.estimator, "predict_proba", None)):
            raise AttributeError("Final estimator does not implement predict_proba()")
        if self._predict_transformers is None:
            raise RuntimeError("Pipeline has not trained a model yet")
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("Prediction input feature count does not match training")
        for _, transformer in self._predict_transformers:
            X = transformer.transform(X)
        return self.estimator.predict_proba(X)

    def transform(self, X):
        """Transform with the *current* preprocessor state (not the frozen one)."""
        X = np.asarray(X)
        if X.ndim != 2 or self.n_features_in_ is None:
            raise RuntimeError("Pipeline is not fitted or X is not 2D")
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Incorrect input feature count")
        return self._live_transform(X)

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

    def get_step(self, name):
        """Return the live step by its registered name, or raise KeyError."""
        for step_name, step in self.steps:
            if name == step_name:
                return step
        raise KeyError(f"No pipeline step named {name}")

    @property
    def buffer_memory_bytes(self):
        return sum(a.nbytes for a in self._pending_X + self._pending_y) + getattr(self.estimator, "buffer_memory_bytes", 0)
