"""Prequential stream orchestrator: evaluate first, then learn from each chunk."""

from time import perf_counter
import tracemalloc
import numpy as np

from ._validation import labels_1d
from .metrics import StreamingClassificationMetrics


class StreamTrainer:
    """Run progressive evaluation/training and keep per-chunk diagnostic logs.

    Parameters
    ----------
    model : DecisionTreeClassifier or streaming Pipeline
        Must implement partial_fit and predict; should expose n_retrains_.
    window_size : int or None
        Optional rolling metric horizon (in samples).
    average : str
        Precision/recall/F1 averaging method.

    ``process_chunk`` calls score_chunk *before* fit_chunk, avoiding test-on-
    training-data leakage. Until the first retrain predictions do not exist:
    those chunks have accuracy=NaN and are omitted from cumulative evaluation.
    Memory fields report NumPy buffered bytes and peak Python tracing bytes,
    not whole-system resident memory. Predicted labels and chunk truth are
    retained only for the most recent evaluated chunk.
    """

    def __init__(self, model, window_size=None, average="macro"):
        if not callable(getattr(model, "partial_fit", None)) or not callable(getattr(model, "predict", None)):
            raise TypeError("model must implement partial_fit() and predict()")
        self.model = model
        self.metrics = StreamingClassificationMetrics(window_size=window_size, average=average)
        self.logs = []
        self.last_y_true_ = None
        self.last_y_pred_ = None
        self.n_chunks_ = 0

    def _is_trained(self):
        if hasattr(self.model, "_predict_transformers"):
            return self.model._predict_transformers is not None
        if hasattr(self.model, "root_"):
            return self.model.root_ is not None
        return getattr(self.model, "n_retrains_", 0) > 0

    def score_chunk(self, X_chunk, y_chunk):
        """Evaluate *without fitting*; updates cumulative metrics if trained."""
        X = np.asarray(X_chunk)
        y = labels_1d(y_chunk, len(X))
        if not self._is_trained():
            self.last_y_true_ = None
            self.last_y_pred_ = None
            return {"evaluated": False, "chunk_accuracy": float("nan"),
                    "cumulative_accuracy": self.metrics.result()["accuracy"]}
        pred = self.model.predict(X)
        pred = labels_1d(pred, len(y))
        self.last_y_true_ = y.copy()
        self.last_y_pred_ = pred.copy()
        # Optional class-1 probability enables genuine score-based ROC-AUC.
        scores = None
        if callable(getattr(self.model, "predict_proba", None)):
            probs = self.model.predict_proba(X)
            fitted = self.model.estimator if hasattr(self.model, "estimator") else self.model
            classes = getattr(fitted, "classes_", None)
            if classes is not None and np.any(classes == self.metrics.positive_label):
                j = int(np.flatnonzero(classes == self.metrics.positive_label)[0])
                scores = probs[:, j]
        self.metrics.update(y, pred, scores)
        current = self.metrics.result()
        return {"evaluated": True, "chunk_accuracy": float(np.mean(y == pred)),
                "cumulative_accuracy": current["accuracy"],
                "cumulative_f1": current["f1"], "cumulative_auc": current["auc"],
                "rolling_accuracy": current.get("rolling_accuracy", float("nan"))}

    def fit_chunk(self, X_chunk, y_chunk):
        """Train one chunk by calling the model/pipeline partial_fit method."""
        start = perf_counter()
        self.model.partial_fit(X_chunk, y_chunk)
        return {"fit_seconds": perf_counter() - start,
                "n_retrains": self.n_retrains,
                "buffer_memory_bytes": self.buffer_memory_bytes}

    @property
    def n_retrains(self):
        model = self.model.estimator if hasattr(self.model, "estimator") else self.model
        return int(getattr(model, "n_retrains_", 0))

    @property
    def buffer_memory_bytes(self):
        return int(getattr(self.model, "buffer_memory_bytes", 0))

    def process_chunk(self, X_chunk, y_chunk):
        """Score, fit, and log a chunk; return its flat metrics/log dictionary."""
        # Exclude user data generation and CSV-reading time from benchmark.
        tracemalloc.start()
        try:
            score = self.score_chunk(X_chunk, y_chunk)
            before = self.n_retrains
            training = self.fit_chunk(X_chunk, y_chunk)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.n_chunks_ += 1
        log = {"chunk": self.n_chunks_, "samples": len(X_chunk),
               **score, **training,
               "retrained": self.n_retrains > before,
               "traced_peak_memory_bytes": peak}
        self.logs.append(log)
        return log

    def run(self, chunks):
        """Process an iterable of (X_chunk, y_chunk) pairs, returning logs."""
        for X, y in chunks:
            self.process_chunk(X, y)
        return self.logs

    def reset_logs(self):
        """Reset evaluation/logging but do not discard the trained model."""
        self.metrics.reset()
        self.logs.clear()
        self.n_chunks_ = 0
        self.last_y_true_ = None
        self.last_y_pred_ = None
        return self
