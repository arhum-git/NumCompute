"""Progressive classification metrics, confusion matrices and exact binary ROC-AUC.

Counts are accumulated using vectorised NumPy indexed counting. Exact AUC
retains scores and truth values, O(N) memory; other cumulative metrics use
O(K^2) memory for K observed labels. Rolling metrics retain at most W pairs.
"""

from collections import deque
import numpy as np
from ._validation import labels_1d, validate_positive_int


def _matrix_metrics(cm, average="macro", positive_label_idx=None):
    """Calculate precision, recall, F1 using counts, never divide by zero."""
    tp = np.diag(cm).astype(float)
    predicted = cm.sum(axis=0)
    actual = cm.sum(axis=1)
    precision = np.divide(tp, predicted, out=np.zeros(len(tp)), where=predicted > 0)
    recall = np.divide(tp, actual, out=np.zeros(len(tp)), where=actual > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(len(tp)), where=(precision + recall) > 0)
    if average == "binary":
        if positive_label_idx is None:
            return (0.0, 0.0, 0.0)
        idx = positive_label_idx
        return float(precision[idx]), float(recall[idx]), float(f1[idx])
    if average == "micro":
        val = float(tp.sum() / cm.sum()) if cm.sum() else 0.0
        return val, val, val
    if average == "weighted":
        weights = np.divide(actual, actual.sum(), out=np.zeros(len(tp)), where=actual.sum() > 0)
        return (float(weights @ precision), float(weights @ recall), float(weights @ f1))
    if average == "macro":
        # Do not count a label with no actual/predicted observations.
        active = (actual + predicted) > 0
        return (float(np.mean(precision[active])) if active.any() else 0.0,
                float(np.mean(recall[active])) if active.any() else 0.0,
                float(np.mean(f1[active])) if active.any() else 0.0)
    raise ValueError("average must be 'binary', 'macro', 'micro' or 'weighted'")


def _binary_auc(y_true, scores, positive_label):
    """ROC-AUC via averaged ranks; tied scores count as half a win."""
    y = np.asarray(y_true) == positive_label
    scores = np.asarray(scores, dtype=float)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if not n_pos or not n_neg:
        return float("nan")
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    starts = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1]
    ends = np.r_[starts[1:], len(scores)]
    avg = (starts + ends + 1) / 2.0   # average 1-based rank of each tie group
    ranks = np.repeat(avg, ends - starts)
    rank_sum = float(ranks[np.argsort(order)][y].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


class StreamingClassificationMetrics:
    """Cumulative and rolling accuracy/precision/recall/F1/confusion/AUC.

    Parameters
    ----------
    window_size : int or None
        If given, retain the last W *observations* for rolling metrics.
    average : {'macro','micro','weighted','binary'}
    positive_label : scalar
        Label treated as positive for binary averaging and ROC-AUC.

    ``update(y_true_chunk, y_pred_chunk, y_score=None)`` can receive a 1D
    probability/score vector for ROC-AUC. Each update returns self. AUC is NaN
    until both labels and scores are observed. Use reset() to restart.
    """

    def __init__(self, window_size=None, average="macro", positive_label=1):
        if window_size is not None:
            validate_positive_int(window_size, "window_size")
        if average not in ("macro", "micro", "weighted", "binary"):
            raise ValueError("Invalid average")
        self.window_size = window_size
        self.average = average
        self.positive_label = positive_label
        self.reset()

    def reset(self):
        self.classes_ = np.array([], dtype=object)
        self.confusion_matrix_ = np.zeros((0, 0), dtype=np.int64)
        self.n_samples_ = 0
        self._window = deque(maxlen=self.window_size) if self.window_size else None
        self._auc_true = []
        self._auc_score = []
        return self

    def update(self, y_true_chunk, y_pred_chunk, y_score=None):
        """Accumulate one chunk; confusion update is O(n + K²) time."""
        truth = labels_1d(y_true_chunk)
        pred = labels_1d(y_pred_chunk, len(truth))
        score = None
        if y_score is not None:
            score = np.asarray(y_score, dtype=float)
            if score.shape != truth.shape or not np.isfinite(score).all():
                raise ValueError("y_score must be finite and shape (n_samples,)")
        # Reindex the old confusion matrix if new labels were introduced.
        observed = np.unique(np.concatenate((truth, pred)))
        new_classes = np.union1d(self.classes_, observed)
        if len(new_classes) != len(self.classes_) or not np.array_equal(new_classes, self.classes_):
            cm = np.zeros((len(new_classes), len(new_classes)), dtype=np.int64)
            if len(self.classes_):
                old = np.searchsorted(new_classes, self.classes_)
                cm[np.ix_(old, old)] = self.confusion_matrix_
            self.confusion_matrix_ = cm
            self.classes_ = new_classes
        true_idx = np.searchsorted(self.classes_, truth)
        pred_idx = np.searchsorted(self.classes_, pred)
        np.add.at(self.confusion_matrix_, (true_idx, pred_idx), 1)
        self.n_samples_ += len(truth)
        if self._window is not None:
            self._window.extend(zip(truth.tolist(), pred.tolist(),
                                    score.tolist() if score is not None else [None] * len(truth)))
        if score is not None:
            self._auc_true.append(truth.copy())
            self._auc_score.append(score.copy())
        return self

    partial_fit = update

    def _score_matrix(self, cm):
        index = np.flatnonzero(self.classes_ == self.positive_label)
        p, r, f = _matrix_metrics(cm, self.average, int(index[0]) if len(index) else None)
        total = cm.sum()
        return {"accuracy": float(np.trace(cm) / total) if total else float("nan"),
                "precision": p, "recall": r, "f1": f, "confusion_matrix": cm}

    def result(self):
        """Return current metrics and a copy of the cumulative confusion matrix."""
        result = self._score_matrix(self.confusion_matrix_.copy())
        result["n_samples"] = self.n_samples_
        result["classes"] = self.classes_.copy()
        result["auc"] = (_binary_auc(np.concatenate(self._auc_true),
                                      np.concatenate(self._auc_score), self.positive_label)
                         if self._auc_true else float("nan"))
        if self._window is not None:
            window = list(self._window)
            rolling = np.zeros_like(self.confusion_matrix_)
            if window:
                t = np.array([v[0] for v in window])
                p = np.array([v[1] for v in window])
                np.add.at(rolling, (np.searchsorted(self.classes_, t),
                                    np.searchsorted(self.classes_, p)), 1)
            roll = self._score_matrix(rolling)
            result.update({f"rolling_{k}": v for k, v in roll.items()})
            result["rolling_n_samples"] = len(window)
            scored = [(v[0], v[2]) for v in window if v[2] is not None]
            result["rolling_auc"] = (_binary_auc([a for a, _ in scored], [b for _, b in scored], self.positive_label)
                                     if scored else float("nan"))
        return result


class _SingleMetric:
    """Common update/reset/result API for a named progressive metric."""

    key = None

    def __init__(self, window_size=None, average="macro", positive_label=1):
        self.tracker = StreamingClassificationMetrics(window_size, average, positive_label)

    def update(self, y_true_chunk, y_pred_chunk, y_score=None):
        self.tracker.update(y_true_chunk, y_pred_chunk, y_score)
        return self

    partial_fit = update

    def reset(self):
        self.tracker.reset()
        return self

    def result(self):
        return self.tracker.result()[self.key]


class StreamingAccuracy(_SingleMetric):
    """Incremental accuracy; constant-memory confusion counters."""
    key = "accuracy"


class StreamingPrecision(_SingleMetric):
    """Incremental precision with macro/micro/weighted/binary averaging."""
    key = "precision"


class StreamingRecall(_SingleMetric):
    """Incremental recall with macro/micro/weighted/binary averaging."""
    key = "recall"


class StreamingF1(_SingleMetric):
    """Incremental F1 with macro/micro/weighted/binary averaging."""
    key = "f1"


class StreamingConfusionMatrix(_SingleMetric):
    """Incremental confusion matrix with class expansion and deterministic order."""
    key = "confusion_matrix"


class StreamingAUC(_SingleMetric):
    """Exact binary AUC; must pass y_score to update; retains O(N) scores."""
    key = "auc"
