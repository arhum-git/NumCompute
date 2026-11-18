"""Reusable Matplotlib figures for progressive model training and evaluation."""

import numpy as np
import matplotlib.pyplot as plt


def _finish(fig, save_path=None, show=False):
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_metric_over_time(metric_values, title="Metric over time", ylabel="Value",
                          *, save_path=None, show=False):
    """Line chart over chunks; return (fig, ax). NaNs create intentional gaps."""
    values = np.asarray(metric_values, dtype=float)
    if values.ndim != 1:
        raise ValueError("metric_values must be 1D")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(np.arange(1, len(values) + 1), values, marker="o", markersize=3)
    ax.set(xlabel="Chunk number", ylabel=ylabel, title=title)
    ax.grid(alpha=0.3)
    return _finish(fig, save_path, show), ax


def compare_models(metric1, metric2, labels=("Model 1", "Model 2"),
                   *, title="Streaming model comparison", ylabel="Metric",
                   save_path=None, show=False):
    """Compare two streaming metrics (lists of per-chunk results)."""
    a = np.asarray(metric1, dtype=float)
    b = np.asarray(metric2, dtype=float)
    if a.ndim != 1 or b.ndim != 1 or len(labels) != 2:
        raise ValueError("Metric arrays must be 1D and labels must have two names")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(np.arange(1, len(a) + 1), a, label=str(labels[0]), marker=".")
    ax.plot(np.arange(1, len(b) + 1), b, label=str(labels[1]), marker=".")
    ax.set(xlabel="Chunk number", ylabel=ylabel, title=title)
    ax.legend()
    ax.grid(alpha=0.3)
    return _finish(fig, save_path, show), ax


def plot_predictions_vs_ground_truth(y_true, y_pred, *,
                                     title="Predictions vs ground truth",
                                     save_path=None, show=False):
    """Scatter of category labels for each observation in the latest chunk."""
    actual = np.asarray(y_true)
    predicted = np.asarray(y_pred)
    if actual.ndim != 1 or predicted.shape != actual.shape:
        raise ValueError("y_true and y_pred must be equal-length 1D arrays")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(1, len(actual) + 1)
    ax.scatter(x, actual, label="Ground truth", marker="o", alpha=0.75)
    ax.scatter(x, predicted, label="Prediction", marker="x", alpha=0.75)
    ax.set(xlabel="Sample within chunk", ylabel="Class label", title=title)
    ax.legend()
    ax.grid(alpha=0.3)
    return _finish(fig, save_path, show), ax
