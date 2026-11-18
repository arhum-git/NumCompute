"""Matplotlib integration and reproducible benchmarking."""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest
from matplotlib import pyplot as plt
from numcompute_stream.visualise import (plot_metric_over_time, compare_models,
                                          plot_predictions_vs_ground_truth)
from benchmark.stream_benchmark import (make_stream, confusion_loop,
    confusion_vectorised, benchmark_buffers)


def test_metric_plot_save(tmp_path):
    path = tmp_path / "accuracy.png"
    fig, ax = plot_metric_over_time([np.nan, .5, .8], "Accuracy", "Value", save_path=path)
    assert path.exists() and ax.get_xlabel() == "Chunk number"
    plt.close(fig)


def test_model_comparison_plot():
    fig, ax = compare_models([.3, .7], [.5, .6], ("small", "large"))
    assert len(ax.get_lines()) == 2
    plt.close(fig)


def test_prediction_plot():
    fig, ax = plot_predictions_vs_ground_truth([0, 1], [0, 0])
    assert len(ax.collections) == 2
    plt.close(fig)


def test_plot_invalid_shapes():
    with pytest.raises(ValueError):
        plot_metric_over_time([[1, 2]], "metric", "y")
    with pytest.raises(ValueError):
        plot_predictions_vs_ground_truth([0, 1], [1])


def test_loop_equals_vectorised():
    a = np.array([1, 0, 1, 1, 0], dtype=int)
    b = np.array([1, 0, 0, 1, 0], dtype=int)
    np.testing.assert_array_equal(confusion_loop(a, b), confusion_vectorised(a, b))


def test_buffer_benchmark_returns_rows():
    X, y = make_stream(96)
    rows = benchmark_buffers(X, y, sizes=(16, 32), chunk_size=8)
    assert len(rows) == 2
    assert all(r["n_retrains"] >= 1 for r in rows)
    assert all(0 <= r["prequential_accuracy"] <= 1 for r in rows)
