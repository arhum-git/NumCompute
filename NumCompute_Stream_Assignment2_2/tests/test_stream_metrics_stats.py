"""Progressive metrics and moments, including known numerical answers."""
import numpy as np
import pytest
from numcompute_stream import (StreamingStats, StreamingClassificationMetrics,
    StreamingAccuracy, StreamingPrecision, StreamingRecall, StreamingF1,
    StreamingConfusionMatrix, StreamingAUC)


def test_welford_merge_matches_numpy():
    X = np.array([[1., 2.], [3., 5.], [9., 7.], [11., 8.]])
    s = StreamingStats().update_stats(X[:1]).update_stats(X[1:3]).update_stats(X[3:])
    np.testing.assert_allclose(s.result()["mean"], X.mean(axis=0))
    np.testing.assert_allclose(s.result()["variance"], X.var(axis=0))


def test_stats_missing_column():
    s = StreamingStats().update_stats([[np.nan, 1], [np.nan, 3]])
    r = s.result()
    assert np.isnan(r["mean"][0])
    assert np.isnan(r["variance"][0])
    np.testing.assert_allclose(r["mean"][1], 2)


def test_stats_quantiles_known():
    s = StreamingStats().update_stats(np.array([[1], [2], [3], [4], [5]], float))
    np.testing.assert_allclose(s.quantile([0., .5, 1.]).ravel(), [1, 3, 5])


def test_stats_chunk_histogram():
    s = StreamingStats(bins=2, histogram_range=(0, 4))
    s.update_stats([[0.5], [1.5]]).update_stats([[2.5], [3.5]])
    counts, edges = s.histogram()
    np.testing.assert_array_equal(counts, [[2, 2]])
    np.testing.assert_array_equal(edges, [[0, 2, 4]])


def test_stats_dynamic_histogram():
    s = StreamingStats(bins=2).fit([[1], [2], [3], [4]])
    counts, edges = s.histogram()
    assert counts.sum() == 4
    assert len(edges[0]) == 3


def test_stats_rolling_history():
    s = StreamingStats(history_size=3).update_stats([[1], [2]])
    s.update_stats([[3], [4]])
    assert s._history_rows == 3
    np.testing.assert_allclose(s.quantile([0.5]).ravel(), [3])
    np.testing.assert_allclose(s.result()["mean"], [2.5])


def test_stats_reset():
    s = StreamingStats().fit([[1]])
    s.reset()
    with pytest.raises(RuntimeError):
        s.result()


def test_stats_reject_bad_quantile():
    with pytest.raises(ValueError):
        StreamingStats().fit([[1]]).quantile([2])


def test_metrics_accuracy_known():
    m = StreamingClassificationMetrics()
    m.update([0, 0, 1, 1], [0, 1, 1, 1])
    assert m.result()["accuracy"] == 0.75


def test_metrics_confusion_known():
    m = StreamingClassificationMetrics()
    m.update([0, 0, 1, 1], [0, 1, 1, 1])
    np.testing.assert_array_equal(m.result()["confusion_matrix"], [[1, 1], [0, 2]])


def test_metrics_precision_recall_f1_binary():
    m = StreamingClassificationMetrics(average="binary")
    m.update([0, 0, 1, 1], [0, 1, 1, 1])
    r = m.result()
    np.testing.assert_allclose([r["precision"], r["recall"], r["f1"]], [2 / 3, 1, .8])


def test_metrics_rolling_last_three():
    m = StreamingClassificationMetrics(window_size=3)
    m.update([0, 0, 1], [0, 1, 1])
    m.update([1, 1], [0, 0])
    assert m.result()["rolling_n_samples"] == 3
    np.testing.assert_allclose(m.result()["rolling_accuracy"], 1 / 3)
    np.testing.assert_allclose(m.result()["accuracy"], 2 / 5)


def test_metrics_class_expands():
    m = StreamingClassificationMetrics().update([0, 1], [0, 1])
    m.update([2], [2])
    np.testing.assert_array_equal(m.result()["confusion_matrix"], np.eye(3, dtype=int))


def test_metrics_auc_known():
    m = StreamingClassificationMetrics()
    m.update([0, 0], [0, 1], [0.1, 0.4])
    m.update([1, 1], [1, 1], [0.35, 0.8])
    assert m.result()["auc"] == 0.75


def test_metrics_auc_tie_half_credit():
    m = StreamingClassificationMetrics()
    m.update([0, 1], [0, 1], [0.5, 0.5])
    assert m.result()["auc"] == 0.5


def test_metrics_auc_no_pos_returns_nan():
    m = StreamingClassificationMetrics().update([0, 0], [0, 0], [0.1, 0.2])
    assert np.isnan(m.result()["auc"])


def test_metrics_reset():
    m = StreamingClassificationMetrics().update([0], [0])
    m.reset()
    assert m.result()["n_samples"] == 0
    assert np.isnan(m.result()["accuracy"])


def test_metrics_invalid_predictions():
    with pytest.raises(ValueError):
        StreamingClassificationMetrics().update([0, 1], [1])


def test_metrics_invalid_scores():
    with pytest.raises(ValueError):
        StreamingClassificationMetrics().update([0, 1], [0, 1], [np.nan, 1.])


@pytest.mark.parametrize("metric,key", [(StreamingAccuracy, 0.5), (StreamingPrecision, 0.5),
                                          (StreamingRecall, 0.5), (StreamingF1, 0.5)])
def test_individual_metric_classes(metric, key):
    m = metric(average="micro")
    m.update([0, 1], [0, 0])
    assert m.result() == key
    m.reset()
    assert np.isnan(m.result()) if metric is StreamingAccuracy else m.result() == 0.0


def test_individual_confusion_metric():
    m = StreamingConfusionMatrix().update([0, 1], [0, 0])
    np.testing.assert_array_equal(m.result(), [[1, 0], [1, 0]])


def test_individual_auc_metric():
    m = StreamingAUC().update([0, 1], [0, 1], [0.2, 0.8])
    assert m.result() == 1.0
