"""Tests of buffered retraining, split selection, NaN paths and exact prediction."""
import numpy as np
import pytest
from numcompute_stream import DecisionTreeClassifier


@pytest.fixture
def two_class():
    X = np.array([[0], [1], [2], [3], [4], [5]], dtype=float)
    return X, np.array([0, 0, 0, 1, 1, 1])


def test_gini_known_prediction(two_class):
    X, y = two_class
    tree = DecisionTreeClassifier(max_depth=2).fit(X, y)
    np.testing.assert_array_equal(tree.predict(X), y)


def test_entropy_known_prediction(two_class):
    X, y = two_class
    tree = DecisionTreeClassifier(criterion="entropy").fit(X, y)
    np.testing.assert_array_equal(tree.predict(X), y)


def test_max_depth_one_root_split(two_class):
    X, y = two_class
    tree = DecisionTreeClassifier(max_depth=1).fit(X, y)
    assert tree.root_.feature == 0
    assert tree.root_.left.feature == -1


def test_no_split_on_constant_features():
    tree = DecisionTreeClassifier().fit(np.ones((4, 2)), [0, 1, 0, 1])
    assert tree.root_.feature == -1
    np.testing.assert_array_equal(tree.predict(np.ones((2, 2))), [0, 0])


def test_tie_majority_deterministic():
    tree = DecisionTreeClassifier(max_depth=1, min_samples_split=4).fit(np.ones((4, 1)), [2, 1, 2, 1])
    assert tree.predict([[1.]])[0] == 1


def test_nan_train_and_predict():
    X = np.array([[0], [1], [10], [11], [np.nan]], dtype=float)
    tree = DecisionTreeClassifier().fit(X, np.array([0, 0, 1, 1, 1]))
    assert tree.predict([[np.nan]])[0] in (0, 1)
    assert tree.predict([[10]])[0] == 1


def test_predict_proba_sums_to_one(two_class):
    X, y = two_class
    tree = DecisionTreeClassifier().fit(X, y)
    p = tree.predict_proba(X)
    assert p.shape == (6, 2)
    np.testing.assert_allclose(p.sum(axis=1), 1)


def test_tree_untrained_predict_raises():
    with pytest.raises(RuntimeError):
        DecisionTreeClassifier(buffer_size=3).predict([[1.0]])


def test_tree_untrained_predict_proba_raises():
    with pytest.raises(RuntimeError):
        DecisionTreeClassifier().predict_proba([[1.0]])


def test_buffer_below_threshold_no_tree():
    tree = DecisionTreeClassifier(buffer_size=4)
    tree.partial_fit([[0], [1]], [0, 0])
    assert tree.root_ is None
    assert tree.n_buffered_samples_ == 2


def test_buffer_retrain_at_threshold_clears():
    tree = DecisionTreeClassifier(buffer_size=4)
    tree.partial_fit([[0], [1]], [0, 0]).partial_fit([[2], [3]], [1, 1])
    assert tree.n_retrains_ == 1
    assert tree.n_buffered_samples_ == 0
    np.testing.assert_array_equal(tree.predict([[0], [3]]), [0, 1])


def test_previous_tree_remains_between_retrains():
    tree = DecisionTreeClassifier(buffer_size=4)
    tree.partial_fit([[0], [1], [2], [3]], [0, 0, 1, 1])
    before = tree.predict([[0], [3]]).copy()
    tree.partial_fit([[0]], [1])
    np.testing.assert_array_equal(tree.predict([[0], [3]]), before)
    assert tree.n_retrains_ == 1


def test_buffer_overshoot_single_retrain():
    tree = DecisionTreeClassifier(buffer_size=3)
    tree.partial_fit(np.arange(8).reshape(-1, 1), np.arange(8) > 3)
    assert tree.n_retrains_ == 1
    assert tree.n_buffered_samples_ == 0


def test_retrain_replaces_previous_tree():
    tree = DecisionTreeClassifier(buffer_size=2)
    tree.partial_fit([[0], [1]], [0, 0])
    tree.partial_fit([[0], [1]], [1, 1])
    assert tree.n_retrains_ == 2
    assert tree.predict([[0]])[0] == 1


def test_fit_discards_buffer():
    tree = DecisionTreeClassifier(buffer_size=5)
    tree.partial_fit([[1]], [0])
    tree.fit([[0], [1]], [0, 1])
    assert tree.n_buffered_samples_ == 0


def test_buffer_memory_bytes_nonzero_then_zero():
    tree = DecisionTreeClassifier(buffer_size=3)
    tree.partial_fit([[1]], [0])
    assert tree.buffer_memory_bytes > 0
    tree.partial_fit([[2], [3]], [0, 1])
    assert tree.buffer_memory_bytes == 0


@pytest.mark.parametrize("config", [{"buffer_size": 0}, {"max_depth": 0},
                                  {"criterion": "bad"}, {"max_features": 0},
                                  {"max_features": 2.5}, {"min_samples_split": 0}])
def test_bad_hyperparameters(config):
    with pytest.raises(ValueError):
        DecisionTreeClassifier(**config)


def test_reject_infinity():
    with pytest.raises(ValueError):
        DecisionTreeClassifier().fit([[np.inf], [1.]], [0, 1])


def test_reject_mismatched_targets():
    with pytest.raises(ValueError):
        DecisionTreeClassifier().fit([[0], [1]], [0])


def test_reject_predict_feature_mismatch():
    tree = DecisionTreeClassifier().fit([[1, 2], [3, 4]], [0, 1])
    with pytest.raises(ValueError):
        tree.predict([[2]])


def test_feature_subsample_deterministic():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(50, 5))
    y = (X[:, 0] > 0).astype(int)
    a = DecisionTreeClassifier(max_features="sqrt", random_state=3).fit(X, y)
    b = DecisionTreeClassifier(max_features="sqrt", random_state=3).fit(X, y)
    np.testing.assert_array_equal(a.predict(X), b.predict(X))


def test_empty_prediction_shape():
    tree = DecisionTreeClassifier().fit([[1], [2]], [0, 1])
    assert tree.predict(np.empty((0, 1))).shape == (0,)
    assert tree.predict_proba(np.empty((0, 1))).shape == (0, 2)
