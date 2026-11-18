"""Incremental preprocessing: known values and input validation."""
import numpy as np
import pytest
from numcompute_stream import StandardScaler, MinMaxScaler, SimpleImputer, OneHotEncoder


def test_standard_online_matches_batch():
    X = np.array([[1., 3.], [3., 5.], [5., 7.], [7., 9.]])
    incremental = StandardScaler().partial_fit(X[:2]).partial_fit(X[2:])
    batch = StandardScaler().fit(X)
    np.testing.assert_allclose(incremental.mean_, batch.mean_)
    np.testing.assert_allclose(incremental.var_, batch.var_)
    np.testing.assert_allclose(incremental.transform(X), batch.transform(X))


def test_standard_online_unequal_chunks():
    X = np.arange(17., dtype=float).reshape(-1, 1)
    s = StandardScaler().partial_fit(X[:2]).partial_fit(X[2:7]).partial_fit(X[7:])
    np.testing.assert_allclose(s.mean_, X.mean(axis=0))
    np.testing.assert_allclose(s.var_, X.var(axis=0))


def test_standard_constant_scale_one():
    s = StandardScaler().partial_fit([[5.], [5.]])
    np.testing.assert_allclose(s.scale_, [1.])
    np.testing.assert_allclose(s.transform([[5.]]), [[0.]])


def test_standard_ignores_nans():
    s = StandardScaler().partial_fit([[1., np.nan], [3., 5.]])
    np.testing.assert_allclose(s.mean_, [2., 5.])
    assert np.isnan(s.transform([[np.nan, 5.]])[0, 0])


def test_standard_all_nan_column():
    s = StandardScaler().fit([[np.nan, 2], [np.nan, 4]])
    assert s.count_[0] == 0
    assert s.scale_[0] == 1
    assert np.isnan(s.transform([[np.nan, 4]])[0, 0])


def test_standard_fit_resets():
    s = StandardScaler().partial_fit([[100.]])
    s.fit([[1.], [3.]])
    np.testing.assert_allclose(s.mean_, [2.])
    assert s.count_[0] == 2


def test_standard_feature_mismatch():
    s = StandardScaler().fit([[1.], [2.]])
    with pytest.raises(ValueError):
        s.partial_fit([[1., 2.]])


def test_standard_transform_before_fit():
    with pytest.raises(RuntimeError):
        StandardScaler().transform([[1.]])


def test_minmax_incremental_extrema():
    s = MinMaxScaler().partial_fit([[3., 7.], [2., 8.]]).partial_fit([[1., 12.]])
    np.testing.assert_allclose(s.data_min_, [1., 7.])
    np.testing.assert_allclose(s.data_max_, [3., 12.])
    np.testing.assert_allclose(s.transform([[1., 12.]]), [[0., 1.]])


def test_minmax_constant_column():
    s = MinMaxScaler(feature_range=(-1., 1.)).fit([[5.], [5.]])
    np.testing.assert_allclose(s.transform([[5.], [8.]]), [[-1.], [-1.]])


def test_minmax_ignores_missing():
    s = MinMaxScaler().fit([[np.nan], [2.], [4.]])
    assert np.isnan(s.transform([[np.nan]])[0, 0])
    np.testing.assert_allclose(s.transform([[3.]]), [[0.5]])


def test_minmax_bad_range():
    with pytest.raises(ValueError):
        MinMaxScaler(feature_range=(1, 1))


def test_imputer_running_mean():
    imp = SimpleImputer().partial_fit([[1.], [np.nan]])
    imp.partial_fit([[3.], [5.]])
    np.testing.assert_allclose(imp.statistics_, [3.])
    np.testing.assert_allclose(imp.transform([[np.nan]]), [[3.]])


def test_imputer_median():
    imp = SimpleImputer(strategy="median").partial_fit([[1.], [100.]])
    imp.partial_fit([[2.]])
    np.testing.assert_allclose(imp.statistics_, [2.])


def test_imputer_constant_and_all_nan():
    imp = SimpleImputer(strategy="constant", fill_value=-9).fit([[np.nan], [np.nan]])
    np.testing.assert_allclose(imp.transform([[np.nan]]), [[-9.]])


def test_imputer_mean_all_nan_fallback():
    imp = SimpleImputer(strategy="mean", fill_value=-7).fit([[np.nan], [np.nan]])
    np.testing.assert_allclose(imp.transform([[np.nan]]), [[-7.]])


def test_imputer_invalid_strategy():
    with pytest.raises(ValueError):
        SimpleImputer(strategy="guess")


def test_encoder_grows_without_shifting_old_columns():
    enc = OneHotEncoder().partial_fit(np.array([["red"], ["blue"]]))
    before = enc.transform([["red"]]).copy()
    enc.partial_fit(np.array([["green"]]))
    np.testing.assert_array_equal(enc.transform([["red"]])[:, :2], before)
    assert enc.n_output_features_ == 3


def test_encoder_multiple_features():
    enc = OneHotEncoder().fit([["x", "a"], ["y", "a"], ["x", "b"]])
    assert enc.transform([["x", "b"]]).shape == (1, 4)
    assert enc.transform([["x", "b"]]).sum() == 2


def test_encoder_unknown_ignore():
    enc = OneHotEncoder(handle_unknown="ignore").fit([["a"], ["b"]])
    np.testing.assert_allclose(enc.transform([["never-seen"]]), [[0, 0]])


def test_encoder_unknown_error():
    enc = OneHotEncoder(handle_unknown="error").fit([["a"]])
    with pytest.raises(ValueError):
        enc.transform([["b"]])


def test_encoder_missing_rejected():
    with pytest.raises(ValueError):
        OneHotEncoder().fit(np.array([[np.nan]]))


def test_encoder_input_width_validation():
    enc = OneHotEncoder().fit([["a", "b"]])
    with pytest.raises(ValueError):
        enc.transform([["a"]])


def test_encoder_transform_missing_rejected_even_if_unknown_ignored():
    enc = OneHotEncoder(handle_unknown="ignore").fit([[1.]])
    with pytest.raises(ValueError, match="Missing"):
        enc.transform([[np.nan]])
