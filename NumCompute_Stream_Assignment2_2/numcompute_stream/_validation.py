"""Shared, strict input validation for the streaming NumCompute modules."""

import numpy as np


def numeric_2d(X, *, allow_nan=True, empty=False):
    """Return numeric 2D float data; reject infinity and incorrect dimensions."""
    try:
        X = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("X must contain numeric values") from exc
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (samples, features), got {X.ndim}D")
    if X.shape[1] == 0 or (not empty and X.shape[0] == 0):
        raise ValueError("X must contain at least one feature and one sample")
    if np.isinf(X).any() or (not allow_nan and np.isnan(X).any()):
        raise ValueError("X contains invalid infinite or missing values")
    return X


def labels_1d(y, n_samples=None):
    """Return a 1D label array, validating its length and missing labels."""
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if y.size == 0:
        raise ValueError("y must not be empty")
    if n_samples is not None and y.size != n_samples:
        raise ValueError(f"X has {n_samples} samples but y has {y.size}")
    if y.dtype.kind in "fc" and not np.isfinite(y).all():
        raise ValueError("y cannot contain NaN or infinite labels")
    if y.dtype.kind == "O":
        for value in y:
            if value is None or (isinstance(value, (float, np.floating)) and not np.isfinite(value)):
                raise ValueError("y cannot contain missing labels")
    return y


def categorical_2d(X, *, empty=False):
    """Require 2D category data, preserving its input dtype."""
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be 2D (samples, features)")
    if X.shape[1] == 0 or (not empty and X.shape[0] == 0):
        raise ValueError("X must contain samples and features")
    return X


def check_features(X, n_features):
    if n_features is None:
        raise RuntimeError("This component must be fitted before transform/predict")
    if X.shape[1] != n_features:
        raise ValueError(f"Expected {n_features} features, got {X.shape[1]}")


def validate_positive_int(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)
