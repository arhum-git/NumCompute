"""
Module: utils.py
Description: Utility functions for the NumCompute toolkit. Includes numerically
             stable activations, distance metrics, and data batching helpers.
"""
import numpy as np


def logsumexp(x, axis=None):
    """Compute log(sum(exp(x))) with max shifting.

    Parameters
    ----------
    x : array_like
        Numeric input of any shape; must be nonempty on reduced axes.
    axis : int or None
        Reduction axis, or all elements when None.

    Returns
    -------
    ndarray
        Reduced values (axis removed unless axis is None, then shape (1,...,1)).

    Raises
    ------
    ValueError
        If the reduction contains no elements.

    Complexity
    ----------
    O(n) time and O(n) temporary memory for n input elements.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0 or (axis is not None and x.shape[axis] == 0):
        raise ValueError("logsumexp requires a nonempty reduction axis")
    m = np.max(x, axis=axis, keepdims=True)
    # -inf inputs contribute zero; +inf dominates their reduction slice.
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        shifted = np.where(np.isfinite(m), x - m, -np.inf)
        out = m + np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
    out = np.where(np.isposinf(m), np.inf, out)
    out = np.where(np.isneginf(m), -np.inf, out)
    return np.squeeze(out, axis=axis) if axis is not None else out


def softmax(x, axis=None):
    """Return normalized exponentials along an axis with stable shifting.

    An all -inf slice has no defined softmax distribution and raises
    ValueError. A slice containing +inf assigns equal probability to its
    +inf entries and zero elsewhere. NaN values propagate.

    Parameters
    ----------
    x : array_like
        Numeric input of any shape.
    axis : int or None
        Axis to normalize; None normalizes the complete array.

    Returns
    -------
    ndarray
        Probabilities with the same shape as x.

    Raises
    ------
    ValueError
        Empty arrays or all-negative-infinity slices.

    Complexity
    ----------
    O(n) time and O(n) space.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        raise ValueError("softmax requires a nonempty array")
    m = np.max(x, axis=axis, keepdims=True)
    if np.any(np.isneginf(m)):
        raise ValueError("softmax is undefined for an all -inf slice")
    with np.errstate(invalid="ignore"):
        e = np.exp(np.where(np.isfinite(m), x - m, -np.inf))
    pos = np.isposinf(x)
    count_pos = np.sum(pos, axis=axis, keepdims=True)
    if np.any(count_pos):
        e = np.where(np.isposinf(m), pos.astype(float), e)
    return e / np.sum(e, axis=axis, keepdims=True)


def sigmoid(x):
    """Compute a numerically stable elementwise logistic activation.

    Parameters
    ----------
    x : array_like
        Numeric values of arbitrary shape.

    Returns
    -------
    ndarray
        Values between 0 and 1, with the same shape as x.

    Complexity
    ----------
    O(n) time and O(n) space.
    """
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    mask = x >= 0
    with np.errstate(under="ignore"):
        out[mask] = 1.0 / (1.0 + np.exp(-x[mask]))
        ex = np.exp(x[~mask])
    out[~mask] = ex / (1.0 + ex)
    return out


def relu(x):
    """
    Compute the ReLU activation element-wise.
    Parameters
    x : np.ndarray

    Returns
    np.ndarray, same shape as x

    Time complexity  : O(n)
    Space complexity : O(n)
    """
    return np.maximum(0, np.asarray(x, dtype=float))


def tanh(x):
    """
    Compute the tanh activation element-wise.

    Parameters
    x : np.ndarray
    Returns
    np.ndarray, same shape as x, values in (-1, 1)

    Time complexity  : O(n)
    Space complexity : O(n)
    """
    return np.tanh(np.asarray(x, dtype=float))


def euclidean_distance(a, b):
    """
    Compute Euclidean distance between two vectors.
    Parameters
    a : np.ndarray, shape (n,)
    b : np.ndarray, shape (n,)
    Returns
    float
    Time complexity  : O(n)
    Space complexity : O(1)
    """
    a = np.asarray(a)
    b = np.asarray(b)
    return np.sqrt(np.sum((a - b) ** 2))


def manhattan_distance(a, b):
    """
    Compute Manhattan (L1) distance between two vectors.

    Parameters
    a : np.ndarray, shape (n,)
    b : np.ndarray, shape (n,)

    Returns
    float

    Time complexity  : O(n)
    Space complexity : O(1)
    """
    a = np.asarray(a)
    b = np.asarray(b)
    return np.sum(np.abs(a - b))


def cosine_similarity(a, b):
    """
    Compute cosine similarity between two vectors.

    Parameters
    a : np.ndarray, shape (n,)
    b : np.ndarray, shape (n,)

    Returns
    float, in [-1, 1]

    Raises
    ValueError
        If either vector has zero norm.

    Time complexity  : O(n)
    Space complexity : O(1)
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        raise ValueError("Cosine similarity is undefined for zero-norm vectors.")
    return np.dot(a, b) / (norm_a * norm_b)


def pairwise_euclidean(X, Y):
    """
    Compute pairwise Euclidean distances between two sets of vectors.

    Parameters
    X : np.ndarray, shape (n, d)
    Y : np.ndarray, shape (m, d)

    Returns
    np.ndarray, shape (n, m)
        Entry (i, j) is the distance between X[i] and Y[j].

    Time complexity  : O(n * m * d)
    Space complexity : O(n * m)
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    X_norm = np.sum(X ** 2, axis=1, keepdims=True)
    Y_norm = np.sum(Y ** 2, axis=1, keepdims=True).T
    dist = X_norm + Y_norm - 2 * np.dot(X, Y.T)
    return np.sqrt(np.maximum(dist, 0))


def topk_indices(values, k, largest=True):
    """
    Return indices of the top-k values without full sorting.

    Parameters
    values  : np.ndarray, shape (n,)
    k       : int, number of elements to select
    largest : bool, if True return largest k, else smallest k

    Returns
    np.ndarray, shape (k,)
        Indices of the top-k elements (unordered).

    Raises
    ValueError
        If k < 1 or k > len(values).

    Time complexity  : O(n)  average via np.argpartition
    Space complexity : O(k)
    """
    values = np.asarray(values)
    n = len(values)
    if k < 1 or k > n:
        raise ValueError(f"k must be between 1 and {n}, got {k}.")
    if largest:
        return np.argpartition(values, n - k)[n - k:]
    return np.argpartition(values, k)[:k]


def batch_iter(X, batch_size, y=None, shuffle=False, seed=None):
    """
    Yield mini-batches from X (and optionally y).

    Parameters
    X          : np.ndarray, shape (n, ...)
    batch_size : int
    y          : np.ndarray, shape (n,), optional
    shuffle    : bool, whether to shuffle before batching
    seed       : int or None, random seed for reproducibility

    Yields

    X_batch : np.ndarray, shape (batch_size, ...)
    y_batch : np.ndarray, shape (batch_size,)  — only if y is provided

    Raises
    ValueError
        If batch_size < 1.

    Time complexity  : O(n) per full pass
    Space complexity : O(batch_size)
    """
    X = np.asarray(X)
    n = X.shape[0]
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}.")

    indices = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    for start in range(0, n, batch_size):
        idx = indices[start:start + batch_size]
        if y is not None:
            yield X[idx], np.asarray(y)[idx]
        else:
            yield X[idx]