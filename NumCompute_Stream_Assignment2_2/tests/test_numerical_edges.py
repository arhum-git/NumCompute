"""Additional numerical-stability and deterministic-selection tests."""
import numpy as np
import pytest
from numcompute.utils import sigmoid, logsumexp, softmax
from numcompute.sort_search import topk
from numcompute.optim import grad, jacobian

def test_sigmoid_extreme_no_warnings():
    with np.errstate(all='raise'):
        np.testing.assert_array_equal(sigmoid(np.array([-1000., 1000.])), [0., 1.])

def test_logsumexp_negative_infinity():
    assert np.isneginf(logsumexp(np.array([-np.inf, -np.inf])))

def test_logsumexp_positive_infinity():
    assert np.isposinf(logsumexp(np.array([np.inf, 0.])))

def test_softmax_negative_infinity():
    with pytest.raises(ValueError, match='undefined'):
        softmax(np.array([-np.inf, -np.inf]))

def test_softmax_positive_infinity():
    np.testing.assert_allclose(softmax(np.array([np.inf, np.inf, 1.])), [0.5, 0.5, 0.])

def test_topk_ties():
    values, indices = topk(np.array([5, 5, 5, 5]), 2)
    np.testing.assert_array_equal(indices, [0, 1])

@pytest.mark.parametrize('h', [0, -1, np.inf, np.nan])
def test_gradient_rejects_invalid_steps(h):
    with pytest.raises(ValueError):
        grad(lambda x: x[0] ** 2, np.array([1.]), h=h)

def test_jacobian_rejects_invalid_step():
    with pytest.raises(ValueError):
        jacobian(lambda x: x ** 2, np.array([1.]), h=0)
