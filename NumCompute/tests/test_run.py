"""Basic smoke test for descriptive statistics."""
import numpy as np
from numcompute.stats import mean


def test_mean_smoke():
    assert np.isclose(mean(np.array([5, 10, 3])), 6.0)
