import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from docknet.util.geometry import random_to_polar, polar_to_cartesian

random_to_polar_test_cases = [
    (np.array([0., 0.]), np.array([0., 0.])),
    (np.array([0.5, 0.5]), np.array([0.5, np.pi])),
    (np.array([1., 1.]), np.array([1., 2 * np.pi]))
]


@pytest.mark.parametrize("x, expected", random_to_polar_test_cases)
def test_random_to_polar(x: np.array, expected: np.array):
    actual = random_to_polar(x)
    assert_array_almost_equal(actual, expected, verbose=True)


polar_to_cartesian_test_cases = [
    (np.array([0., 0.]), np.array([0., 0.])),
    (np.array([1., 0.]), np.array([1., 0.])),
    (np.array([2., 2 * np.pi]), np.array([2., 0.])),
    (np.array([3., np.pi]), np.array([-3., 0.]))
]


@pytest.mark.parametrize("x, expected", polar_to_cartesian_test_cases)
def test_polar_to_cartesian(x: np.array, expected: np.array):
    actual = polar_to_cartesian(x)
    assert_array_almost_equal(actual, expected, verbose=True)
