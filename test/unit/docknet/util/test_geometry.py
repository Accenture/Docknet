import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest

from docknet.util.geometry import polar_to_cartesian, random_to_polar


random_to_polar_test_cases = [
    (np.array([0., 0.]), np.array([0., 0.])),
    (np.array([0.5, 0.5]), np.array([0.5, np.pi])),
    (np.array([1., 1.]), np.array([1., 2 * np.pi]))
]


@pytest.mark.parametrize('x, expected', random_to_polar_test_cases)
def test_random_to_polar(x: np.ndarray, expected: np.ndarray):
    actual = random_to_polar(x)
    assert_array_almost_equal(actual, expected, verbose=True)


polar_to_cartesian_test_cases = [
    (np.array([0., 0.]), np.array([0., 0.])),
    (np.array([1., 0.]), np.array([1., 0.])),
    (np.array([2., 2 * np.pi]), np.array([2., 0.])),
    (np.array([3., np.pi]), np.array([-3., 0.]))
]


@pytest.mark.parametrize('x, expected', polar_to_cartesian_test_cases)
def test_polar_to_cartesian(x: np.ndarray, expected: np.ndarray):
    actual = polar_to_cartesian(x)
    assert_array_almost_equal(actual, expected, verbose=True)
