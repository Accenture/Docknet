import sys

import numpy as np
import pytest

from docknet.function.cost_function import cross_entropy, dcross_entropy_dYcirc

cross_entropy_test_cases = [
    (np.array([[1., 0.]]), np.array([[1., 0.]]), 0.),
    (np.array([[1., 0.]]), np.array([[0., 1.]]), np.inf),
    (np.array([[1., 0.]]), np.array([[0.5, 0.]]), 0.6931471805599453),
    (np.array([[1., 0.]]), np.array([[1., 0.5]]), 0.6931471805599453),
    (np.array([[1., 0.]]), np.array([[0.5, 0.5]]), 1.3862943611198906),
]


@pytest.mark.parametrize("Y, Y_circ, expected", cross_entropy_test_cases)
def test_cross_entropy(Y: np.array, Y_circ: np.array, expected: float):
    # Disable Numpy warnings when trying to divide by 0 in the border cases, otherwise these tests produce warnings
    np.seterr(divide='ignore', over='ignore')
    actual = cross_entropy(Y_circ, Y)
    np.testing.assert_almost_equal(actual, expected)


dcross_entropy_dYcirc_test_cases = [
    (np.array([[1., 0.]]), np.array([[1., 0.]]), np.array([[-1., 1.]])),
    # Note sys.float_info.max returns the maximum float the machine can represent, which is an approximation of inf
    (np.array([[1., 0.]]), np.array([[0., 1.]]), np.array([[-sys.float_info.max, sys.float_info.max]])),
    (np.array([[1., 0.]]), np.array([[0.5, 0.]]), np.array([[-2., 1.]])),
    (np.array([[1., 0.]]), np.array([[1., 0.5]]), np.array([[-1., 2.]])),
    (np.array([[1., 0.]]), np.array([[0.5, 0.5]]), np.array([[-2., 2.]])),
]


@pytest.mark.parametrize("Y, Y_circ, expected", dcross_entropy_dYcirc_test_cases)
def test_dcross_entropy_dYcirc(Y: np.array, Y_circ: np.array, expected: np.array):
    # Disable Numpy warnings when trying to divide by 0 in the border cases, otherwise these tests produce warnings
    np.seterr(divide='ignore', invalid='ignore')
    actual = dcross_entropy_dYcirc(Y_circ, Y)
    np.testing.assert_array_almost_equal(actual, expected)
