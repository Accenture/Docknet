from typing import Union

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from docknet.function.activation_function import relu, sigmoid, tanh


sigmoid_test_cases = [
    (np.array([-100., 0., 100]), np.array([0., 0.5, 1.])),
    (-100., 0.),
    (0., 0.5),
    (100., 1.),
    (np.array([0.]), np.array([0.5])),
]


@pytest.mark.parametrize("x, expected", sigmoid_test_cases)
def test_sigmoid(x: Union[float, np.array], expected: Union[float, np.array]):
    actual  = sigmoid(x)
    assert_array_almost_equal(actual, expected, verbose=True)


relu_test_cases = [
    (-1., 0.),
    (0., 0.),
    (1., 1.),
    (5., 5.),
    (np.array(0.), np.array(0.)),
    (np.array([-1., 0., 1., 5.]), np.array([0., 0., 1., 5.]))
]


@pytest.mark.parametrize("x, expected", relu_test_cases)
def test_relu(x: Union[float, np.array], expected: Union[float, np.array]):
    actual = relu(x)
    assert_array_almost_equal(actual, expected, verbose=True)


tanh_test_cases = [
    (-100., -1.),
    (0., 0.),
    (100., 1.),
    (np.array(0.), np.array(0.)),
    (np.array([-100., 0., 100.]), np.array([-1., 0., 1.]))
]


@pytest.mark.parametrize("x, expected", tanh_test_cases)
def test_tanh(x: Union[float, np.array], expected: Union[float, np.array]):
    actual = tanh(x)
    assert_array_almost_equal(actual, expected, verbose=True)
