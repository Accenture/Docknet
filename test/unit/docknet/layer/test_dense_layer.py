from numpy.testing import assert_array_almost_equal
import pytest

from docknet.layer.dense_layer import DenseLayer
from test.unit.docknet.dummy_docknet import *


@pytest.fixture
def layer1() -> DenseLayer:
    l1 = DenseLayer(2, 3, 'relu')
    l1.W = W1
    l1.b = b1
    yield l1


def test_forward_linear(layer1):
    expected = Z1
    actual = layer1.forward_linear(X)
    assert_array_almost_equal(actual, expected, verbose=True)


def test_forward_activation(layer1):
    expected = A1
    actual = layer1.forward_activation(Z1)
    assert_array_almost_equal(actual, expected, verbose=True)


def test_forward_propagate(layer1):
    expected = A1
    actual = layer1.forward_propagate(X)
    assert_array_almost_equal(actual, expected, verbose=True)


def test_cached_forward_propagate(layer1):
    expected = A1
    actual = layer1.cached_forward_propagate(X)
    assert_array_almost_equal(actual, expected, verbose=True)
    assert_array_almost_equal(layer1.cached_A_previous, A0, verbose=True)
    assert_array_almost_equal(layer1.cached_Z, Z1, verbose=True)


def test_backward_linear(layer1):
    layer1.cached_A_previous = A0
    expected_dJdA_prev, expected_dJdW, expected_dJdb = dJdA0, dJdW1, dJdb1
    actual_dJdA_prev, actual_dJdW, actual_dJdb = layer1.backward_linear(dJdZ1)
    assert_array_almost_equal(actual_dJdA_prev, expected_dJdA_prev)
    assert_array_almost_equal(actual_dJdW, expected_dJdW)
    assert_array_almost_equal(actual_dJdb, expected_dJdb)


def test_backward_activation(layer1):
    layer1.cached_Z = Z1
    expected_dJdZ = dJdZ1
    actual_dJdZ = layer1.backward_activation(dJdA1)
    assert_array_almost_equal(actual_dJdZ, expected_dJdZ)


def test_backward_propagate(layer1):
    layer1.cached_A_previous = A0
    layer1.cached_Z = Z1
    expected_dJdA_prev = dJdA0
    expected_dJdW = dJdW1
    expected_dJdb = dJdb1
    actual_dJdA_prev, actual_parameter_gradients = layer1.backward_propagate(
        dJdA1)
    assert_array_almost_equal(actual_dJdA_prev, expected_dJdA_prev)
    assert_array_almost_equal(actual_parameter_gradients['W'], expected_dJdW)
    assert_array_almost_equal(actual_parameter_gradients['b'], expected_dJdb)
