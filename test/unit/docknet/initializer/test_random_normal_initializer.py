from numpy.testing import assert_array_almost_equal
import pytest

from docknet.initializer.random_normal_initializer import (
    RandomNormalInitializer)
from docknet.layer.dense_layer import DenseLayer
from docknet.layer.input_layer import InputLayer
from test.unit.docknet.dummy_docknet import *


@pytest.fixture
def initializer1():
    initializer = RandomNormalInitializer(mean=0., stddev=0.05)
    yield initializer


def test_initialize(initializer1):
    np.random.seed(1)
    expected_W1 = [[0.08121727, -0.03058782],
                   [-0.02640859, -0.05364843],
                   [0.04327038, -0.11507693]]
    expected_b1 = [[0.08724059], [-0.03806035], [0.01595195]]
    l0 = InputLayer(2)
    l1 = DenseLayer(2, 3, 'relu')
    layers = [l0, l1]
    initializer1.initialize(layers)
    assert l0.params == {}
    assert_array_almost_equal(l1.params['W'], expected_W1)
    assert_array_almost_equal(l1.params['b'], expected_b1)
