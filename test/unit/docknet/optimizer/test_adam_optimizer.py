import pytest
from numpy.testing import assert_array_almost_equal

from docknet.layer.dense_layer import DenseLayer
from docknet.layer.input_layer import InputLayer
from docknet.optimizer.adam_optimizer import AdamOptimizer
from test.unit.docknet.dummy_docknet import *


@pytest.fixture
def optimizer1():
    optimizer1 = AdamOptimizer(0.01)
    yield optimizer1


def test_optimizer(optimizer1):
    l0 = InputLayer(2)
    l1 = DenseLayer(2, 3, 'relu')
    l1.W = W1
    l1.b = b1
    layers = [l0, l1]

    gradients = [{
        'W': dJdW1,
        'b': dJdb1
    }]

    expected_optimized_W1 = optimized_W1
    expected_optimized_b1 = optimized_b1

    optimizer1.reset(layers)
    optimizer1.optimize(layers, gradients)

    assert_array_almost_equal(l1.params['W'], expected_optimized_W1, decimal=4)
    assert_array_almost_equal(l1.params['b'], expected_optimized_b1, decimal=4)
