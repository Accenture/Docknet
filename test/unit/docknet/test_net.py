import io
import os
from typing import List

from numpy.testing import assert_array_almost_equal
import pytest

from docknet import net
from docknet.data_generator.data_generator import DataGenerator
from docknet.net import Docknet, read_pickle
from docknet.initializer.abstract_initializer import AbstractInitializer
from docknet.layer.abstract_layer import AbstractLayer
from docknet.optimizer.adam_optimizer import AdamOptimizer
from docknet.optimizer.gradient_descent_optimizer import (
    GradientDescentOptimizer)
from test.unit.docknet.dummy_docknet import *


data_dir = os.path.join(
    os.path.dirname(__file__), '..', '..', 'data', 'docknet')
temp_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'temp')

if not os.path.exists(temp_dir):
    os.mkdir(temp_dir)


class DummyInitializer(AbstractInitializer):
    def initialize(self, network_layers: List[AbstractLayer]):
        network_layers[1].params['W'] = W1
        network_layers[1].params['b'] = b1
        network_layers[2].params['W'] = W2
        network_layers[2].params['b'] = b2


class DummyDataGenerator(DataGenerator):
    @staticmethod
    def func0(x: np.ndarray) -> np.ndarray:
        return np.array([-5, -5])

    @staticmethod
    def func1(x: np.ndarray) -> np.ndarray:
        return np.array([5, 5])

    def __init__(self):
        super().__init__((self.func0, self.func1))


@pytest.fixture
def docknet1():
    docknet1 = Docknet()
    docknet1.add_input_layer(2)
    docknet1.add_dense_layer(3, 'relu')
    docknet1.add_dense_layer(1, 'sigmoid')
    docknet1.cost_function = 'cross_entropy'
    docknet1.initializer = DummyInitializer()
    docknet1.optimizer = GradientDescentOptimizer()
    yield docknet1


@pytest.fixture
def docknet2():
    docknet1 = Docknet()
    docknet1.add_input_layer(2)
    docknet1.add_dense_layer(3, 'relu')
    docknet1.add_dense_layer(1, 'sigmoid')
    docknet1.cost_function = 'cross_entropy'
    docknet1.initializer = DummyInitializer()
    docknet1.optimizer = AdamOptimizer()
    yield docknet1


def test_predict(docknet1):
    # Set network parameters as for the dummy initializer in order to enforce a
    # specific expected output
    docknet1.initializer.initialize(docknet1.layers)
    expected = Y_circ
    actual = docknet1.predict(X)
    assert_array_almost_equal(actual, expected)


def test_train(docknet1):
    docknet1.train(X, Y, batch_size=2, max_number_of_epochs=1)
    expected_optimized_W1 = optimized_W1
    expected_optimized_b1 = optimized_b1
    expected_optimized_W2 = optimized_W2
    expected_optimized_b2 = optimized_b2
    actual_optimized_W1 = docknet1.layers[1].params['W']
    actual_optimized_b1 = docknet1.layers[1].params['b']
    actual_optimized_W2 = docknet1.layers[2].params['W']
    actual_optimized_b2 = docknet1.layers[2].params['b']
    assert_array_almost_equal(actual_optimized_W1, expected_optimized_W1)
    assert_array_almost_equal(actual_optimized_b1, expected_optimized_b1)
    assert_array_almost_equal(actual_optimized_W2, expected_optimized_W2)
    assert_array_almost_equal(actual_optimized_b2, expected_optimized_b2)


def test_train_with_generated_data_then_predict(docknet1):
    np.random.seed(1)
    data_generator = DummyDataGenerator()
    samples, labels = data_generator.generate_balanced_shuffled_sample(8)
    docknet1.train(samples, labels, batch_size=4, max_number_of_epochs=40)
    predicted_labels = docknet1.predict(samples)
    predicted_labels = np.round(predicted_labels)
    assert_array_almost_equal(predicted_labels, labels)


def test_to_json(docknet1):
    # Set network parameters as for the dummy initializer in order to enforce a
    # specific expected output
    docknet1.initializer.initialize(docknet1.layers)
    expected_path = os.path.join(data_dir, 'docknet1.json')
    with open(expected_path, 'rt', encoding='UTF-8') as fp:
        expected = fp.read()
    actual_file = io.StringIO()
    docknet1.to_json(actual_file, True)
    actual = actual_file.getvalue()
    assert actual == expected


def test_read_json_to_json():
    expected_path = os.path.join(data_dir, 'docknet1.json')
    with open(expected_path, 'rt', encoding='UTF-8') as fp:
        expected_json = fp.read()
    actual_docknet = net.read_json(expected_path)
    actual_file = io.StringIO()
    actual_docknet.to_json(actual_file, True)
    actual_json = actual_file.getvalue()
    assert actual_json == expected_json


def test_read_pickle_to_json():
    expected_json_path = os.path.join(data_dir, 'docknet1.json')
    pickle_path = os.path.join(data_dir, 'docknet1.pkl')
    with open(expected_json_path, 'rt', encoding='UTF-8') as fp:
        expected_json = fp.read()
    actual_docknet = read_pickle(pickle_path)
    actual_file = io.StringIO()
    actual_docknet.to_json(actual_file, True)
    actual_json = actual_file.getvalue()
    assert actual_json == expected_json


def test_to_pickle_read_pickle_to_json(docknet1):
    # Set network parameters as for the dummy initializer in order to enforce a
    # specific expected output
    docknet1.initializer.initialize(docknet1.layers)
    pkl_path = os.path.join(temp_dir, 'docknet1.pkl')
    expected_json_path = os.path.join(data_dir, 'docknet1.json')
    with open(expected_json_path, 'rt', encoding='UTF-8') as fp:
        expected_json = fp.read()
    docknet1.to_pickle(pkl_path)
    docknet2 = read_pickle(pkl_path)
    actual_file = io.StringIO()
    docknet2.to_json(actual_file, True)
    actual_json = actual_file.getvalue()
    assert actual_json == expected_json
