import matplotlib.pyplot as plt
import numpy as np
import pytest

from docknet.data_generator.chessboard_data_generator import (
    ChessboardDataGenerator)
from test.unit.docknet.data_generator.plot_utils import plot_scatter


x_range = (0., 10.)
y_range = (-5., 0.)


@pytest.fixture
def data_generator1():
    generator = ChessboardDataGenerator(x_range, y_range)
    yield generator


def test_generate_class_sample(data_generator1):
    np.random.seed(1)
    size = 50
    sample0 = data_generator1.generate_class_sample(0, size)
    sample1 = data_generator1.generate_class_sample(1, size)
    fig, axes = plt.subplots(1, 2)
    plot_scatter(axes[0], sample0[0, :], sample0[1, :], (1, 0, 0), x_range,
                 y_range, 'Chessboard class 0')
    plot_scatter(axes[1], sample1[0, :], sample1[1, :], (0, 0, 1), x_range,
                 y_range, 'Chessboard class 1')
    assert sample0.shape == (2, size)
    assert sample1.shape == (2, size)


def test_generate_sample(data_generator1):
    size = 100
    X, Y = data_generator1.generate_balanced_shuffled_sample(size)
    axe = plt.subplot()
    plot_scatter(axe, X[0, :], X[1, :], Y[0, :], x_range, y_range,
                 'Chessboard sample')
    assert X.shape == (2, size)
    assert Y.shape == (1, size)
