from typing import Tuple

import numpy as np

from docknet.data_generator.data_generator import DataGenerator


class ChessboardDataGenerator(DataGenerator):
    """
    The chessboard data generator generates two classes (0 and 1) of 2D vectors
    distributed as follows:

        0011
        0011
        1100
        1100
    """
    def func0(self, x: np.ndarray):
        """
        Generator function of 2D vectors of class 0 (top-left and bottom-right
        squares)
        :param x: a 2D random generated vector
        :return: the corresponding individual of class 0
        """
        f0 = x[0] * self.x_half_scale + self.x_min
        f1 = x[1] * self.y_scale + self.y_min
        if x[1] < 0.5:
            f0 += self.x_half_scale
        return np.array([f0, f1])

    def func1(self, x: np.ndarray):
        """
        Generator function of 2D vectors of class 1 (top-right and bottom-left
        squares)
        :param x: a 2D random generated vector
        :return: the corresponding individual of class 1
        """
        f0 = x[0] * self.x_scale + self.x_min
        f1 = x[1] * self.y_half_scale + self.y_min
        if x[0] >= 0.5:
            f1 += self.y_half_scale
        return np.array([f0, f1])

    def __init__(self, x0_range: Tuple[float, float],
                 x1_range: Tuple[float, float]):
        """
        Initializes the chessboard data generator
        :param x0_range: tuple of minimum and maximum x values
        :param x1_range: tuple of minimum and maximum y values
        """
        super().__init__((self.func0, self.func1))
        self.x_scale = x0_range[1] - x0_range[0]
        self.x_min = x0_range[0]
        self.x_half_scale = self.x_scale / 2
        self.y_scale = x1_range[1] - x1_range[0]
        self.y_min = x1_range[0]
        self.y_half_scale = self.y_scale / 2
