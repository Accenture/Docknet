from typing import Tuple

import numpy as np

from docknet.data_generator.data_generator import DataGenerator
from docknet.util.geometry import polar_to_cartesian, random_to_polar


class IslandDataGenerator(DataGenerator):
    """
    The chessboard data generator generates two classes (0 and 1) of 2D vectors distributed as follows:

         111
        1   1
        1 0 1
        1   1
         111
    """

    def island(self, x: np.array):
        """
        Generator function of 2D vectors of class 0 (the island in the center)
        :param x: a 2D random generated vector
        :return: the corresponding individual of class 0
        """
        cartesian = polar_to_cartesian(random_to_polar(x))
        f = cartesian * self.island_radius + self.island_origin
        return f

    def sea(self, x: np.array):
        """
        Generator function of 2D vectors of class 0 (the ring around the island)
        :param x: a 2D random generated vector
        :return: the corresponding individual of class 0
        """
        polar = random_to_polar(x)
        polar[0] = polar[0] * self.sea_width + self.sea_inner_diameter
        cartesian = polar_to_cartesian(polar) * self.sea_scale + self.island_origin
        return cartesian

    def __init__(self, x0_range: Tuple[float, float], x1_range: Tuple[float, float]):
        """
        Initializes the island data data generator
        :param x0_range: tuple of minimum and maximum x values
        :param x1_range: tuple of minimum and maximum y values
        """
        super().__init__((self.island, self.sea))
        x_center = (x0_range[1] + x0_range[0]) / 2
        y_center = (x1_range[1] + x1_range[0]) / 2
        x_length = x0_range[1] - x0_range[0]
        y_length = x1_range[1] - x1_range[0]
        self.island_origin = np.array([x_center, y_center])
        self.island_radius = np.array([x_length / 6, y_length / 6])
        self.sea_width = 1/3
        self.sea_inner_diameter = 2/3
        self.sea_scale = np.array([x_length / 2, y_length / 2])
