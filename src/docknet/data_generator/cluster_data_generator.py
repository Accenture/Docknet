from typing import Tuple

import numpy as np

from docknet.data_generator.data_generator import DataGenerator
from docknet.util.geometry import random_to_polar, polar_to_cartesian


class ClusterDataGenerator(DataGenerator):
    """
    The cluster data generator generates two classes (0 and 1) of 2D vectors distributed as follows:

        0XX
        XXX
        XX1
    """

    def unitary_cluster(self, x: np.array):
        polar = random_to_polar(x)
        polar[0] = 11.**polar[0] / 10. - 0.6
        f = polar_to_cartesian(polar)
        return f

    def func0(self, x: np.array):
        """
        Generator function of 2D vectors of class 0 (the upper-left cluster)
        :param x: a 2D random generated vector
        :return: the corresponding individual of class 0
        """
        f = self.unitary_cluster(x) * self.cluster_diameter + self.cluster0_origin
        return f

    def func1(self, x: np.array):
        """
        Generator function of 2D vectors of class 1 (the bottom-right cluster)
        :param x: a 2D random generated vector
        :return: the corresponding individual of class 1
        """
        f = self.unitary_cluster(x) * self.cluster_diameter + self.cluster1_origin
        return f

    def __init__(self, x0_range: Tuple[float, float], x1_range: Tuple[float, float]):
        """
        Initializes the cluster data generator
        :param x0_range: tuple of minimum and maximum x values
        :param x1_range: tuple of minimum and maximum y values
        """
        super().__init__((self.func0, self.func1))
        x_length = x0_range[1] - x0_range[0]
        y_length = x1_range[1] - x1_range[0]
        x_center = (x0_range[0] + x0_range[1]) / 2
        y_center = (x1_range[0] + x1_range[1]) / 2
        self.cluster0_origin = np.array([x_center - x_length / 3, y_center + y_length / 3])
        self.cluster1_origin = np.array([x_center + x_length / 3, y_center - y_length / 3])
        self.cluster_diameter = np.array([x_length / 3, y_length / 3])
