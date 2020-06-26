from typing import Tuple

import numpy as np

from docknet.data_generator.data_generator import DataGenerator
from docknet.util.geometry import random_to_polar, polar_to_cartesian


class SwirlDataGenerator(DataGenerator):
    """
    The swirl data generator generates two classes (0 and 1) of 2D vectors distributed as 2 spirals with opposite phase
    that touch in their centers
    """

    def swirl(self, x: np.array, phase):
        """
        Generates individuals of a swirl with a given phase
        :param x: a random 2D array of scalars between 0 and 1
        :param phase: the swirl phase
        :return: a swirl individual
        """
        polar = random_to_polar(x)
        polar[1] = polar[1] * self.turns + phase
        distance_offset = polar[0] * self.arm_max_offset
        angle_offset = polar[1] - np.pi
        polar_offset = np.array([distance_offset, angle_offset])
        polar[0] = (polar[1] - phase) / (self.turns * 4 * np.pi)
        cartesian = polar_to_cartesian(polar)
        if (polar[1] - phase) < np.pi:
            polar_offset[0] *= (polar[1] - phase) / np.pi
        cartesian_offset = polar_to_cartesian(polar_offset)
        cartesian += cartesian_offset
        f = cartesian * self.scale + self.origin
        return f

    def swirl_phase_0(self, x: np.array):
        """
        Generates individuals of a swirl with phase 0
        :param x: a random 2D array of scalars between 0 and 1
        :param phase: the swirl phase
        :return: a swirl individual
        """
        f = self.swirl(x, 0.)
        return f

    def swirl_phase_180(self, x):
        """
        Generates individuals of a swirl with phase 180
        :param x: a random 2D array of scalars between 0 and 1
        :param phase: the swirl phase
        :return: a swirl individual
        """
        f = self.swirl(x, np.pi)
        return f

    def __init__(self, x0_range: Tuple[float, float], x1_range: Tuple[float, float]):
        """
        Initializes the island data data generator
        :param x0_range: tuple of minimum and maximum x values
        :param x1_range: tuple of minimum and maximum y values
        """
        super().__init__((self.swirl_phase_0, self.swirl_phase_180))
        self.arm_max_rel_offset = 0.5
        self.turns = 2
        self.arm_sep = 1. / (self.turns * 4)
        self.arm_max_offset = self.arm_max_rel_offset * self.arm_sep
        self.x_length = x0_range[1] - x0_range[0]
        self.y_length = x1_range[1] - x1_range[0]
        x_center = (x0_range[1] + x0_range[0]) / 2
        y_center = (x1_range[1] + x1_range[0]) / 2
        self.origin = np.array([x_center, y_center])
        self.scale = np.array([self.x_length, self.y_length])
