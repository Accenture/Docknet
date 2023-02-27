import numpy as np


def random_to_polar(x: np.ndarray) -> np.ndarray:
    """
    Converts an array of 2 random numbers between 0 and 1 into a polar
    coordinate with distance between 0 and 1 and angle between 0 and 2 * pi
    radians
    :param x: an array [x0, x1] of 2 random numbers
    :return: the corresponding polar coordinate [distance, angle]
    """
    magnitude = x[0]
    angle = x[1] * 2 * np.pi
    return np.array([magnitude, angle])


def polar_to_cartesian(x: np.ndarray) -> np.ndarray:
    """
    Converts a polar coordinate into the corresponding 2D cartesian coordinate
    :param x: an array [distance, angle] representing a polar coordinate
    :return: the corresponding 2D cartesian coordinate [x0, x1]
    """
    cartesian = np.array([x[0] * np.cos(x[1]), x[0] * np.sin(x[1])])
    return cartesian
