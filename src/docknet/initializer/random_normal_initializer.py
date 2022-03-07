from typing import List

import numpy as np

from docknet.initializer.abstract_initializer import AbstractInitializer
from docknet.layer.abstract_layer import AbstractLayer


class RandomNormalInitializer(AbstractInitializer):
    """
    Random normal initializer sets all network parameters randomly using a
    normal distribution with a given mean and
    standard deviation
    """
    def __init__(self, mean: float = 0.0, stddev: float = 0.05):
        """
        Initialize the random normal initializer, given a mean a standard
        deviation
        :param mean: the mean of the normal distribution
        :param stddev: the standard deviation of the normal distribution
        """
        self.mean = mean
        self.stddev = stddev

    def initialize(self, network_layers: List[AbstractLayer]):
        """
        Initializes the parameters of the passed layers
        :param network_layers: a list of layers
        """
        # For each layer
        for p in [layer.params for layer in network_layers]:
            # For each parameter
            for k in p.keys():
                # Randomly initialize the parameter
                p[k] = np.random.randn(*p[k].shape) * self.stddev + self.mean
