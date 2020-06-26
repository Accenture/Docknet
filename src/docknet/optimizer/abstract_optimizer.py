from abc import ABC, abstractmethod
from typing import List, Tuple, Dict

import numpy as np

from docknet.layer.abstract_layer import AbstractLayer


class AbstractOptimizer(ABC):
    @abstractmethod
    def reset(self, layers: List[AbstractLayer]):
        """
        Perform any required initialization of this optimizer before starting to train, given the network parameters.
        :param layers: the list of network layers
        """
        pass

    @abstractmethod
    def optimize(self, network_layers: List[AbstractLayer], network_gradients: List[Dict[str, np.array]]):
        """
        Optimize the network parameters, given the gradients of the cost function wrt the network parameters for one
        given batch during one training iteration
        :param network_layers: the list of network layers
        :param network_gradients: the network parameter gradients as a list of dictionaries, one dictionary per layer,
        where each dictionary contains key/value pairs (k, v) with k the name of a layer parameter and v the
        the corresponding gradient
        """
        pass
