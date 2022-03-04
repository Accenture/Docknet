from typing import Dict, List, Tuple

import numpy as np

from docknet.layer.abstract_layer import AbstractLayer
from docknet.optimizer.abstract_optimizer import AbstractOptimizer


class GradientDescentOptimizer(AbstractOptimizer):
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        self.network_params: List[Tuple]

    def reset(self, network_layers: List[AbstractLayer]):
        """
        Initialization of gradient descent optimizer before starting a
        training: nothing is required to be done
        :param network_layers: the list of network layers
        """
        pass

    def optimize(self, network_layers: List[AbstractLayer],
                 network_gradients: List[Dict[str, np.ndarray]]):
        """
        Optimize the network parameters, given the gradients of the cost
        function wrt the network parameters for one given batch during one
        training iteration
        :param network_layers: the list of network layers
        :param network_gradients: the network parameter gradients as a list of
        dictionaries, one dictionary per layer, where each dictionary contains
        key/value pairs (k, v) with k the name of a layer parameter and v the
        corresponding gradient
        """
        # For each layer except the input layer (since no gradients are
        # computed for the input layer)
        for p, dJdp in zip([layer.params for layer in network_layers],
                           network_gradients):
            # For each param in the layer
            for k in p.keys():
                p[k] = p[k] - self.learning_rate * dJdp[k]
