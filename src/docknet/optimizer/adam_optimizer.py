from typing import Dict, List, Optional, Tuple

import numpy as np

from docknet.layer.abstract_layer import AbstractLayer


class AdamOptimizer(object):
    def __init__(self, learning_rate: float = 0.01, beta1: float = 0.9,
                 beta2: float = 0.999, epsilon: float = 1e-8):
        """
        Builds a new Adam Optimizer storing the optimizer parameters and
        defining additional parameters to be used during the optimization
        process
        """

        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.network_params: Optional[List[Tuple]] = None
        self.network_v: List[Dict[str, np.ndarray]] = []
        self.network_s: List[Dict[str, np.ndarray]] = []
        self.t = 0

    def reset(self, layers: List[AbstractLayer]):
        """
        Initializes Adam's v and s parameters as 0 for each network parameter,
        and Adam's counter t as 0
        :param layers: the list of network layers
        """
        self.network_v.clear()
        self.network_s.clear()
        for layer in layers:
            layer_v = {name: np.zeros(value.shape)
                       for name, value in layer.params.items()}
            layer_s = {name: np.zeros(value.shape)
                       for name, value in layer.params.items()}
            self.network_v.append(layer_v)
            self.network_s.append(layer_s)
        self.t = 1

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
        for p, dp, v, s in zip([layer.params for layer in network_layers],
                               network_gradients, self.network_v,
                               self.network_s):
            # For each param in the layer
            for k in p.keys():
                # Compute moving average of the gradients
                v[k] = self.beta1 * v[k] + (1 - self.beta1) * dp[k]
                # Compute bias-corrected first moment estimate
                v_corrected = v[k] / (1 - self.beta1 ** self.t)
                # Compute moving average of the squared gradients
                s[k] = self.beta2 * s[k] + (1 - self.beta2) * dp[k]**2
                # Compute bias-corrected second raw moment estimate
                s_corrected = s[k] / (1 - self.beta2 ** self.t)
                # Update parameter
                p[k] = p[k] - self.learning_rate * v_corrected / (
                    np.sqrt(s_corrected + self.epsilon))
