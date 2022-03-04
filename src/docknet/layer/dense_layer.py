from typing import Dict, Tuple, Union

import numpy as np

from docknet.function.activation_function import get_activation_function
from docknet.layer.abstract_layer import AbstractLayer


class DenseLayer(AbstractLayer):
    def __init__(self, previous_layer_dimension: int, layer_dimension: int,
                 activation_function_name: str):
        """
        Initializes the dense layer, given the previous layer dimension, this
        layer dimension and the activation function to use
        :param previous_layer_dimension: the previous layer dimension (number
        of outputs)
        :param layer_dimension: this layer dimension (number of outputs)
        :param activation_function_name: the name of the activation function to
        use (e.g. 'sigmoid', 'relu', etc.)
        """
        params = {
            'W': np.zeros((layer_dimension, previous_layer_dimension)),
            'b': np.zeros((layer_dimension, 1))
        }
        super().__init__(layer_dimension, params)
        self.forward_activation_function, self.backward_activation_function = (
            get_activation_function(activation_function_name))

    def forward_linear(self, A_previous: np.ndarray) -> np.ndarray:
        """
        Computes the linear part of this layer's forward propagation
        Z = W * A_previous + b
        :param A_previous: activation of the previous layer, or X (the input
        data) if this is layer 1
        :return: Z = W * A_previous + b
        """
        Z = np.dot(self.W, A_previous) + self.b
        return Z

    def forward_activation(self, Z: np.ndarray) -> np.ndarray:
        """
        Computes the activation of this layer's forward propagation A = g(Z)
        :param Z: linear part of this layer
        :return: A = g(Z)
        """
        A = self.forward_activation_function(Z)
        return A

    def forward_propagate(self, A_previous: np.ndarray) -> np.ndarray:
        """
        Computes A, the output of this layer, given A_previous, the output of
        the previous layer
        :param A_previous: the output of the previous layer
        :return: A, the output of this layer
        """
        Z = self.forward_linear(A_previous)
        A = self.forward_activation(Z)
        return A

    def cached_forward_propagate(self, A_previous: np.ndarray) -> np.ndarray:
        """
        Performs the same computation as forward_propagate but also caches the
        values required to later perform the backward propagation, A_previous
        and Z
        :param A_previous: the output of the previous layer
        :return: A, the output of this layer
        """
        self.cached_A_previous = A_previous
        Z = self.forward_linear(A_previous)
        self.cached_Z = Z
        A = self.forward_activation(Z)
        return A

    def backward_linear(self, dJdZ: np.ndarray
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Given the partial derivative of the cost function J wrt Z, ∂J/∂Z,
        computes the following partial derivatives:
        1) ∂J/∂A_prev,
        2) ∂J/∂W, and
        3) ∂J/∂b
        :param dJdZ: ∂J/∂Z or partial derivative of the cost function J wrt Z,
        the linear part of this layer computed during the previous forward
        propagation
        :return: triplet of partial derivatives (∂J/∂A_prev, ∂J/∂W, ∂J/∂b)
        """
        m = self.cached_A_previous.shape[1]
        dJdW = np.dot(dJdZ, self.cached_A_previous.T) / m
        dJdb = np.sum(dJdZ, axis=1, keepdims=True) / m
        dJdA_prev = np.dot(self.W.T, dJdZ)
        return dJdA_prev, dJdW, dJdb

    def backward_activation(self, dJdA: np.ndarray) -> np.ndarray:
        """
        Given the partial derivative of the cost function J wrt A, ∂J/∂A, and
        the linear part of this layer computed and cached during the previous
        forward propagation, Z, computes the following partial derivatives:
        1) ∂J/∂A_prev,
        2) ∂J/∂W, and
        3) ∂J/∂b
        :param dJdA: ∂J/∂A or partial derivative of the cost function J wrt A,
        the activation values of this layer during the previous forward
        propagation
        :return: ∂J/∂Z, the partial derivative of the cost function wrt Z, the
        linear part of this layer during the previous forward propagation
        """
        dJdZ = dJdA * self.backward_activation_function(self.cached_Z)
        return dJdZ

    def backward_propagate(self, dJdA: Union[float, np.ndarray]
                           ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Given the gradient of the cost function J wrt A, ∂J/∂A, computes a
        tuple containing:
        1) ∂J/∂A_prev, the gradient of the cost function wrt the activation of
        the previous layer
        2) the gradients of J wrt each parameter of this layer, which are to be
        cached in the layer object
        :param dJdA: the gradient of the cost function wrt the activation of
        this layer
        :return: ∂J/∂A_prev and a dictionary with the gradients of J wrt the
        layer parameters W and b
        """
        dJdZ = self.backward_activation(dJdA)
        dJdA_prev, dJdW, dJdb = self.backward_linear(dJdZ)
        parameter_gradients = {
            'W': dJdW,
            'b': dJdb
        }
        return dJdA_prev, parameter_gradients

    def clean_cache(self):
        """
        Delete the layer cache (to be used once training in order to free
        resources, or before saving the Docknet as pickle in order not to save
        the cached values)
        """
        del self.cached_A_previous
        del self.cached_Z

    def to_dict(self) -> Dict[str, str]:
        """
        Converts this layer to a Python dictionary for JSON serialization
        (keeping the info needed to later create the layer back from the same
        Python dictionary)
        :return: a Python dictionary with all the info needed to create the
        layer back
        """
        d = super().to_dict()
        d['activation_function'] = self.forward_activation_function.__name__
        return d
