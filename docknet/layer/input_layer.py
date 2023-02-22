from typing import Dict, Tuple

import numpy as np

from docknet.layer.abstract_layer import AbstractLayer
from docknet.exception import IncompatibleSizeArrayException


class InputLayer(AbstractLayer):
    def __init__(self, dimension: int):
        super().__init__(dimension, {})

    def forward_propagate(self, X: np.ndarray) -> np.ndarray:
        """
        Forward propagation computation of the input layer: it just checks that
        the size of the input vectors is correct; otherwise it raises an
        IncompatibleSizeArrayException.
        :param X: 2-dimensional array where each column is an input vector
        :return:
        """
        if X.shape[0] != self.dimension:
            raise IncompatibleSizeArrayException(X.shape[1], self.dimension)
        return X

    def cached_forward_propagate(self, X: np.ndarray) -> np.ndarray:
        """
        Equivalent to the non-cached version of the forward propagate method,
        since the input layer does not need to cache anything for backward
        propagation
        :param X:
        :return:
        """
        return self.forward_propagate(X)

    def backward_propagate(self, X: np.ndarray
                           ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Returns an empty array and empty dictionary of gradients since this
        layer has no parameters and does not require to pass a derivative to a
        previous layer to continue with the backward propagation
        :param X:
        :return:
        """
        return np.array([]), {}

    def clean_cache(self) -> None:
        """
        Does nothing since the input layer does not require a cache
        """
        pass
