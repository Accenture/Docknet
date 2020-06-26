from abc import ABC, abstractmethod
from typing import Union, Dict, Tuple

import numpy as np


class AbstractLayer(ABC):
    def __init__(self, dimension: int, params: Dict[str, np.array]):
        """
        Initializes the layer, given a dimension and a dictionary of parameters
        :param dimension: the layer dimension (number of outputs)
        :param params: the dictionary of layer parameters
        """
        # Initialize '_p', the layer's dictionary of parameters, bypassing method __setattr__; note that since the
        # parameter dictionary doesn't exist yet, __setattr__ would fail if not bypassed since it will try to access the
        # '_p' to check if '_p' is a layer parameter
        self.__dict__['_p'] = params
        self._dimension = dimension
        self._something = "asdf"


    @property
    def dimension(self) -> int:
        """
        Getter of the layer dimension (number of outputs)
        :return: the layer dimension
        """
        return self._dimension

    @property
    def params(self) -> Dict[str, np.array]:
        """
        Getter of the layer parameters
        :return: a dictionary of layer parameters
        """
        return self._p

    @params.setter
    def params(self, p: Dict[str, np.array]):
        self._p = p

    def __getattr__(self, attr: str):
        """
        Let l be a layer object and p be a parameter name of l, this method allows for getting p's value with notation
            l.p
        apart from the more cumbersome notation
            l.params['p']
        """
        # This function is called if attr is not a member of the class, so we simply try to get that attribute from the
        # dictionary of parameters. If neither found there, then throw the standard exception
        try:
            return self.__dict__['_p'][attr]
        except KeyError:
            raise AttributeError("'{}' object has no attribute '{}'".format(self.__class__.__name__, attr))

    def __setattr__(self, attr: str, value):
        """
        Let l be a layer object, p be a parameter name of l, and v the value to assign to p, this method allows for
        setting p's value with notation
            l.p = v
        apart from the more cumbersome notation
            l.params['p'] = v
        """
        # If attribute name is a key in the dictionary of parameters of this layer, assign it to that parameter key
        if '_p' in self.__dict__ and attr in self._p:
            self._p[attr] = value
        # Otherwise proceed as usual
        else:
            super().__setattr__(attr, value)

    @abstractmethod
    def forward_propagate(self, A_previous) -> np.array:
        """
        Computes A, the output of this layer, given A_previous, the output of the previous layer
        :param A_previous: the output of the previous layer
        :return: A, the output of this layer
        """
        pass

    @abstractmethod
    def cached_forward_propagate(self, A_previous) -> np.array:
        """
        Performs the same computation than forward_propagate but also caches the values required to later
        perform the backward propagation
        :param A_previous: the output of the previous layer
        :return: A, the output of this layer
        """
        pass

    @abstractmethod
    def backward_propagate(self, dJdA: Union[float, np.array]) -> Tuple[np.array, Dict[str, np.array]]:
        """
        Given the gradient of the cost function J wrt A, ∂J/∂A, computes a tuple containing:
        1) ∂J/∂A_prev, the gradient of the cost function wrt the activation of the previous layer
        2) the gradients of J wrt each parameter of this layer
        :param dJdA: the gradient of the cost function wrt the activation of this layer
        :return: ∂J/∂A_prev and a dictionary with the gradients of J wrt the layer parameters
        """
        pass

    @abstractmethod
    def clean_cache(self) -> None:
        """
        Delete the layer cache (to be used once training in order to free resources, or before saving the docknet as
        pickle in order not to save the cached values)
        """
        pass

    def to_dict(self) -> Dict:
        d = {
            'dimension': self.dimension,
            'params': self.params,
            'type': self.__class__.__name__[:-5].lower()
        }
        return d

