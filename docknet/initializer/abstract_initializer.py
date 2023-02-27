from abc import ABC, abstractmethod
from typing import List

from docknet.layer.abstract_layer import AbstractLayer


class AbstractInitializer(ABC):
    @abstractmethod
    def initialize(self, layers: List[AbstractLayer]):
        """
        Initializes the parameters of the passed layers
        :param layers: a list of layers
        """
        pass
