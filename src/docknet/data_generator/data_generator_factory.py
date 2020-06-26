from typing import Dict, Tuple, Type

from docknet.data_generator.chessboard_data_generator import ChessboardDataGenerator
from docknet.data_generator.cluster_data_generator import ClusterDataGenerator
from docknet.data_generator.data_generator import DataGenerator
from docknet.data_generator.island_data_generator import IslandDataGenerator
from docknet.data_generator.swirl_data_generator import SwirlDataGenerator

# The dictionary of data generators, to be able to retrieve them by name
data_generators: Dict[str, Type[DataGenerator]] = {
    'chessboard': ChessboardDataGenerator,
    'cluster': ClusterDataGenerator,
    'island': IslandDataGenerator,
    'swirl': SwirlDataGenerator
}


class UnknownDataGeneratorName(Exception):
    def __init__(self, data_generator_name: str):
        message = "Unknown data generator name {}".format(data_generator_name)
        super().__init__(message)


def make_data_generator(data_generator_name: str, x0_range: Tuple[float, float], x1_range: Tuple[float, float]):
    """
    Given the name of a data generator and the ranges of the vectors they are to generate, create and return the
    corresponding data generator instance
    :param data_generator_name: the name of the data generator
    :param x0_range: the x0 range as a tuple (min, max)
    :param x1_range: the x1 range as a tuple (min, max)
    :return: an instance of the corresponding data generator
    """
    try:
        generator_class = data_generators[data_generator_name](x0_range, x1_range)
    except KeyError:
        raise UnknownDataGeneratorName(data_generator_name)
    return generator_class
