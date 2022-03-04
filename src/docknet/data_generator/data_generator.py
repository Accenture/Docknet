from typing import Callable, Tuple

import numpy as np


class DataGenerator(object):
    def __init__(self, class_functions: Tuple[Callable, Callable]):
        """
        Initializes this data generator, given a list of functions, one per
        class, that return points of each class for a given pair of random
        numbers between 0 and 1
        :param class_functions: list of class generator functions
        """
        self.class_functions = class_functions

    def generate_class_sample(self, class_number: int, size: int
                              ) -> np.ndarray:
        """
        Generates a sample of individuals of the given class and size
        :param class_number: the class for which individuals are to be randomly
        generated
        :param size: the sample size
        :return: a Numpy array with the generated sample, one column per
        individual
        """
        random_values = np.random.rand(2, size)
        samples = np.apply_along_axis(self.class_functions[class_number], 0,
                                      random_values)
        return samples

    def generate_balanced_shuffled_sample(self, size: int
                                          ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates a shuffled balanced sample of individuals of both classes, 0
        and 1, with their corresponding labels
        :param size: the sample size
        :return: 2 Numpy arrays X and Y of shapes (2, sample size) and
        (1, sample size), respectively, the former containing the individuals
        (one per column) and the later the labels
        """
        class_0_size = round(size / 2)
        class_1_size = size - class_0_size
        X0 = self.generate_class_sample(0, class_0_size)
        X1 = self.generate_class_sample(1, class_1_size)
        Y0 = np.zeros((1, class_0_size))
        Y1 = np.ones((1, class_1_size))
        sample0 = np.concatenate([X0, Y0], axis=0)
        sample1 = np.concatenate([X1, Y1], axis=0)
        sample = np.concatenate([sample0, sample1], axis=1)
        np.random.shuffle(np.transpose(sample))
        X = sample[:-1, :]
        Y = sample[-1:, :]
        return X, Y
