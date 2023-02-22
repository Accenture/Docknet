from typing import Callable, Dict, Tuple

import numpy as np


def cross_entropy(Y_circ: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Cross entropy cost function
    :param Y_circ: The actual network output
    :param Y: The expected network output
    :return: the cross entropy cost
    """
    # Note we use here np.nan_to_num to repair nans returned when computing
    # log(0)
    logprobs = np.multiply(Y, -np.nan_to_num(np.log(Y_circ))) + np.multiply(
        1. - Y, -np.nan_to_num(np.log(1. - Y_circ)))
    cost = np.sum(logprobs)
    return cost


def dcross_entropy_dYcirc(Y_circ: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Partial derivative of the cross entropy cost function wrt Y_circ
    :param Y_circ: The actual network output
    :param Y: The expected network output
    :return: the cross entropy cost
    """
    # Note we use here np.nan_to_num to repair infs returned when computing Y/0
    dJdY_circ = - (np.nan_to_num(np.divide(Y, Y_circ))
                   - np.nan_to_num(np.divide(1 - Y, 1 - Y_circ)))
    return dJdY_circ


# Dictionary of cost functions and their derivatives, to be able to retrieve
# them by name
cost_functions: Dict[str, Tuple[Callable, Callable]] = {
    cross_entropy.__name__: (cross_entropy, dcross_entropy_dYcirc),
}


class UnknownCostFunctionName(Exception):
    def __init__(self, cost_function_name: str):
        message = f'Unknown cost function name {cost_function_name}'
        super().__init__(message)


def get_cost_function(cost_function_name: str):
    """
    Given the name of a cost function, retrieve the corresponding function and
    its partial derivative wrt Y_circ
    :param cost_function_name: the name of the cost function
    :return: the corresponding cost function and its partial derivative wrt
    Y_circ
    """
    try:
        return cost_functions[cost_function_name]
    except KeyError:
        raise UnknownCostFunctionName(cost_function_name)
