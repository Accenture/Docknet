from typing import Callable, Dict, Tuple, Union

import numpy as np


def sigmoid(X: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Sigmoid activation function σ(X) = 1 / /1 + e^(-x))
    :param X: A scalar or numpy array of any size.
    :return: σ(X) = 1 / (1 + e^(-x))
    """
    return 1 / (1 + np.exp(-X))


def sigmoid_prime(X: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Derivative of the sigmoid function σ'(X) = σ(X) * (1 - σ(X))
    :param X: A scalar or numpy array of any size
    :return: σ'(X) = σ(X) * (1 - σ(X))
    """
    s = sigmoid(X)
    return s * (1 - s)


def relu(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    ReLU activation function implementation relu(X) = max(0, X)
    :param x: A scalar or numpy array of any size
    :return: relu(X) = max(0, x)
    """
    return np.maximum(0, x)


def relu_prime(X: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Derivative of the relu function relu'(X) = 0 if x <= 0, 1 if x > 0
    :param X: A scalar or numpy array of any size
    :return: σ'(X) = σ(X) * (1 - σ(X))
    """
    return X > 0


def tanh(X: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    tanh activation function tanh(X) = (e^x - e^(-x)) / (e^x + e^(-x))
    :param X: A scalar or numpy array of any size
    :return: tanh(X) = (e^X - e^(-X)) / (e^X + e^(-X))
    """
    e_x = np.exp(X)
    e_minus_x = np.exp(-X)
    return (e_x - e_minus_x) / (e_x + e_minus_x)


def tanh_prime(X: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Derivative of the tanh function tanh'(X) = 1 - (tanh(X))^2
    :param X: A scalar or numpy array of any size
    :return: tanh'(X) = 1 - (tanh(X))^2
    """
    t = tanh(X)
    return 1 - t * t


# The dictionary of activation functions, to be able to retrieve them by name
activation_functions: Dict[str, Tuple[Callable, Callable]] = {
    sigmoid.__name__: (sigmoid, sigmoid_prime),
    relu.__name__: (relu, relu_prime),
    tanh.__name__: (tanh, tanh_prime)
}


class UnknownActivationFunctionName(Exception):
    def __init__(self, activation_function_name: str):
        message = (
            f'Unknown activation function name {activation_function_name}')
        super().__init__(message)


def get_activation_function(activation_function_name: str):
    """
    Given the name of an activation function, retrieve the corresponding
    function and its derivative
    :param activation_function_name: the name of the activation function
    :return: the corresponding activation function and its derivative
    """
    try:
        return activation_functions[activation_function_name]
    except KeyError:
        raise UnknownActivationFunctionName(activation_function_name)
