import json
import math
import os
import pickle
import sys
from typing import List, Optional, Union, TextIO, BinaryIO

import numpy as np

from docknet.initializer.abstract_initializer import AbstractInitializer
from docknet.layer.abstract_layer import AbstractLayer
from docknet.function.cost_function import get_cost_function
from docknet.layer.dense_layer import DenseLayer
from docknet.layer.input_layer import InputLayer
from docknet.optimizer.abstract_optimizer import AbstractOptimizer
from docknet.util.notifier import Notifier


class DocknetJSONEncoder(json.JSONEncoder):
    """
    JSON encoder needed for serializing a docknet to JSON format; defines how to serialize special docknet classes such
    as the Docknet itself, the layers and Numpy arrays
    """
    def default(self, obj):
        if isinstance(obj, Docknet):
            return obj.layers
        elif isinstance(obj, AbstractLayer):
            return obj.to_dict()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super().default(obj)


class Docknet(object):
    """
    The Docknet class, an extensible implementation of neural networks comprising a sequence of layers, a parameter
    initializer, a cost function and its derivative, and a parameter optimizer. A Docknet instance is first to be
    created, then its methods add_XXX_layer invoked in the proper sequence in order to configure the network
    architecture. In order to train the network, the initializer, cost function and optimizer must be first set. After
    training, methods to_pickle and to_json can be used to save the network to a file. In case the docknet has
    previously been saved to a file, use methods read_pickle or read_json to create the Docknet.
    """
    def __init__(self, notifier=Notifier()):
        """
        Initializes the docknet as an empty network (no layers)
        :param notifier:
        """
        self.layers: List[AbstractLayer] = []
        self._cost_function_name: Optional[str] = None
        self._initializer: Optional[AbstractInitializer] = None
        self._optimizer: Optional[AbstractOptimizer] = None
        self.notifier = notifier

    def add_input_layer(self, dimension: int):
        """
        Add an input layer to this DockNet after the last layer; note a Docknet is supposed to have a single input layer
        as first layer
        :param dimension: input vector size
        """
        layer = InputLayer(dimension)
        self.layers.append(layer)

    def add_dense_layer(self, dimension: int, activation_function_name: str):
        """
        Add a dense layer to this DockNet after the last layer
        :param dimension: number of neurons of the layer to add
        :param activation_function_name: name of the activation function to use in this layer
        """
        layer = DenseLayer(self.layers[-1].dimension, dimension, activation_function_name)
        self.layers.append(layer)

    @property
    def initializer(self) -> AbstractInitializer:
        """
        Gets the network initializer
        :return: the network initializer
        """
        return self._initializer

    @initializer.setter
    def initializer(self, initializer: AbstractInitializer):
        """
        Sets the network parameter initializer; required for training only
        :param initializer: an initializer object (e.g. an instance of RandomNormalInitializer)
        :return:
        """
        self._initializer = initializer

    @property
    def cost_function(self) -> str:
        """
        Gets the networks's cost function name
        :return: the network's cost function name
        """
        return self._cost_function_name

    @cost_function.setter
    def cost_function(self, cost_function_name: str):
        """
        Sets the network cost function and its derivative, given a cost function name; required for training only
        :param cost_function_name: the cost function name (e.g. 'cross_entropy')
        """
        self._cost_function, self._cost_function_prime = get_cost_function(cost_function_name)

    @property
    def optimizer(self) -> AbstractOptimizer:
        """
        Gets the network parameter optimizer
        :return: the network parameter optimizer
        """
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer: AbstractOptimizer):
        """
        Sets the network's optimizer
        :param optimizer: the network's optimizer (e.g. an instance of GradicentDescentOptimizer)
        :return:
        """
        self._optimizer = optimizer

    def train_batch(self, X: np.array, Y: np.array) -> float:
        """
        Train the network for a batch of data
        :param X: 2-dimensional array of input vectors, one vector per column
        :param Y: 2-dimensional array of expected values to predict, one single row with same amount of columns than X
        :return: aggregated cost for the entire batch (without averaging)
        """
        A = X
        for layer in self.layers:
            A = layer.cached_forward_propagate(A)
        Y_circ = A
        J = self._cost_function(Y_circ, Y)
        dJdY_circ = self._cost_function_prime(Y_circ, Y)
        dJdA = dJdY_circ
        network_gradients = []
        for layer in reversed(self.layers):
            dJdA, layer_gradients = layer.backward_propagate(dJdA)
            network_gradients.insert(0, layer_gradients)
        self._optimizer.optimize(self.layers, network_gradients)
        return J

    def train(self, X: np.array, Y: np.array, batch_size: int, max_number_of_epochs: int, error_delta=0.,
              max_epochs_within_delta=-1, stop_file_pathname: str = None, initialize=True):
        """
        Train the network for a given set of input vectors and expected predictions up to reaching one of 2 stopping
        conditions:
        1) a maximum number of epochs is attained
        2) the error difference between 2 consecutive epochs is below a threshold for a given amount of epochs
        3) a file at a given location exists (create the file to manually stop the training)
        :param X: 2-dimensional array of input vectors, one vector per column
        :param Y: 2-dimensional array of expected values to predict, one single row with same amount of columns than X
        :param batch_size: amount of input vectors to with use per training iteration
        :param max_number_of_epochs: maximum number of epochs for stop condition 1
        :param error_delta: error difference threshold of stop condition 2
        :param max_epochs_within_delta: maximum number of epochs for stop condition 2
        :param stop_file_pathname: path of file that
        :param initialize: initialize parameters before starting to train
        :return lists of average error per epoch and per iteration
        """
        if max_number_of_epochs < 0:
            max_number_of_epochs = sys.maxsize
        if max_epochs_within_delta < 0:
            max_epochs_within_delta = sys.maxsize
        epoch = 1
        epochs_within_delta = 0
        batch_count = math.ceil(X.shape[1] / batch_size)
        if initialize:
            self.initializer.initialize(self.layers)
        self.optimizer.reset(self.layers)
        iteration_errors = []
        epoch_errors = []
        while epoch <= max_number_of_epochs and\
                epochs_within_delta <= max_epochs_within_delta and\
                not (stop_file_pathname and os.path.exists(stop_file_pathname)):
            batch_begin = 0
            epoch_error = 0.
            for i in range(batch_count - 1):
                batch_end = batch_begin + batch_size
                X_batch = X[:, batch_begin:batch_end]
                Y_batch = Y[:, batch_begin:batch_end]
                iteration_error = self.train_batch(X_batch, Y_batch)
                iteration_errors.append(iteration_error / X_batch.shape[1])
                epoch_error += iteration_error
                batch_begin = batch_end
            X_batch = X[:, batch_begin:]
            Y_batch = Y[:, batch_begin:]
            iteration_error = self.train_batch(X_batch, Y_batch)
            iteration_errors.append(iteration_error / X_batch.shape[1])
            epoch_error += iteration_error
            epoch_error /= X.shape[1]
            epoch_errors.append(epoch_error)
            self.notifier.info("Loss after epoch {}: {}".format(epoch, epoch_error))
            if epoch_error > error_delta:
                epochs_within_delta = 0
            else:
                epochs_within_delta += 1
            epoch += 1
        return epoch_errors, iteration_errors

    def predict(self, X: np.array):
        """
        Compute the predictions for a batch of input vectors
        :param X: 2-dimensional array of input vectors, one vector per column
        :return: 1-dimensional array with the computed predictions,
        """
        A = X
        for layer in self.layers:
            A = layer.forward_propagate(A)
        return A

    def to_pickle(self, pathname_or_file: Union[str, BinaryIO]):
        """
        Save the current network parameters to a pickle file; to be used after training so that the model can be later
        reused for making predictions without having to train the network again
        :param pathname_or_file: either a path to a pkl file or a file-like object
        """
        if isinstance(pathname_or_file, str):
            with open(pathname_or_file, 'wb') as fp:
                pickle.dump(self, fp)
        else:
            pickle.dump(self, pathname_or_file)

    def to_json(self, pathname_or_file: Union[str, TextIO], pretty_print=False):
        """
        Save the current network parameters to a json file. Intended for debugging/testing purposes. For making actually
        using the network for making predictions, use method to_pickle, with will save the parameters in a more
        efficient binary format
        :param pathname_or_file: either a path to a JSON file or a file-like object
        :param pretty_print: generate a well formatted JSON for manual review
        """
        kwargs = {'cls': DocknetJSONEncoder}
        if pretty_print:
            kwargs['indent'] = 4
            kwargs['sort_keys'] = True
        if isinstance(pathname_or_file, str):
            with open(pathname_or_file, 'wb', encoding='UTF-8') as fp:
                json.dump(self, fp, **kwargs)
        else:
            json.dump(self, pathname_or_file, **kwargs)


def read_pickle(pathname: str) -> Docknet:
    """
    Create a new DockNet initialized with previously saved parameters in pickle format
    :param pathname: path and name of the pickle file
    :return: the initialized DockNet
    """
    with open(pathname, 'rb') as fp:
        docknet = pickle.load(fp)
    return docknet


def read_json(pathname: str) -> Docknet:
    """
    Create a new DockNet initialized with previously saved parameters in json format
    :param pathname: path and name of the json file
    :return: the initialized DockNet
    """
    with open(pathname, 'rb') as fp:
        layers_description = json.load(fp, encoding='UTF-8')
    docknet = Docknet()
    for desc in layers_description:
        if desc['type'] == 'input':
            docknet.add_input_layer(desc['dimension'])
        elif desc['type'] == 'dense':
            docknet.add_dense_layer(desc['dimension'], desc['activation_function'])
        if 'params' in desc:
            params = {k: np.array(v) for k, v in desc['params'].items()}
            docknet.layers[-1].params = params
    return docknet
