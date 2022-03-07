###############################################################################################################
# Here we define a dummy DockNet for testing purposes only.                                                   #
# We illustrate what is the expected behavior of a neural network with:                                       #
# - input vectors (also called examples) of size 2                                                            #
# - an input batch of 2 vectors (a matrix with an input example per column)                                   #
# - a single hidden dense layer with 3 neurons (1 neuron per row) and ReLU as activation function             #
# - an output layer with a single neuron and sigmoid as activation function (for binary classification)       #
# Note we define all input values an parameters as small integers to ease the manual check of each            #
# calculation. Each input, output and operation at each step in this file. The diagram below summarizes the   #
# network structure and notation:                                                                             #
#                                                                                                             #
#                                                                                                             #
#         Input layer L0           Hidden layer L1                Output layer L2                             #
#                                                                                                             #
#      example 0  example 1   linear part | activation        linear part | activation                        #
#                                                                                                             #
#         --          --                                                                                      #
#         | a000  a001 |  /-[Z10=X*W10+b10|A10=relu(Z10)]-\                                                   #
#         |            | /                                 \                                  --       --     #
# X = A0 =|            | ---[Z11=X*W11+b11|A11=relu(Z11)]---[Z20=A1*W20+b20|A1=sigmoid(Z20)]--| a20 a21 | = Ŷ #
#         |            | \                                 /                                 --       --      #
#         | a010  a011 |  \-[Z12=X*W12+b12|A12=relu(Z12)]-/                                                   #
#         --          --                                                                                      #
#                                                                                                             #
###############################################################################################################

import numpy as np

from docknet.function.activation_function import (relu,
                                                  relu_prime,
                                                  sigmoid,
                                                  sigmoid_prime)

###############################################################################################################
#                                              Forward propagation                                            #
#                                                                                                             #
# Given a batch of input vectors, feed them to the network to compute the corresponding outputs               #
###############################################################################################################

# Dummy input values alij, where l is the layer index (the input layer is
# layer 0), i is the index of the scalar within the input vector, and j is the
# input example index
from docknet.function.cost_function import cross_entropy, dcross_entropy_dYcirc

a000 = 1
a001 = 2
a010 = 3
a011 = 4

# Dummy layer 0: all input vectors in a single matrix. Each column contains one
# input vector.
X = A0 = np.array([
    [a000, a001],
    [a010, a011]
])

# m is the amount of input vectors
m = X.shape[1]

# Dummy dataset labels
Y = np.array([[0, 1]])

# Dummy parameter values wlij for layer 1, 3 neurons X 2 input values = 6
# parameters; l is the layer index, i is the neuron index, and j is the
# parameter index within neuron i (one parameter per input number received).
w100 = 0.01
w101 = 0.02
w110 = 0.03
w111 = 0.04
w120 = 0.05
w121 = 0.06

# All layer 1 w parameters in a single matrix. Each row contains all the
# parameters of a single neuron, one parameter per layer input (since the
# input vectors contain 2 scalars, layer 1 has 2 inputs).
W1 = np.array([
    [w100, w101],
    [w110, w111],
    [w120, w121]
])

# Dummy bias values bli for layer 1; each neuron has a bias constant that is
# added, so 3 bias values for 3 neurons; l is the layer index and i the neuron
# index.
b10 = 0.01
b11 = 0.02
b12 = 0.03

# All layer 1 biases in one single matrix, one per neuron.
b1 = np.array([
    [b10],
    [b11],
    [b12]
])

# Linear computation of layer 1: dot product of W1 and A0 plus bias b1; each
# column corresponds to the linear computation of the entire layer for one
# single input example:
Z1 = np.array([
    [w100 * a000 + w101 * a010 + b10, w100 * a001 + w101 * a011 + b10],
    [w110 * a000 + w111 * a010 + b11, w110 * a001 + w111 * a011 + b11],
    [w120 * a000 + w121 * a010 + b12, w120 * a001 + w121 * a011 + b12]
])

# Activation of layer 1: application of the activation function to each
# element in Z1; each column corresponds to the output of the entire layer for
# one input example:
A1 = relu(Z1)
a100, a101 = A1[0][0], A1[0][1]
a110, a111 = A1[1][0], A1[1][1]
a120, a121 = A1[2][0], A1[2][1]

# Dummy parameter values for layer 2 with a single neuron for binary: 1 neuron
# x 3 input values = 3 parameters:
w200 = 0.01
w201 = 0.02
w202 = 0.03

# All layer 2 w parameters in a single matrix. Each row contains all the
# parameters of a single neuron, one parameter per layer input (since layer 1
# has 3 neurons, this layer has 3 inputs).
W2 = np.array([
    [w200, w201, w202]
])

# Dummy bias value for layer 2
b20 = 0.01

# All layer 2 biases in one single matrix, one per neuron.
b2 = np.array([
    [b20]
])

# Linear computation of layer 2: dot product of W2 and A1 plus bias b2; each
# column corresponds to the linear computation of the entire layer for one
# single input example:
Z2 = np.array([
    [w200 * a100 + w201 * a110 + w202 * a120 + b20, w200 * a101 + w201 * a111
     + w202 * a121 + b20]
])

# Activation of layer 2 and final network output Ŷ: application of the
# activation function to each element in Z1:
Y_circ = A2 = sigmoid(Z2)

# The network output is a horizontal vector with as many scalars as input
# examples in the input batch:
y_circ0, y_circ1 = Y_circ[0][0], Y_circ[0][1]

###############################################################################################################
#                                              Backward propagation                                           #
#                                                                                                             #
# After a forward propagation, compute the gradients of the cost function wrt each parameter in the network.  #
# These gradients are the direction in which the parameters are to be subtracted some amount so that in the   #
# next iteration the cost is reduced. Forward and backward propagation steps are repeated until finding a set #
# of parameter values close enough to a local minimum                                                         #
# Backward propagation is heavily based on the chain rule:                                                    #
#                                                                                                             #
#   (f(g(x))' = f'(g(x)) * g'(x)                                                                              #
#                                                                                                             #
# We start computing the derivative of the cost function J wrt the network output Ŷ, then keep computing the  #
# derivatives of the cost function wrt each activation, linear part and layer parameters from the previously  #
# computed derivatives                                                                                        #
###############################################################################################################

# The cost function:
J = cross_entropy

# Gradient of J wrt the network output:
dJdY_circ = dcross_entropy_dYcirc(Y_circ, Y)

# The network output Ŷ is A2, the activation of the second and last layer
dJdA2 = dJdY_circ
# Note we compute different derivatives for each input example
# (1 neuron X 2 inputs = 2 derivatives)
dJda200, dJda201 = dJdA2[0][0], dJdA2[0][1]

# Backward activation of layer 2 computes dJdZ2 from dJdA2 and Z2
dJdZ2 = dJdA2 * sigmoid_prime(Z2)
# Derivatives of J wrt the linear part of the neuron;
# again 1 neuron X 2 inputs = 2 derivatives
dJdz200, dzda201 = dJdZ2[0][0], dJdZ2[0][1]

# Average derivatives of J wrt the parameters dJdW2 and dJdb2 inside the linear
# part of the neuron. Note we compute the averages for all input examples in
# order to minimize the average cost for all input examples
dJdW2 = np.dot(dJdZ2, A1.T) / m

# Same amount of derivatives as parameters W in the second layer, that is,
# 1 neuron X 3 inputs = 3 parameters
dJdw200, dJdw201, dJd202 = dJdW2[0][0], dJdW2[0][1], dJdW2[0][2]

# Same amount of derivatives as parameters b in the second layer, that is,
# 1 neuron X 3 inputs = 3 parameters
dJdb2 = np.sum(dJdZ2, axis=1, keepdims=True) / m

# Derivative of the cost function
dJdA1 = np.dot(W2.T, dJdZ2)

# Backward activation of layer 1 computes dJdZ1 from dJdA1 and Z1
dJdZ1 = dJdA1 * relu_prime(Z1)

# Backward linear computes dJdW1, dJdb1 and dJdA0 from dJdZ1 and A0
dJdW1 = np.dot(dJdZ1, A0.T) / m
dJdb1 = np.sum(dJdZ1, axis=1, keepdims=True) / m
dJdA0 = np.dot(W1.T, dJdZ1)

###############################################################################################################
#                                            Parameter optimization                                           #
#                                                                                                             #
# Gradient descent decrements each parameter in the direction of the gradient of the cost function wrt the    #
# parameter. The magnitude of the decrement is proportional to hyperparameter α, the learning rate            #
###############################################################################################################

learning_rate = 0.01

optimized_W2 = W2 - learning_rate * dJdW2
optimized_b2 = b2 - learning_rate * dJdb2
optimized_W1 = W1 - learning_rate * dJdW1
optimized_b1 = b1 - learning_rate * dJdb1
