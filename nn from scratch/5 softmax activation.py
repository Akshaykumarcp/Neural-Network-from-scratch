""" Why Softmax ?
- In relu, cliping negative values to 0. Post that if we do probablity distribution, values will be zero. learning will be difficult
- In linear activation func or squaring negative values or obsolute value for handling negative values, later need to do back prop and
    optimizer var then there'll be big difference in mearning (-9 to 9).
- softmax for:
    - softmax uses exponentional function for input values (solves negative value issues) and
    - for probable prediction """

# exponential func code
import math

layer_outputs = [4.8, 1.21, 2.385]

E = math.e

exp_values = []

# exponentiate values because to get rid of negative values and retain -ve value meaning
for output in layer_outputs:
    exp_values.append(E**output)

# exponential values
print(exp_values)
# [121.51041751873483, 3.353484652549023, 10.859062664920513]

# next step: normalize values (NORMALIZATION)

norm_base = sum(exp_values)

norm_values = []

for value in exp_values:
    norm_values.append(value/norm_base)

# normalized exponential values, provides in probablistic values
print(norm_values) # [0.8952826639572619, 0.024708306782099374, 0.0800090292606387]

# all values add upto 1
print(sum(norm_values)) # 0.9999999999999999

# let's convert above code to numpy

import numpy as np

exp_values = np.exp(layer_outputs)

print(exp_values) # [121.51041752   3.35348465  10.85906266]

norm_values = exp_values / np.sum(exp_values)
print(norm_values) # [0.89528266 0.02470831 0.08000903]
print(sum(norm_values)) # 0.9999999999999999

# combination of exponential and normalization makes up SOFTMAX

# let's apply for batch of output layer

layer_outputs = [[4.8, 1.21, 2.385],
                    [8.9, -1.81, 0.2],
                    [1.41, 1.051, 0.020]]

exp_values = np.exp(layer_outputs)

print(exp_values)
"""
[[1.21510418e+02 3.35348465e+00 1.08590627e+01]
 [7.33197354e+03 1.63654137e-01 1.22140276e+00]
 [4.09595540e+00 2.86051020e+00 1.02020134e+00]] """

norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)

print(norm_values)
"""
[[8.95282664e-01 2.47083068e-02 8.00090293e-02]
 [9.99811129e-01 2.23163963e-05 1.66554348e-04]
 [5.13492093e-01 3.58609707e-01 1.27898200e-01]] """\

# exponential number grows massively
# overflow issue
# fix: subtract with max value with all the values. doing this, largest value will become zero and rest of values becomes less than zero
# range of possibilities becomes 0 to 1. No more worry about overflow
# how does it reflect on output ?
# output identitically same, just protected from overflow

# let's implement in objects

import nnfs
from nnfs.datasets import spiral_data


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLu:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

X, y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2,3)

activation1 = Activation_ReLu()

dense2 = Layer_Dense(3, 3)

activation2 = Activation_Softmax()

dense1.forward(X)

activation1.forward(dense1.output)

dense2.forward(activation1.output)

activation2.forward(dense2.output)

print(activation2.output[:5])
"""
[[0.33333333 0.33333333 0.33333333]
 [0.33332279 0.33335661 0.3333206 ]
 [0.33331688 0.33336964 0.33331347]
 [0.33333774 0.33334007 0.33332219]
 [0.3332896  0.33342985 0.33328055]] """