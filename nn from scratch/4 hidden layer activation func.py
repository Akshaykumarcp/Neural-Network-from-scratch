import numpy as np
# np.random.seed(0)
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

X = [[1.0, 2.0, 3.0, 2.5],
            [2.0, 5.0, -1.0, 2.0],
            [-1.5, 2.7, 3.3, -0.8]]

# ---- rectified linear func -----
inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]

output = []

# way 1
for i in inputs:
    if i > 0:
        output.append(i)
    elif i <= 0:
        output.append(0)

print(output) # [0, 2, 0, 3.3, 0, 1.1, 2.2, 0]

# way 2
for i in inputs:
    output.append(max(0,i))

print(output) # [0, 2, 0, 3.3, 0, 1.1, 2.2, 0]

# ---- rectified linear in objects -----

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLu:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

layer1 = Layer_Dense(2, 5) # 2 input size (no of feature per sample), 5 neurons in first hidden layer
activation1 = Activation_ReLu()
layer1.forward(X)
print(layer1.output)
"""
[[ 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
   0.00000000e+00]
 [-1.16649511e-03 -9.20157982e-04  6.89033840e-04  1.50167989e-03
  -1.01675784e-03]
 [ 3.07943265e-04  7.25480501e-05  8.33256414e-04  2.82917751e-03
  -3.05135404e-03]
 ...
 [ 1.62512759e-01  1.22753374e-01 -6.35764101e-02 -1.06203661e-01
   3.56387570e-02]
 [ 1.42081151e-01  1.09395809e-01 -6.79499300e-02 -1.32145521e-01
   7.15993290e-02]
 [ 1.76513471e-01  1.31510979e-01 -5.82219888e-02 -8.09362888e-02
   3.28741503e-03]]

lot of negative values
post applying activation func, negative values should be converted to 0"""

activation1.forward(layer1.output) # apply activation
print(activation1.output)
"""
[[0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
  0.00000000e+00]
 [0.00000000e+00 0.00000000e+00 6.89033840e-04 1.50167989e-03
  0.00000000e+00]
 [3.07943265e-04 7.25480501e-05 8.33256414e-04 2.82917751e-03
  0.00000000e+00]
 ...
 [1.62512759e-01 1.22753374e-01 0.00000000e+00 0.00000000e+00
  3.56387570e-02]
 [1.42081151e-01 1.09395809e-01 0.00000000e+00 0.00000000e+00
  7.15993290e-02]
 [1.76513471e-01 1.31510979e-01 0.00000000e+00 0.00000000e+00
  3.28741503e-03]]

  further optimizer will tweak the values """
