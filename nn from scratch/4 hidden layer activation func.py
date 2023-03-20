"""
Activation functions (AF)

The Linear Activation Function
- A linear function is simply the equation of a line.
- It will appear as a straight line when graphed, where y=x and the output value equals the input.
- This activation function is usually applied to the last layer’s output in the case of a regression
    model

Ex:
- First, let’s consider a situation
  where neurons have no activation function, which would be the same as having an activation
  function of y=x . With this linear activation function in a neural network with 2 hidden layers of 8
  neurons each, the result of training this model will look like:
- Diagram: Neural network with linear activation functions in hidden layers attempting to fit
  y=sin(x)
- No matter what we do with this neuron’s weights and biases, the output of this neuron will be
  perfectly linear to y=x of the activation function.
- This linear nature will continue throughout the
  entire network:
- No matter what we do, however many layers we have, this network can only depict linear
  relationships if we use linear activation functions.
- It should be fairly obvious that this will be the
  case as each neuron in each layer acts linearly, so the entire network is a linear function as well.

Non-linear functions
- A nonlinear function cannot be represented well by a
  straight line, such as a sine function (diagram)


why use AF ?
- to add non-linearilty to NN



1. step function
- the step function meant to mimic a neuron in the brain, either “firing”
    or not — like an on-off switch. In programming, an on-off switch as a function would be called a
    step function
- For a step function, if the neuron’s output value, which is calculated by sum(inputs · weights)
    + bias , is greater than 0, the neuron fires (so it would output a 1). Otherwise, it does not fire
    and would pass along a 0.
- The formula for a single neuron might look something like:
    output = sum (inputs * weights) + bias
- apply an activation function to this output, noted by activation()
  output = activation(output)
- Used historically, now a days its rare.
- Problems:
  - not very informative gathered from this function coz either 0 or 1
  - hard to tel how close this function was to activate or deactivate a neuron
  - thus, at optimize w and b, it's easier for optimizer to have activation func that are more granular and informative


2. Sigmoid function
- function returns a value in the range of 0 for negative infinity, through 0.5 for the input of 0,
  and to 1 for positive infinity.
- The output from the Sigmoid function, being in the range of 0 to 1, also works
  better with neural networks — especially compared to the range of the negative to the positive
  infinity — and adds nonlinearity
- The Sigmoid function, historically used in hidden layers, was eventually replaced
  by the Rectified Linear Units activation function (or ReLU)

RELU
- It’s quite literally y=x , clipped
  at 0 from the negative side. If x is less than or equal to 0 , then y is 0 — otherwise, y is equal to x .
- Widely used coz of speed and efficiency
- The ReLU activation function is extremely close to being a linear activation
  function while remaining nonlinear, due to that bend after 0. This simple property is, however,
  very effective.
 """

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

# final code

import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()


# Dense layer
class Layer_Dense:

    # Layer initialization
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    # Forward pass
    def forward(self, inputs):
        # Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases

# ReLU activation
class Activation_ReLU:

    # Forward pass
    def forward(self, inputs):
        # Calculate output values from inputs
        self.output = np.maximum(0, inputs)

# Softmax activation
class Activation_Softmax:

    # Forward pass
    def forward(self, inputs):

        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1,
                                            keepdims=True))
        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1,
                                            keepdims=True)

        self.output = probabilities


# Create dataset
X, y = spiral_data(samples=100, classes=3)

# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)

# Create ReLU activation (to be used with Dense layer):
activation1 = Activation_ReLU()

# Create second Dense layer with 3 input features (as we take output
# of previous layer here) and 3 output values
dense2 = Layer_Dense(3, 3)

# Create Softmax activation (to be used with Dense layer):
activation2 = Activation_Softmax()

# Make a forward pass of our training data through this layer
dense1.forward(X)

# Make a forward pass through activation function
# it takes the output of first dense layer here
activation1.forward(dense1.output)

# Make a forward pass through second Dense layer
# it takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output)

# Make a forward pass through activation function
# it takes the output of second dense layer here
activation2.forward(dense2.output)

# Let's see output of the first few samples:
print(activation2.output[:5])

'''
>>>
[[0.33333334 0.33333334 0.33333334]
 [0.33333316 0.3333332  0.33333364]
 [0.33333287 0.3333329  0.33333418]
 [0.3333326  0.33333263 0.33333477]
 [0.33333233 0.3333324  0.33333528]]
'''