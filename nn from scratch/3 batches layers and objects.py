import numpy as np
np.random.seed(0)

# BATCHES
#  why batches?
# - perform parallel operations
# - helps in generalization (show multiple samples (8, 16, 32) to model at the time of training)

# ------ 1 layer of 3 neuron ------

# inputs = [1.0, 2.0, 3.0, 2.5] # features of single sample

inputs = [[1.0, 2.0, 3.0, 2.5],
            [2.0, 5.0, -1.0, 2.0],
            [-1.5, 2.7, 3.3, -0.8]] # features in batch of input

# weights and biases are associated to each neuron so we don't want to change weights and biases
# if there is changes in input batches
weights = [[0.2, 0.8, -0.5, 1.0] ,
            [0.5, -0.91, 0.26, -0.5],
            [-0.26, -0.27, 0.17, 0.87]] # weights for 3 neurons
biases = [2, 3, 0.5] # biases for 3 neurons

output = np.dot(weights, inputs) + biases
"""
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<__array_function__ internals>", line 200, in dot
ValueError: shapes (3,4) and (3,4) not aligned: 4 (dim 1) != 3 (dim 0)

error because shape of matrices inputs and weights doesn't match for performing dot product
so lets transpose weights matrix
"""

output = np.dot(inputs, np.array(weights).T ) + biases

print(output)
"""
batches of output

[[ 4.8    1.21   2.385]
 [ 8.9   -1.81   0.2  ]
 [ 1.41   1.051  0.026]] """

# ------ 2 layer of 3 neuron ------

# first layer
inputs = [[1.0, 2.0, 3.0, 2.5],
            [2.0, 5.0, -1.0, 2.0],
            [-1.5, 2.7, 3.3, -0.8]]

weights = [[0.2, 0.8, -0.5, 1.0] ,
            [0.5, -0.91, 0.26, -0.5],
            [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]


# second layer
weights2 = [[0.1, -0.14, 0.5] ,
            [-0.5, 0.12, -0.33],
            [-0.44, 0.73, -0.13]]

biases2 = [-1, 2, -0.5]

layer1_outputs = np.dot(inputs, np.array(weights).T ) + biases

layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T ) + biases2
print(layer2_outputs)
"""
[[ 0.5031  -1.04185 -2.03875]
 [ 0.2434  -2.7332  -5.7633 ]
 [-0.99314  1.41254 -0.35655]] """

 # ------ convert above code to objects -------

X = [[1.0, 2.0, 3.0, 2.5],
            [2.0, 5.0, -1.0, 2.0],
            [-1.5, 2.7, 3.3, -0.8]]

# def two hidden (we're not incharge of how these layers changes because of hyperparameters for the optimizers) layers

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

layer1 = Layer_Dense(4, 5) # 5 neurons
layer2 = Layer_Dense(5, 2) # 2 neurons

layer1.forward(X)
print(layer1.output)
"""
[[ 0.10758131  1.03983522  0.24462411  0.31821498  0.18851053]
 [-0.08349796  0.70846411  0.00293357  0.44701525  0.36360538]
 [-0.50763245  0.55688422  0.07987797 -0.34889573  0.04553042]] """

layer2.forward(layer1.output)
print(layer2.output)
"""
[[ 0.148296   -0.08397602]
 [ 0.14100315 -0.01340469]
 [ 0.20124979 -0.07290616]] """

