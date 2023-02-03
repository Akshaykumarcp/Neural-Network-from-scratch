# simplify previous code
# modelling 3 neurons with 4 inputs and 3 outputs
inputs = [1.0, 2.0, 3.0, 2.5]
weights = [[0.2, 0.8, -0.5, 1.0] ,[0.5, -0.91, 0.26, -0.5],[-0.26, -0.27, 0.17, 0.87]] # weights for 3 neurons
biases = [2, 3, 0.5] # biases for 3 neurons


layer_outputs = [] # output of current layer
for neuron_weights, neuron_bias in zip(weights, biases):
    neuron_output = 0 # output of given neuron
    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += n_input * weight
    neuron_output += neuron_bias
    layer_outputs.append(neuron_output)

print(layer_outputs) # [4.8, 1.21, 2.385]

# dot product on single neuron

import numpy as np

inputs = [1.0, 2.0, 3.0, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2

output = np.dot(inputs, weights) + bias
print(output) # 4.8

# dot product on layer of neuron

inputs = [1.0, 2.0, 3.0, 2.5]
weights = [[0.2, 0.8, -0.5, 1.0] ,[0.5, -0.91, 0.26, -0.5],[-0.26, -0.27, 0.17, 0.87]] # weights for 3 neurons
biases = [2, 3, 0.5] # biases for 3 neurons

# based on first element (weights) that we pass, return will be indexed (we want weights to be indexed)
# so passing weights as first input to dot()
output = np.dot(weights, inputs) + biases
print(output) # [4.8   1.21  2.385]

output = np.dot(inputs, weights) + biases
"""
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<__array_function__ internals>", line 200, in dot
ValueError: shapes (4,) and (3,4) not aligned: 4 (dim 0) != 3 (dim 0) """
