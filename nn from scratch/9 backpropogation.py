
# backprop with relu func for a single neuron

# Forward pass
x = [ 1.0 , - 2.0 , 3.0 ] # input values
w = [ - 3.0 , - 1.0 , 2.0 ] # weights
b = 1.0 # bias

# Multiplying inputs by weights
xw0 = x[ 0 ] * w[ 0 ]
xw1 = x[ 1 ] * w[ 1 ]
xw2 = x[ 2 ] * w[ 2 ]
print (xw0, xw1, xw2, b) # -3.0 2.0 6.0 1.0

# Adding weighted inputs and a bias
z = xw0 + xw1 + xw2 + b
print (z) # 6.0

# ReLU activation function
y = max (z, 0 )
print (y) # 6.0

# -----------

import numpy as np

# Passed in gradient from the next layer
# for the purpose of this example we're going to use
# a vector of 1s
dvalues = np.array([[ 1. , 1. , 1. ]])

# We have 3 sets of weights - one set for each neuron
# we have 4 inputs, thus 4 weights
# recall that we keep weights transposed
weights = np.array([[ 0.2 , 0.8 , - 0.5 , 1 ],
[ 0.5 , - 0.91 , 0.26 , - 0.5 ],
[ - 0.26 , - 0.27 , 0.17 , 0.87 ]]).T

# sum weights of given input
# and multiply by the passed in gradient for this neuron
dx0 = sum (weights[ 0 ]) * dvalues[ 0 ]
dx1 = sum (weights[ 1 ]) * dvalues[ 0 ]
dx2 = sum (weights[ 2 ]) * dvalues[ 0 ]
dx3 = sum (weights[ 3 ]) * dvalues[ 0 ]

dinputs = np.array([dx0, dx1, dx2, dx3])
print (dinputs) # dinputs is a gradient of the neuron function with respect to inputs.
"""
[[ 0.44  0.44  0.44]
 [-0.38 -0.38 -0.38]
 [-0.07 -0.07 -0.07]
 [ 1.37  1.37  1.37]] """


"""
 From NumPy’s perspective, and since both weights and dvalues are NumPy arrays, we
can simplify the dx0 to dx3 calculation. Since the weights array is formatted so that the rows
contain weights related to each input (weights for all neurons for the given input), we can
multiply them by the gradient vector directly: """

# Passed in gradient from the next layer
# for the purpose of this example we're going to use
# a vector of 1s
dvalues = np.array([[ 1. , 1. , 1. ]])
# We have 3 sets of weights - one set for each neuron
# we have 4 inputs, thus 4 weights
# recall that we keep weights transposed
weights = np.array([[ 0.2 , 0.8 , - 0.5 , 1 ],
[ 0.5 , - 0.91 , 0.26 , - 0.5 ],
[ - 0.26 , - 0.27 , 0.17 , 0.87 ]]).T
# sum weights of given input
# and multiply by the passed in gradient for this neuron
dx0 = sum (weights[ 0 ] * dvalues[ 0 ])
dx1 = sum (weights[ 1 ] * dvalues[ 0 ])
dx2 = sum (weights[ 2 ] * dvalues[ 0 ])
dx3 = sum (weights[ 3 ] * dvalues[ 0 ])
dinputs = np.array([dx0, dx1, dx2, dx3])
print (dinputs) # [ 0.44 -0.38 -0.07  1.37]

"""
We can achieve the same result by using the np.dot function. For
this to be possible, we need to match the “inner” shapes and decide the first dimension of the
result, which is the first dimension of the first parameter. """

# Passed in gradient from the next layer
# for the purpose of this example we're going to use
# a vector of 1s
dvalues = np.array([[ 1. , 1. , 1. ]])
# We have 3 sets of weights - one set for each neuron
# we have 4 inputs, thus 4 weights
# recall that we keep weights transposed
weights = np.array([[ 0.2 , 0.8 , - 0.5 , 1 ],
[ 0.5 , - 0.91 , 0.26 , - 0.5 ],
[ - 0.26 , - 0.27 , 0.17 , 0.87 ]]).T
# sum weights of given input
# and multiply by the passed in gradient for this neuron
dinputs = np.dot(dvalues[ 0 ], weights.T)
print (dinputs)
print (dinputs) # [ 0.44 -0.38 -0.07  1.37]

# We have to account for a batch of samples

# Passed in gradient from the next layer
# for the purpose of this example we're going to use
# an array of an incremental gradient values
dvalues = np.array([[ 1. , 1. , 1. ],
[ 2. , 2. , 2. ],
[ 3. , 3. , 3. ]])
# We have 3 sets of weights - one set for each neuron
# we have 4 inputs, thus 4 weights
# recall that we keep weights transposed
weights = np.array([[ 0.2 , 0.8 , - 0.5 , 1 ],
[ 0.5 , - 0.91 , 0.26 , - 0.5 ],
[ - 0.26 , - 0.27 , 0.17 , 0.87 ]]).T

print(weights)
"""
[[ 0.2   0.5  -0.26]
 [ 0.8  -0.91 -0.27]
 [-0.5   0.26  0.17]
 [ 1.   -0.5   0.87]] """

# sum weights of given input
# and multiply by the passed in gradient for this neuron
dinputs = np.dot(dvalues, weights.T)
print (dinputs)
"""
[[ 0.44 -0.38 -0.07  1.37]
 [ 0.88 -0.76 -0.14  2.74]
 [ 1.32 -1.14 -0.21  4.11]] """

""" Let’s combine the forward and backward pass of a single neuron with a full layer and batch-based
partial derivatives. We’ll minimize ReLU’s output, once again, only for this example: """

# Passed in gradient from the next layer
# for the purpose of this example we're going to use
# an array of an incremental gradient values
dvalues = np.array([[ 1. , 1. , 1. ],
[ 2. , 2. , 2. ],
[ 3. , 3. , 3. ]])

# We have 3 sets of inputs - samples
inputs = np.array([[ 1 , 2 , 3 , 2.5 ],
[ 2. , 5. , - 1. , 2 ],
[ - 1.5 , 2.7 , 3.3 , - 0.8 ]])

# We have 3 sets of weights - one set for each neuron
# we have 4 inputs, thus 4 weights
# recall that we keep weights transposed
weights = np.array([[ 0.2 , 0.8 , - 0.5 , 1 ],
[ 0.5 , - 0.91 , 0.26 , - 0.5 ],
[ - 0.26 , - 0.27 , 0.17 , 0.87 ]]).T
"""
array([[ 0.2 ,  0.5 , -0.26],
       [ 0.8 , -0.91, -0.27],
       [-0.5 ,  0.26,  0.17],
       [ 1.  , -0.5 ,  0.87]]) """

# One bias for each neuron
# biases are the row vector with a shape (1, neurons)
biases = np.array([[ 2 , 3 , 0.5 ]])

# Forward pass
layer_outputs = np.dot(inputs, weights) + biases # Dense layer
"""
array([[ 4.8  ,  1.21 ,  2.385],
       [ 8.9  , -1.81 ,  0.2  ],
       [ 1.41 ,  1.051,  0.026]]) """


relu_outputs = np.maximum( 0 , layer_outputs) # ReLU activation
"""
array([[4.8  , 1.21 , 2.385],
       [8.9  , 0.   , 0.2  ],
       [1.41 , 1.051, 0.026]]) """

# Let's optimize and test backpropagation here
# ReLU activation - simulates derivative with respect to input values
# from next layer passed to current layer during backpropagation
drelu = relu_outputs.copy()
"""
array([[4.8  , 1.21 , 2.385],
       [8.9  , 0.   , 0.2  ],
       [1.41 , 1.051, 0.026]]) """

drelu[layer_outputs <= 0 ] = 0

# Dense layer
# dinputs - multiply by weights
dinputs = np.dot(drelu, weights.T)
"""
array([[ 0.9449 ,  2.09495, -1.67995,  6.26995],
       [ 1.728  ,  7.066  , -4.416  ,  9.074  ],
       [ 0.80074,  0.16457, -0.42732,  0.90712]]) """

# dweights - multiply by inputs
dweights = np.dot(inputs.T, drelu)
"""
array([[20.485 , -0.3665,  2.746 ],
       [57.907 ,  5.2577,  5.8402],
       [10.153 ,  7.0983,  7.0408],
       [28.672 ,  2.1842,  6.3417]]) """

# dbiases - sum values, do this over samples (first axis), keepdims
# since this by default will produce a plain list -
# we explained this in the chapter 4
dbiases = np.sum(drelu, axis = 0 , keepdims = True )
# array([[15.11 ,  2.261,  2.611]])

# Update parameters
weights += - 0.001 * dweights
biases += - 0.001 * dbiases

print(weights)
"""
[[ 0.179515   0.5003665 -0.262746 ]
 [ 0.742093  -0.9152577 -0.2758402]
 [-0.510153   0.2529017  0.1629592]
 [ 0.971328  -0.5021842  0.8636583]] """

print(biases) # [[1.98489  2.997739 0.497389]]

# --------- forward pass with single relu neuron and reduce loss ---------

# Forward pass
x = [ 1.0 , - 2.0 , 3.0 ] # input values
w = [ - 3.0 , - 1.0 , 2.0 ] # weights
b = 1.0 # bias

# Multiplying inputs by weights
xw0 = x[ 0 ] * w[ 0 ]
xw1 = x[ 1 ] * w[ 1 ]
xw2 = x[ 2 ] * w[ 2 ]
print (xw0, xw1, xw2, b) # -3.0 2.0 6.0 1.0

# Adding weighted inputs and a bias
z = xw0 + xw1 + xw2 + b
print (z) # 6.0

# ReLU activation function
y = max (z, 0 ) # 6.0
print (y)

# current task is to calculate how much each of the inputs, weights, and a bias impacts the output.

# Backward pass
# The derivative from the next layer
dvalue = 1.0

# Derivative of ReLU and the chain rule
drelu_dz = dvalue * ( 1. if z > 0 else 0. )
print (drelu_dz) # 1.0

# Partial derivatives of the multiplication, the chain rule
dsum_dxw0 = 1
dsum_dxw1 = 1
dsum_dxw2 = 1
dsum_db = 1
drelu_dxw0 = drelu_dz * dsum_dxw0
drelu_dxw1 = drelu_dz * dsum_dxw1
drelu_dxw2 = drelu_dz * dsum_dxw2
drelu_db = drelu_dz * dsum_db
print (drelu_dxw0, drelu_dxw1, drelu_dxw2, drelu_db) # # 1.0 1.0 1.0 1.0

# Partial derivatives of the multiplication, the chain rule
dmul_dx0 = w[ 0 ]
dmul_dx1 = w[ 1 ]
dmul_dx2 = w[ 2 ]
dmul_dw0 = x[ 0 ]
dmul_dw1 = x[ 1 ]
dmul_dw2 = x[ 2 ]
drelu_dx0 = drelu_dxw0 * dmul_dx0
drelu_dw0 = drelu_dxw0 * dmul_dw0
drelu_dx1 = drelu_dxw1 * dmul_dx1
drelu_dw1 = drelu_dxw1 * dmul_dw1
drelu_dx2 = drelu_dxw2 * dmul_dx2
drelu_dw2 = drelu_dxw2 * dmul_dw2
print (drelu_dx0, drelu_dw0, drelu_dx1, drelu_dw1, drelu_dx2, drelu_dw2) # -3.0 1.0 -1.0 -2.0 2.0 3.0

# All together, the partial derivatives above, combined into a vector, make up our gradients. Our
# gradients could be represented as:
dx = [drelu_dx0, drelu_dx1, drelu_dx2] # gradients on inputs
dw = [drelu_dw0, drelu_dw1, drelu_dw2] # gradients on weights
db = drelu_db # gradient on bias...just 1 bias here.

# gradients to the weights to hopefully minimize the output
# applying a negative fraction of the gradient to our weights. We apply a negative fraction to this gradient
# since we want to decrease the final output value, and the gradient shows the direction of the steepest ascent.

print(w,b) # [-3.0, -1.0, 2.0] 1.0

# We can then apply a fraction of the gradients to these values:
w[ 0 ] += - 0.001 * dw[ 0 ]
w[ 1 ] += - 0.001 * dw[ 1 ]
w[ 2 ] += - 0.001 * dw[ 2 ]
b += - 0.001 * db
print (w, b) # [-3.001, -0.998, 1.997] 0.999

# Now, we’ve slightly changed the weights and bias in such a way so as to decrease the output
# somewhat intelligently. We can see the effects of our tweaks on the output by doing another
# forward pass:
# Multiplying inputs by weights
xw0 = x[ 0 ] * w[ 0 ]
xw1 = x[ 1 ] * w[ 1 ]
xw2 = x[ 2 ] * w[ 2 ]

# Adding
z = xw0 + xw1 + xw2 + b

# ReLU activation function
y = max (z, 0 )
print (y) # 5.985

"""
We’ve successfully decreased this neuron’s output from 6.000 to 5.985. Note that it does not
make sense to decrease the neuron’s output in a real neural network; we were doing this purely as
a simpler exercise than the full network. """

import numpy as np
# Passed in gradient from the next layer
# for the purpose of this example we're going to use
# a vector of 1s
dvalues = np.array([[ 1. , 1. , 1. ]])

# We have 3 sets of weights - one set for each neuron
# we have 4 inputs, thus 4 weights
# recall that we keep weights transposed
weights = np.array([[ 0.2 , 0.8 , - 0.5 , 1 ],
[ 0.5 , - 0.91 , 0.26 , - 0.5 ],
[ - 0.26 , - 0.27 , 0.17 , 0.87 ]]).T

# sum weights of given input
# and multiply by the passed in gradient for this neuron
dx0 = sum (weights[ 0 ]) * dvalues[ 0 ]
dx1 = sum (weights[ 1 ]) * dvalues[ 0 ]
dx2 = sum (weights[ 2 ]) * dvalues[ 0 ]
dx3 = sum (weights[ 3 ]) * dvalues[ 0 ]

# dinputs is a gradient of the neuron function with respect to inputs.
dinputs = np.array([dx0, dx1, dx2, dx3])
print (dinputs)
"""
[[ 0.44  0.44  0.44]
 [-0.38 -0.38 -0.38]
 [-0.07 -0.07 -0.07]
 [ 1.37  1.37  1.37]] """

# same ops can be done using np

# sum weights of given input
# and multiply by the passed in gradient for this neuron
dinputs = np.dot(dvalues[ 0 ], weights.T)
print (dinputs)

# handle for batch of samples

# Passed in gradient from the next layer
# for the purpose of this example we're going to use
# an array of an incremental gradient values
dvalues = np.array([[ 1. , 1. , 1. ],
[ 2. , 2. , 2. ],
[ 3. , 3. , 3. ]])

# sum weights of given input
# and multiply by the passed in gradient for this neuron
dinputs = np.dot(dvalues, weights.T)
print (dinputs)
"""
[[ 0.44 -0.38 -0.07  1.37]
 [ 0.88 -0.76 -0.14  2.74]
 [ 1.32 -1.14 -0.21  4.11]] """

"""
 we’re going to
be using gradients to update the weights, so we need to match the shape of weights, not inputs.
Since the derivative with respect to the weights equals inputs, weights are transposed, so we need
to transpose inputs to receive the derivative of the neuron with respect to weights. Then we use
these transposed inputs as the first parameter to the dot product — the dot product is going to
multiply rows by inputs, where each row, as it is transposed, contains data for a given input for all
of the samples, by the columns of dvalues . """

import numpy as np
# Passed in gradient from the next layer
# for the purpose of this example we're going to use
# an array of an incremental gradient values
dvalues = np.array([[ 1. , 1. , 1. ],
[ 2. , 2. , 2. ],
[ 3. , 3. , 3. ]])
# We have 3 sets of inputs - samples
inputs = np.array([[ 1 , 2 , 3 , 2.5 ],
[ 2. , 5. , - 1. , 2 ],
[ - 1.5 , 2.7 , 3.3 , - 0.8 ]])
# sum weights of given input
# and multiply by the passed in gradient for this neuron
dweights = np.dot(inputs.T, dvalues) # dweights is a gradient of the neuron function with respect to the weights.
print (dweights)
"""
[[ 0.5  0.5  0.5]
 [20.1 20.1 20.1]
 [10.9 10.9 10.9]
 [ 4.1  4.1  4.1]] """

"""
  For the biases and derivatives with respect to them, the derivatives come from the sum operation
and always equal 1, multiplied by the incoming gradients to apply the chain rule. Since gradients
are a list of gradients (a vector of gradients for each neuron for all samples), we just have to sum
them with the neurons, column-wise, along axis 0. """

# Passed in gradient from the next layer
# for the purpose of this example we're going to use
# an array of an incremental gradient values
dvalues = np.array([[ 1. , 1. , 1. ],
[ 2. , 2. , 2. ],
[ 3. , 3. , 3. ]])
# One bias for each neuron
# biases are the row vector with a shape (1, neurons)
biases = np.array([[ 2 , 3 , 0.5 ]])
# dbiases - sum values, do this over samples (first axis), keepdims
# since this by default will produce a plain list -
# we explained this in the chapter 4
dbiases = np.sum(dvalues, axis = 0 , keepdims = True )
print (dbiases) # [[6. 6. 6.]]

"""
The last thing to cover here is the derivative of the ReLU function. It equals 1 if the input is
greater than 0 and 0 otherwise. The layer passes its outputs through the ReLU() activation during
the forward pass. For the backward pass, ReLU() receives a gradient of the same shape. The
derivative of the ReLU function will form an array of the same shape, filled with 1 when the
related input is greater than 0, and 0 otherwise. To apply the chain rule, we need to multiply this
array with the gradients of the following function: """

# Example layer output
z = np.array([[ 1 , 2 , - 3 , - 4 ],
[ 2 , - 7 , - 1 , 3 ],
[ - 1 , 2 , 5 , - 1 ]])
dvalues = np.array([[ 1 , 2 , 3 , 4 ],
[ 5 , 6 , 7 , 8 ],
[ 9 , 10 , 11 , 12 ]])
# ReLU activation's derivative
drelu = np.zeros_like(z)
drelu[z > 0 ] = 1
print (drelu)
"""
[[1 1 0 0]
 [1 0 0 1]
 [0 1 1 0]] """
# The chain rule
drelu *= dvalues
print (drelu)
"""
[[ 1  2  0  0]
 [ 5  0  0  8]
 [ 0 10 11  0]] """

"""
Since the ReLU() derivative array is filled with 1s, which do
not change the values multiplied by them, and 0s that zero the multiplying value, this means that
we can take the gradients of the subsequent function and set to 0 all of the values that correspond
to the ReLU() input and are equal to or less than 0: """

# Example layer output
z = np.array([[ 1 , 2 , - 3 , - 4 ],
[ 2 , - 7 , - 1 , 3 ],
[ - 1 , 2 , 5 , - 1 ]])
dvalues = np.array([[ 1 , 2 , 3 , 4 ],
[ 5 , 6 , 7 , 8 ],
[ 9 , 10 , 11 , 12 ]])
# ReLU activation's derivative
# with the chain rule applied
drelu = dvalues.copy()
drelu[z <= 0 ] = 0
print (drelu)
"""
[[ 1  2  0  0]
 [ 5  0  0  8]
 [ 0 10 11  0]] """

""" Let’s combine the forward and backward pass of a single neuron with a full layer and batch-based
partial derivatives. We’ll minimize ReLU’s output, once again, only for this example: """

import numpy as np
# Passed in gradient from the next layer
# for the purpose of this example we're going to use
# an array of an incremental gradient values
dvalues = np.array([[ 1. , 1. , 1. ],
[ 2. , 2. , 2. ],
[ 3. , 3. , 3. ]])
# We have 3 sets of inputs - samples
inputs = np.array([[ 1 , 2 , 3 , 2.5 ],
[ 2. , 5. , - 1. , 2 ],
[ - 1.5 , 2.7 , 3.3 , - 0.8 ]])
# We have 3 sets of weights - one set for each neuron
# we have 4 inputs, thus 4 weights
# recall that we keep weights transposed
weights = np.array([[ 0.2 , 0.8 , - 0.5 , 1 ],
[ 0.5 , - 0.91 , 0.26 , - 0.5 ],
[ - 0.26 , - 0.27 , 0.17 , 0.87 ]]).T
# One bias for each neuron
# biases are the row vector with a shape (1, neurons)
biases = np.array([[ 2 , 3 , 0.5 ]])
# Forward pass
layer_outputs = np.dot(inputs, weights) + biases # Dense layer
relu_outputs = np.maximum( 0 , layer_outputs) # ReLU activation
# Let's optimize and test backpropagation here
# ReLU activation - simulates derivative with respect to input values
# from next layer passed to current layer during backpropagation
drelu = relu_outputs.copy()
drelu[layer_outputs <= 0 ] = 0
# Dense layer
# dinputs - multiply by weights
dinputs = np.dot(drelu, weights.T)
# dweights - multiply by inputs
dweights = np.dot(inputs.T, drelu)
# dbiases - sum values, do this over samples (first axis), keepdims
# since this by default will produce a plain list -
# we explained this in the chapter 4
dbiases = np.sum(drelu, axis = 0 , keepdims = True )
# Update parameters
weights += - 0.001 * dweights
biases += - 0.001 * dbiases
print (weights)
print (biases)
"""
>>> print (weights)
[[ 0.179515   0.5003665 -0.262746 ]
 [ 0.742093  -0.9152577 -0.2758402]
 [-0.510153   0.2529017  0.1629592]
 [ 0.971328  -0.5021842  0.8636583]]
>>> print (biases)
[[1.98489  2.997739 0.497389]] """

# same using OOPS

# Dense layer
class Layer_Dense :
       # Layer initialization
       def __init__ ( self , inputs , neurons ):
              self.weights = 0.01 * np.random.randn(inputs, neurons)
              self.biases = np.zeros(( 1 , neurons))
       # Forward pass
       def forward ( self , inputs ):
              self.output = np.dot(inputs, self.weights) + self.biases

""" # ReLU activation
class Activation_ReLU :
       # Forward pass
       def forward ( self , inputs ):
              self.output = np.maximum( 0 , inputs) """

# Dense layer
class Layer_Dense :
       # Forward pass
       def forward ( self , inputs ):
              self.inputs = inputs

class Layer_Dense :
       # Backward pass
       def backward ( self , dvalues ):
              # Gradients on parameters
              self.dweights = np.dot(self.inputs.T, dvalues)
              self.dbiases = np.sum(dvalues, axis = 0 , keepdims = True )
              # Gradient on values
              self.dinputs = np.dot(dvalues, self.weights.T)

# ReLU activation
class Activation_ReLU :
       # Forward pass
       def forward ( self , inputs ):
              # Remember input values
              self.inputs = inputs
              self.output = np.maximum( 0 , inputs)
       # Backward pass
       def backward ( self , dvalues ):
              # Since we need to modify the original variable,
              # let's make a copy of the values first
              self.dinputs = dvalues.copy()
              # Zero gradient where input values were negative
              self.dinputs[self.inputs <= 0 ] = 0
class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

# Cross-entropy loss
class Loss_CategoricalCrossentropy(Loss):
       # Backward pass
       def backward ( self , dvalues , y_true ):
              # Number of samples
              samples = len (dvalues)
              # Number of labels in every sample
              # We'll use the first sample to count them
              labels = len (dvalues[ 0 ])
              # If labels are sparse, turn them into one-hot vector
              if len (y_true.shape) == 1 :
                     y_true = np.eye(labels)[y_true]
              # Calculate gradient
              self.dinputs = - y_true / dvalues
              # Normalize gradient
              self.dinputs = self.dinputs / samples

# Softmax activation
class Activation_Softmax :
       # Backward pass
       def backward ( self , dvalues ):
              # Create uninitialized array
              self.dinputs = np.empty_like(dvalues)
              # Enumerate outputs and gradients
              for index, (single_output, single_dvalues) in \
                     enumerate ( zip (self.output, dvalues)):
                     # Flatten output array
                     single_output = single_output.reshape( - 1 , 1 )
                     # Calculate Jacobian matrix of the output and
                     jacobian_matrix = np.diagflat(single_output) - \
                     np.dot(single_output, single_output.T)
                     # Calculate sample-wise gradient
                     # and add it to the array of sample gradients
                     self.dinputs[index] = np.dot(jacobian_matrix,
                     single_dvalues)

# Softmax classifier - combined Softmax activation
# and cross-entropy loss for faster backward step
class Activation_Softmax_Loss_CategoricalCrossentropy ():
       # Creates activation and loss function objects
       def __init__ ( self ):
              self.activation = Activation_Softmax()
              self.loss = Loss_CategoricalCrossentropy()
       # Forward pass
       def forward ( self , inputs , y_true ):
              # Output layer's activation function
              self.activation.forward(inputs)
              # Set the output
              self.output = self.activation.output
              # Calculate and return loss value
              return self.loss.calculate(self.output, y_true)
       # Backward pass
       def backward ( self , dvalues , y_true ):
              # Number of samples
              samples = len (dvalues)
              # If labels are one-hot encoded,
              # turn them into discrete values
              if len (y_true.shape) == 2 :
                     y_true = np.argmax(y_true, axis = 1 )
              # Copy so we can safely modify
              self.dinputs = dvalues.copy()
              # Calculate gradient
              self.dinputs[ range (samples), y_true] -= 1
              # Normalize gradient
              self.dinputs = self.dinputs / samples

# test
import numpy as np
import nnfs
nnfs.init()
softmax_outputs = np.array([[ 0.7 , 0.1 , 0.2 ],
[ 0.1 , 0.5 , 0.4 ],
[ 0.02 , 0.9 , 0.08 ]])
class_targets = np.array([ 0 , 1 , 1 ])
softmax_loss = Activation_Softmax_Loss_CategoricalCrossentropy()
softmax_loss.backward(softmax_outputs, class_targets)
dvalues1 = softmax_loss.dinputs
activation = Activation_Softmax()
activation.output = softmax_outputs
loss = Loss_CategoricalCrossentropy()
loss.backward(softmax_outputs, class_targets)
activation.backward(loss.dinputs)
dvalues2 = activation.dinputs
print ( 'Gradients: combined loss and activation:' )
print (dvalues1)
"""
[[-0.1         0.03333333  0.06666667]
 [ 0.03333333 -0.16666667  0.13333333]
 [ 0.00666667 -0.03333333  0.02666667]] """

print ( 'Gradients: separate loss and activation:' )
print (dvalues2)
"""
[[-0.09999999  0.03333334  0.06666667]
 [ 0.03333334 -0.16666667  0.13333334]
 [ 0.00666667 -0.03333333  0.02666667]] """

 # The small difference between values in both arrays results from the precision of floating-point values in raw Python and NumPy

# -------- final code -------------

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
        # Remember input values
        self.inputs = inputs
        # Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases

    # Backward pass
    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)


# ReLU activation
class Activation_ReLU:

    # Forward pass
    def forward(self, inputs):

        # Remember input values
        self.inputs = inputs
        # Calculate output values from inputs
        self.output = np.maximum(0, inputs)

    # Backward pass
    def backward(self, dvalues):
        # Since we need to modify original variable,
        # let's make a copy of values first
        self.dinputs = dvalues.copy()

        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0


# Softmax activation
class Activation_Softmax:

    # Forward pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs

        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1,
                                            keepdims=True))
        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1,
                                            keepdims=True)

        self.output = probabilities

    # Backward pass
    def backward(self, dvalues):

        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)

        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in \
                enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - \
                              np.dot(single_output, single_output.T)

            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix,
                                         single_dvalues)


# Common loss class
class Loss:

    # Calculates the data and regularization losses
    # given model output and ground truth values
    def calculate(self, output, y):

        # Calculate sample losses
        sample_losses = self.forward(output, y)

        # Calculate mean loss
        data_loss = np.mean(sample_losses)

        # Return loss
        return data_loss


# Cross-entropy loss
class Loss_CategoricalCrossentropy(Loss):

    # Forward pass
    def forward(self, y_pred, y_true):

        # Number of samples in a batch
        samples = len(y_pred)

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Probabilities for target values -
        # only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
            ]


        # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1
            )

        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    # Backward pass
    def backward(self, dvalues, y_true):

        # Number of samples
        samples = len(dvalues)
        # Number of labels in every sample
        # We'll use the first sample to count them
        labels = len(dvalues[0])

        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # Calculate gradient
        self.dinputs = -y_true / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples


# Softmax classifier - combined Softmax activation
# and cross-entropy loss for faster backward step
class Activation_Softmax_Loss_CategoricalCrossentropy():

    # Creates activation and loss function objects
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    # Forward pass
    def forward(self, inputs, y_true):
        # Output layer's activation function
        self.activation.forward(inputs)
        # Set the output
        self.output = self.activation.output
        # Calculate and return loss value
        return self.loss.calculate(self.output, y_true)


    # Backward pass
    def backward(self, dvalues, y_true):

        # Number of samples
        samples = len(dvalues)

        # If labels are one-hot encoded,
        # turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # Copy so we can safely modify
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples


# Create dataset
X, y = spiral_data(samples=100, classes=3)

# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)

# Create ReLU activation (to be used with Dense layer):
activation1 = Activation_ReLU()

# Create second Dense layer with 3 input features (as we take output
# of previous layer here) and 3 output values (output values)
dense2 = Layer_Dense(3, 3)

# Create Softmax classifier's combined loss and activation
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

# Perform a forward pass of our training data through this layer
dense1.forward(X)

# Perform a forward pass through activation function
# takes the output of first dense layer here
activation1.forward(dense1.output)

# Perform a forward pass through second Dense layer
# takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output)

# Perform a forward pass through the activation/loss function
# takes the output of second dense layer here and returns loss
loss = loss_activation.forward(dense2.output, y)
# Let's see output of the first few samples:
print(loss_activation.output[:5])

# Print loss value
print('loss:', loss)

# Calculate accuracy from output of activation2 and targets
# calculate values along first axis
predictions = np.argmax(loss_activation.output, axis=1)
if len(y.shape) == 2:
    y = np.argmax(y, axis=1)
accuracy = np.mean(predictions==y)

# Print accuracy
print('acc:', accuracy)

# Backward pass
loss_activation.backward(loss_activation.output, y)
dense2.backward(loss_activation.dinputs)
activation1.backward(dense2.dinputs)
dense1.backward(activation1.dinputs)

# Print gradients
print(dense1.dweights)
print(dense1.dbiases)
print(dense2.dweights)
print(dense2.dbiases)