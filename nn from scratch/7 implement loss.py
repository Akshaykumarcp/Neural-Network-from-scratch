import numpy as np

softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])

class_targets = [0, 1, 1]

# interested in values
print(softmax_outputs[[0, 1, 2], class_targets]) # [0.7 0.5 0.9]

# calc loss
print(-np.log(softmax_outputs[[0, 1, 2], class_targets])) # [0.35667494 0.69314718 0.10536052]

softmax_outputs = np.array([[0.0, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])

# calc loss
print(-np.log(softmax_outputs[[0, 1, 2], class_targets]))

"""
<stdin>:1: RuntimeWarning: divide by zero encountered in log
[       inf 0.69314718 0.10536052]

we get infinite for 0 value
"""

# calc average loss
print(np.mean(-np.log(softmax_outputs[[0, 1, 2], class_targets]))) # inf
# got inf

# 1 option is to clip 1e-7

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_categorical_cross_entropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7) # 1e-7: close to zero, but not zero

        # people pass scalar 1D data or one hot encoding data, so dynamically let's handle both

        # if target is in scalar form is passed. ex: [0, 1, 1]
        if len(y_true.shape) == 1:
            # for selecting the required values (max probability value) from softmax output
            correc_confidences = y_pred_clipped[range(samples), y_true]
            # if one hot encoded values are passed. ex: [[1,0,0], [0,1,0],[0,1,0]]
        elif len(y_true.shape) == 2:
            # for selecting the required values (max probability value) from softmax output
            correc_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihoods = -np.log(correc_confidences)

        return negative_log_likelihoods





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

loss_function = Loss_categorical_cross_entropy()

loss = loss_function.calculate(activation2.output,y)

print(loss)
# 1.099040630434811

# goal is to decrease the loss using optimization of weights and biases

# Accuracy Calculation

import numpy as np
# Probabilities of 3 samples
softmax_outputs = np.array([[ 0.7 , 0.2 , 0.1 ],
                            [ 0.5 , 0.1 , 0.4 ],
                            [ 0.02 , 0.9 , 0.08 ]])

# Target (ground-truth) labels for 3 samples
class_targets = np.array([ 0 , 1 , 1 ])

# Calculate values along second axis (axis of index 1)
predictions = np.argmax(softmax_outputs, axis = 1 )

# If targets are one-hot encoded - convert them
if len (class_targets.shape) == 2 :
    class_targets = np.argmax(class_targets, axis = 1 )

# True evaluates to 1; False to 0
accuracy = np.mean(predictions == class_targets)
print ( 'acc:' , accuracy) # acc: 0.6666666666666666