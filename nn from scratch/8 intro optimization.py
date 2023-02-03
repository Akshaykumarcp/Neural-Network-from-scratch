"""
- Previously loss has been found out
- Now have to minimize the loss by fine tuning weights and bias"""

import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import vertical_data
nnfs.init()
import numpy as np

X, y = vertical_data( samples = 100 , classes = 3 )
plt.scatter(X[:, 0 ], X[:, 1 ], c = y, s = 40 , cmap = 'brg' )
plt.show()

# NN Network
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

# Create dataset
X, y = vertical_data( samples = 100 , classes = 3 )

# Create model
dense1 = Layer_Dense( 2 , 3 ) # first dense layer, 2 inputs
activation1 = Activation_ReLu()
dense2 = Layer_Dense( 3 , 3 ) # second dense layer, 3 inputs, 3 outputs
activation2 = Activation_Softmax()
# Create loss function
loss_function = Loss_categorical_cross_entropy()

# ---- let's set large loss and try to reduce it by random values of weights and biases ------

# Helper variables
lowest_loss = 9999999 # some initial value
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()

for iteration in range ( 10000 ):
    # Generate a new set of weights for iteration
    dense1.weights = 0.05 * np.random.randn( 2 , 3 )
    dense1.biases = 0.05 * np.random.randn( 1 , 3 )
    dense2.weights = 0.05 * np.random.randn( 3 , 3 )
    dense2.biases = 0.05 * np.random.randn( 1 , 3 )

    # Perform a forward pass of the training data through this layer
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    # Perform a forward pass through activation function
    # it takes the output of second dense layer here and returns loss
    loss = loss_function.calculate(activation2.output, y)

    # Calculate accuracy from output of activation2 and targets
    # calculate values along first axis
    predictions = np.argmax(activation2.output, axis = 1 )
    accuracy = np.mean(predictions == y)

    # If loss is smaller - print and save weights and biases aside
    if loss < lowest_loss:
        print ( 'New set of weights found, iteration:' , iteration,
                'loss:' , loss, 'acc:' , accuracy)
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
        lowest_loss = loss

"""
New set of weights found, iteration: 0 loss: 1.0987074 acc: 0.3333333333333333
New set of weights found, iteration: 7 loss: 1.0986075 acc: 0.3333333333333333
New set of weights found, iteration: 10 loss: 1.0983617 acc: 0.3333333333333333
New set of weights found, iteration: 11 loss: 1.0978478 acc: 0.3333333333333333
New set of weights found, iteration: 13 loss: 1.0962939 acc: 0.3333333333333333
New set of weights found, iteration: 221 loss: 1.0953202 acc: 0.3333333333333333 """

# loss certainly falls but acc remains the same. this is not a reliable method for minimizing the loss

# ------ another idea: instead of setting parameters randomly each iteration ---------
# let's apply a fraction of these values to parameters
# with this, weights are updated that yields us lowest loss
# if loss increases due to adjustment, then we will revert to previous points

# Create dataset
X, y = vertical_data( samples = 100 , classes = 3 )
# Create model
dense1 = Layer_Dense( 2 , 3 ) # first dense layer, 2 inputs
activation1 = Activation_ReLu()
dense2 = Layer_Dense( 3 , 3 ) # second dense layer, 3 inputs, 3 outputs
activation2 = Activation_Softmax()
# Create loss function
loss_function = Loss_categorical_cross_entropy()
# Helper variables
lowest_loss = 9999999 # some initial value
best_dense1_weights = dense1.weights .copy()
best_dense1_biases = dense1.biases .copy()
best_dense2_weights = dense2.weights .copy()
best_dense2_biases = dense2.biases .copy()
for iteration in range ( 10000 ):
    # Update weights with some small random values
    dense1.weights += 0.05 * np.random.randn( 2 , 3 )
    dense1.biases += 0.05 * np.random.randn( 1 , 3 )
    dense2.weights += 0.05 * np.random.randn( 3 , 3 )
    dense2.biases += 0.05 * np.random.randn( 1 , 3 )
    # Perform a forward pass of our training data through this layer
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    # Perform a forward pass through activation function
    # it takes the output of second dense layer here and returns loss
    loss = loss_function.calculate(activation2.output, y)

    # Calculate accuracy from output of activation2 and targets
    # calculate values along first axis
    predictions = np.argmax(activation2.output, axis = 1 )
    accuracy = np.mean(predictions == y)
    # If loss is smaller - print and save weights and biases aside
    if loss < lowest_loss:
        print ( 'New set of weights found, iteration:' , iteration,
        'loss:' , loss, 'acc:' , accuracy)
        best_dense1_weights = dense1.weights .copy()
        best_dense1_biases = dense1.biases .copy()
        best_dense2_weights = dense2.weights .copy()
        best_dense2_biases = dense2.biases .copy()
        lowest_loss = loss
    # Revert weights and biases
    else :
        dense1.weights = best_dense1_weights .copy()
        dense1.biases = best_dense1_biases .copy()
        dense2.weights = best_dense2_weights .copy()
        dense2.biases = best_dense2_biases .copy()

"""
New set of weights found, iteration: 0 loss: 1.0984875 acc: 0.3333333333333333
New set of weights found, iteration: 1 loss: 1.0947709 acc: 0.3333333333333333
New set of weights found, iteration: 2 loss: 1.0946996 acc: 0.3333333333333333
New set of weights found, iteration: 6 loss: 1.0936711 acc: 0.3333333333333333
New set of weights found, iteration: 8 loss: 1.0934435 acc: 0.3333333333333333
New set of weights found, iteration: 11 loss: 1.0878259 acc: 0.3333333333333333
New set of weights found, iteration: 14 loss: 1.0854708 acc: 0.33666666666666667
New set of weights found, iteration: 15 loss: 1.0848316 acc: 0.3333333333333333
New set of weights found, iteration: 17 loss: 1.0804204 acc: 0.3333333333333333
New set of weights found, iteration: 23 loss: 1.0680876 acc: 0.5066666666666667
New set of weights found, iteration: 24 loss: 1.0628456 acc: 0.62
New set of weights found, iteration: 32 loss: 1.0596823 acc: 0.34
New set of weights found, iteration: 34 loss: 1.0557171 acc: 0.6466666666666666
New set of weights found, iteration: 37 loss: 1.0542742 acc: 0.6366666666666667
New set of weights found, iteration: 43 loss: 1.0457052 acc: 0.4066666666666667
New set of weights found, iteration: 44 loss: 1.041721 acc: 0.37666666666666665
New set of weights found, iteration: 48 loss: 1.0382662 acc: 0.3333333333333333
New set of weights found, iteration: 49 loss: 1.0372553 acc: 0.3333333333333333
New set of weights found, iteration: 51 loss: 1.0225445 acc: 0.3333333333333333
New set of weights found, iteration: 53 loss: 1.0085274 acc: 0.33666666666666667
New set of weights found, iteration: 54 loss: 0.99240905 acc: 0.52
New set of weights found, iteration: 59 loss: 0.9919444 acc: 0.33666666666666667
New set of weights found, iteration: 64 loss: 0.9799587 acc: 0.33666666666666667
New set of weights found, iteration: 65 loss: 0.9766264 acc: 0.39
New set of weights found, iteration: 67 loss: 0.9742292 acc: 0.5
New set of weights found, iteration: 71 loss: 0.9651135 acc: 0.46
New set of weights found, iteration: 72 loss: 0.9615029 acc: 0.45
New set of weights found, iteration: 74 loss: 0.9588685 acc: 0.6466666666666666
New set of weights found, iteration: 76 loss: 0.9451122 acc: 0.49
New set of weights found, iteration: 79 loss: 0.94332457 acc: 0.65
New set of weights found, iteration: 82 loss: 0.9377571 acc: 0.5966666666666667
New set of weights found, iteration: 83 loss: 0.9241103 acc: 0.5533333333333333
New set of weights found, iteration: 84 loss: 0.9148167 acc: 0.4533333333333333
New set of weights found, iteration: 85 loss: 0.9031844 acc: 0.4633333333333333
New set of weights found, iteration: 90 loss: 0.9017159 acc: 0.38666666666666666
New set of weights found, iteration: 91 loss: 0.88953674 acc: 0.5633333333333334
New set of weights found, iteration: 96 loss: 0.8743453 acc: 0.6233333333333333
New set of weights found, iteration: 97 loss: 0.87206805 acc: 0.6133333333333333
New set of weights found, iteration: 98 loss: 0.85777444 acc: 0.63
New set of weights found, iteration: 99 loss: 0.8537147 acc: 0.6
New set of weights found, iteration: 101 loss: 0.85039747 acc: 0.5933333333333334
New set of weights found, iteration: 103 loss: 0.8408952 acc: 0.5433333333333333
New set of weights found, iteration: 104 loss: 0.82623345 acc: 0.49666666666666665
New set of weights found, iteration: 105 loss: 0.8161563 acc: 0.48333333333333334
New set of weights found, iteration: 109 loss: 0.80146503 acc: 0.5433333333333333
New set of weights found, iteration: 110 loss: 0.79921275 acc: 0.5566666666666666
New set of weights found, iteration: 114 loss: 0.7871891 acc: 0.5233333333333333
New set of weights found, iteration: 115 loss: 0.7742186 acc: 0.5633333333333334
New set of weights found, iteration: 117 loss: 0.77323455 acc: 0.6133333333333333
New set of weights found, iteration: 121 loss: 0.7624938 acc: 0.6133333333333333
New set of weights found, iteration: 122 loss: 0.7594491 acc: 0.8833333333333333
New set of weights found, iteration: 124 loss: 0.7553425 acc: 0.7
New set of weights found, iteration: 126 loss: 0.7522245 acc: 0.6966666666666667
New set of weights found, iteration: 129 loss: 0.74410623 acc: 0.49
New set of weights found, iteration: 133 loss: 0.74050874 acc: 0.5033333333333333
New set of weights found, iteration: 135 loss: 0.7329203 acc: 0.6833333333333333
New set of weights found, iteration: 136 loss: 0.71659803 acc: 0.6733333333333333
New set of weights found, iteration: 140 loss: 0.6998589 acc: 0.5533333333333333
New set of weights found, iteration: 141 loss: 0.6978121 acc: 0.8266666666666667
New set of weights found, iteration: 144 loss: 0.6917898 acc: 0.84
New set of weights found, iteration: 146 loss: 0.6779833 acc: 0.8633333333333333
New set of weights found, iteration: 147 loss: 0.671168 acc: 0.8533333333333334
New set of weights found, iteration: 153 loss: 0.6626474 acc: 0.8633333333333333
New set of weights found, iteration: 155 loss: 0.6580105 acc: 0.8566666666666667
New set of weights found, iteration: 157 loss: 0.6545363 acc: 0.89
New set of weights found, iteration: 160 loss: 0.64646363 acc: 0.8866666666666667
New set of weights found, iteration: 173 loss: 0.6443034 acc: 0.8533333333333334
New set of weights found, iteration: 179 loss: 0.63240653 acc: 0.8866666666666667
New set of weights found, iteration: 181 loss: 0.6239661 acc: 0.91
New set of weights found, iteration: 190 loss: 0.61459684 acc: 0.92
New set of weights found, iteration: 191 loss: 0.6143863 acc: 0.9
New set of weights found, iteration: 201 loss: 0.60914457 acc: 0.9166666666666666
New set of weights found, iteration: 203 loss: 0.60130537 acc: 0.9066666666666666
New set of weights found, iteration: 207 loss: 0.5992602 acc: 0.9
New set of weights found, iteration: 213 loss: 0.5948207 acc: 0.8966666666666666
New set of weights found, iteration: 214 loss: 0.5876721 acc: 0.9066666666666666
New set of weights found, iteration: 221 loss: 0.5800158 acc: 0.94
New set of weights found, iteration: 228 loss: 0.5731725 acc: 0.94
New set of weights found, iteration: 229 loss: 0.5642448 acc: 0.9333333333333333
New set of weights found, iteration: 231 loss: 0.55868596 acc: 0.94
New set of weights found, iteration: 237 loss: 0.553303 acc: 0.9266666666666666
New set of weights found, iteration: 238 loss: 0.54731715 acc: 0.9133333333333333
New set of weights found, iteration: 240 loss: 0.5472667 acc: 0.8966666666666666
New set of weights found, iteration: 243 loss: 0.54156727 acc: 0.9033333333333333
New set of weights found, iteration: 245 loss: 0.540352 acc: 0.8966666666666666
New set of weights found, iteration: 246 loss: 0.5366953 acc: 0.9033333333333333
New set of weights found, iteration: 257 loss: 0.5254872 acc: 0.88
New set of weights found, iteration: 262 loss: 0.5241592 acc: 0.9066666666666666
New set of weights found, iteration: 268 loss: 0.5167883 acc: 0.9233333333333333
New set of weights found, iteration: 274 loss: 0.5121611 acc: 0.9133333333333333
New set of weights found, iteration: 281 loss: 0.5085737 acc: 0.9066666666666666
New set of weights found, iteration: 286 loss: 0.50043756 acc: 0.8766666666666667
New set of weights found, iteration: 287 loss: 0.49897054 acc: 0.89
New set of weights found, iteration: 297 loss: 0.49791393 acc: 0.8833333333333333
New set of weights found, iteration: 298 loss: 0.49492615 acc: 0.89
New set of weights found, iteration: 299 loss: 0.48339203 acc: 0.89
New set of weights found, iteration: 302 loss: 0.4759733 acc: 0.9033333333333333
New set of weights found, iteration: 304 loss: 0.47094512 acc: 0.9266666666666666
New set of weights found, iteration: 305 loss: 0.46500778 acc: 0.9366666666666666
New set of weights found, iteration: 310 loss: 0.4577308 acc: 0.9233333333333333
New set of weights found, iteration: 318 loss: 0.454788 acc: 0.9166666666666666
New set of weights found, iteration: 323 loss: 0.4512412 acc: 0.9333333333333333
New set of weights found, iteration: 324 loss: 0.44589365 acc: 0.9166666666666666
New set of weights found, iteration: 330 loss: 0.44283634 acc: 0.9233333333333333
New set of weights found, iteration: 334 loss: 0.4404525 acc: 0.9266666666666666
New set of weights found, iteration: 335 loss: 0.43520635 acc: 0.92
New set of weights found, iteration: 337 loss: 0.43047628 acc: 0.9033333333333333
New set of weights found, iteration: 341 loss: 0.4256771 acc: 0.9266666666666666
New set of weights found, iteration: 342 loss: 0.42330608 acc: 0.9366666666666666
New set of weights found, iteration: 344 loss: 0.41891718 acc: 0.9266666666666666
New set of weights found, iteration: 345 loss: 0.41884944 acc: 0.9233333333333333
New set of weights found, iteration: 347 loss: 0.4173783 acc: 0.9333333333333333
New set of weights found, iteration: 348 loss: 0.41710746 acc: 0.9266666666666666
New set of weights found, iteration: 350 loss: 0.4146216 acc: 0.9266666666666666
New set of weights found, iteration: 353 loss: 0.40956298 acc: 0.93
New set of weights found, iteration: 355 loss: 0.4063392 acc: 0.9133333333333333
New set of weights found, iteration: 359 loss: 0.4022937 acc: 0.9066666666666666
New set of weights found, iteration: 360 loss: 0.40170968 acc: 0.9166666666666666
New set of weights found, iteration: 364 loss: 0.39172816 acc: 0.9266666666666666
New set of weights found, iteration: 366 loss: 0.38865325 acc: 0.9166666666666666
New set of weights found, iteration: 369 loss: 0.37779573 acc: 0.9133333333333333
New set of weights found, iteration: 372 loss: 0.37504587 acc: 0.94
New set of weights found, iteration: 375 loss: 0.370477 acc: 0.9233333333333333
New set of weights found, iteration: 376 loss: 0.3691594 acc: 0.9333333333333333
New set of weights found, iteration: 380 loss: 0.3678167 acc: 0.9233333333333333
New set of weights found, iteration: 384 loss: 0.36266562 acc: 0.94
New set of weights found, iteration: 389 loss: 0.3610994 acc: 0.9433333333333334
New set of weights found, iteration: 390 loss: 0.35996568 acc: 0.94
New set of weights found, iteration: 392 loss: 0.35664627 acc: 0.94
New set of weights found, iteration: 393 loss: 0.35598502 acc: 0.9533333333333334
New set of weights found, iteration: 403 loss: 0.355927 acc: 0.9533333333333334
New set of weights found, iteration: 406 loss: 0.3500056 acc: 0.95
New set of weights found, iteration: 416 loss: 0.34858894 acc: 0.95
New set of weights found, iteration: 419 loss: 0.3474224 acc: 0.95
New set of weights found, iteration: 420 loss: 0.34739456 acc: 0.9466666666666667
New set of weights found, iteration: 430 loss: 0.34497827 acc: 0.95
New set of weights found, iteration: 433 loss: 0.34346908 acc: 0.95
New set of weights found, iteration: 434 loss: 0.3428514 acc: 0.9333333333333333
New set of weights found, iteration: 437 loss: 0.33995706 acc: 0.94
New set of weights found, iteration: 441 loss: 0.33393803 acc: 0.9533333333333334
New set of weights found, iteration: 447 loss: 0.32965618 acc: 0.9433333333333334
New set of weights found, iteration: 452 loss: 0.32867232 acc: 0.9366666666666666
New set of weights found, iteration: 457 loss: 0.32745433 acc: 0.94
New set of weights found, iteration: 458 loss: 0.32656214 acc: 0.9466666666666667
New set of weights found, iteration: 470 loss: 0.32471648 acc: 0.94
New set of weights found, iteration: 473 loss: 0.31998852 acc: 0.9333333333333333
New set of weights found, iteration: 475 loss: 0.31625545 acc: 0.9433333333333334
New set of weights found, iteration: 487 loss: 0.3156864 acc: 0.9466666666666667
New set of weights found, iteration: 488 loss: 0.31501526 acc: 0.94
New set of weights found, iteration: 491 loss: 0.31426266 acc: 0.9433333333333334
New set of weights found, iteration: 493 loss: 0.31413472 acc: 0.9366666666666666
New set of weights found, iteration: 496 loss: 0.31017822 acc: 0.95
New set of weights found, iteration: 498 loss: 0.30378592 acc: 0.9433333333333334
New set of weights found, iteration: 506 loss: 0.3005419 acc: 0.9466666666666667
New set of weights found, iteration: 512 loss: 0.2989446 acc: 0.9466666666666667
New set of weights found, iteration: 515 loss: 0.293427 acc: 0.95
New set of weights found, iteration: 522 loss: 0.2930842 acc: 0.9533333333333334
New set of weights found, iteration: 525 loss: 0.2862963 acc: 0.9566666666666667
New set of weights found, iteration: 526 loss: 0.28059986 acc: 0.9533333333333334
New set of weights found, iteration: 528 loss: 0.27775928 acc: 0.9466666666666667
New set of weights found, iteration: 537 loss: 0.2775494 acc: 0.9566666666666667
New set of weights found, iteration: 539 loss: 0.2718787 acc: 0.9466666666666667
New set of weights found, iteration: 547 loss: 0.27053076 acc: 0.95
New set of weights found, iteration: 554 loss: 0.26756445 acc: 0.94
New set of weights found, iteration: 555 loss: 0.2662266 acc: 0.9433333333333334
New set of weights found, iteration: 556 loss: 0.2615357 acc: 0.9566666666666667
New set of weights found, iteration: 565 loss: 0.26044828 acc: 0.95
New set of weights found, iteration: 566 loss: 0.25842804 acc: 0.9533333333333334
New set of weights found, iteration: 576 loss: 0.255406 acc: 0.95
New set of weights found, iteration: 582 loss: 0.2542282 acc: 0.9466666666666667
New set of weights found, iteration: 584 loss: 0.25041535 acc: 0.9533333333333334
New set of weights found, iteration: 586 loss: 0.24931796 acc: 0.9466666666666667
New set of weights found, iteration: 609 loss: 0.24887629 acc: 0.95
New set of weights found, iteration: 610 loss: 0.24656466 acc: 0.95
New set of weights found, iteration: 614 loss: 0.24427734 acc: 0.9533333333333334
New set of weights found, iteration: 623 loss: 0.24142909 acc: 0.9466666666666667
New set of weights found, iteration: 627 loss: 0.24069193 acc: 0.9466666666666667
New set of weights found, iteration: 632 loss: 0.24060023 acc: 0.9533333333333334
New set of weights found, iteration: 643 loss: 0.23704676 acc: 0.95
New set of weights found, iteration: 650 loss: 0.23666209 acc: 0.9433333333333334
New set of weights found, iteration: 655 loss: 0.23387995 acc: 0.9466666666666667
New set of weights found, iteration: 678 loss: 0.23289882 acc: 0.9466666666666667
New set of weights found, iteration: 689 loss: 0.2293088 acc: 0.9466666666666667
New set of weights found, iteration: 690 loss: 0.22573909 acc: 0.95
New set of weights found, iteration: 693 loss: 0.22465439 acc: 0.95
New set of weights found, iteration: 699 loss: 0.22177124 acc: 0.9466666666666667
New set of weights found, iteration: 713 loss: 0.2205369 acc: 0.9466666666666667
New set of weights found, iteration: 717 loss: 0.21962997 acc: 0.9466666666666667
New set of weights found, iteration: 738 loss: 0.21846616 acc: 0.9466666666666667
New set of weights found, iteration: 744 loss: 0.21809539 acc: 0.9433333333333334
New set of weights found, iteration: 749 loss: 0.21502136 acc: 0.9466666666666667
New set of weights found, iteration: 750 loss: 0.2138932 acc: 0.9466666666666667
New set of weights found, iteration: 756 loss: 0.21125318 acc: 0.9533333333333334
New set of weights found, iteration: 758 loss: 0.20945115 acc: 0.9466666666666667
New set of weights found, iteration: 760 loss: 0.2083191 acc: 0.9533333333333334
New set of weights found, iteration: 778 loss: 0.20572354 acc: 0.9533333333333334
New set of weights found, iteration: 779 loss: 0.20531969 acc: 0.9533333333333334
New set of weights found, iteration: 787 loss: 0.20479542 acc: 0.9533333333333334
New set of weights found, iteration: 795 loss: 0.20433769 acc: 0.95
New set of weights found, iteration: 826 loss: 0.20370159 acc: 0.9466666666666667
New set of weights found, iteration: 850 loss: 0.20260568 acc: 0.9533333333333334
New set of weights found, iteration: 853 loss: 0.20224251 acc: 0.9533333333333334
New set of weights found, iteration: 856 loss: 0.20004989 acc: 0.9466666666666667
New set of weights found, iteration: 859 loss: 0.1965368 acc: 0.9533333333333334
New set of weights found, iteration: 865 loss: 0.19619001 acc: 0.9533333333333334
New set of weights found, iteration: 869 loss: 0.19452196 acc: 0.9533333333333334
New set of weights found, iteration: 871 loss: 0.19371016 acc: 0.9566666666666667
New set of weights found, iteration: 874 loss: 0.18943802 acc: 0.9533333333333334
New set of weights found, iteration: 885 loss: 0.18917651 acc: 0.9533333333333334
New set of weights found, iteration: 887 loss: 0.18754338 acc: 0.9533333333333334
New set of weights found, iteration: 894 loss: 0.18702744 acc: 0.95
New set of weights found, iteration: 896 loss: 0.1864847 acc: 0.95
New set of weights found, iteration: 897 loss: 0.18409947 acc: 0.96
New set of weights found, iteration: 898 loss: 0.18270369 acc: 0.95
New set of weights found, iteration: 903 loss: 0.1814021 acc: 0.9533333333333334
New set of weights found, iteration: 915 loss: 0.17988513 acc: 0.9466666666666667
New set of weights found, iteration: 927 loss: 0.17924024 acc: 0.9633333333333334
New set of weights found, iteration: 929 loss: 0.176509 acc: 0.9466666666666667
New set of weights found, iteration: 933 loss: 0.17502037 acc: 0.95
New set of weights found, iteration: 934 loss: 0.17483254 acc: 0.9433333333333334
New set of weights found, iteration: 944 loss: 0.17276299 acc: 0.9533333333333334
New set of weights found, iteration: 949 loss: 0.1725669 acc: 0.96
New set of weights found, iteration: 974 loss: 0.1715006 acc: 0.9566666666666667
New set of weights found, iteration: 977 loss: 0.1693231 acc: 0.9466666666666667
New set of weights found, iteration: 978 loss: 0.16751765 acc: 0.9466666666666667
New set of weights found, iteration: 979 loss: 0.16676877 acc: 0.95
New set of weights found, iteration: 981 loss: 0.16458519 acc: 0.9566666666666667
New set of weights found, iteration: 985 loss: 0.16319972 acc: 0.95
New set of weights found, iteration: 1010 loss: 0.16276003 acc: 0.9533333333333334
New set of weights found, iteration: 1021 loss: 0.16274287 acc: 0.9566666666666667
New set of weights found, iteration: 1022 loss: 0.16198692 acc: 0.9533333333333334
New set of weights found, iteration: 1036 loss: 0.16114394 acc: 0.9533333333333334
New set of weights found, iteration: 1056 loss: 0.16089249 acc: 0.9566666666666667
New set of weights found, iteration: 1076 loss: 0.16055718 acc: 0.9533333333333334
New set of weights found, iteration: 1077 loss: 0.1589075 acc: 0.9466666666666667
New set of weights found, iteration: 1085 loss: 0.158399 acc: 0.9566666666666667
New set of weights found, iteration: 1093 loss: 0.15805769 acc: 0.9633333333333334
New set of weights found, iteration: 1101 loss: 0.1577295 acc: 0.96
New set of weights found, iteration: 1106 loss: 0.15718092 acc: 0.9633333333333334
New set of weights found, iteration: 1114 loss: 0.15507457 acc: 0.9633333333333334
New set of weights found, iteration: 1115 loss: 0.15403213 acc: 0.95
New set of weights found, iteration: 1136 loss: 0.15354502 acc: 0.96
New set of weights found, iteration: 1160 loss: 0.15296113 acc: 0.96
New set of weights found, iteration: 1163 loss: 0.15143402 acc: 0.9533333333333334
New set of weights found, iteration: 1164 loss: 0.15031217 acc: 0.96
New set of weights found, iteration: 1177 loss: 0.15017991 acc: 0.95
New set of weights found, iteration: 1181 loss: 0.14858207 acc: 0.95
New set of weights found, iteration: 1186 loss: 0.14808612 acc: 0.95
New set of weights found, iteration: 1190 loss: 0.14685278 acc: 0.95
New set of weights found, iteration: 1207 loss: 0.14492567 acc: 0.9566666666666667
New set of weights found, iteration: 1211 loss: 0.14190134 acc: 0.9566666666666667
New set of weights found, iteration: 1221 loss: 0.14132632 acc: 0.9533333333333334
New set of weights found, iteration: 1234 loss: 0.14101921 acc: 0.9566666666666667
New set of weights found, iteration: 1238 loss: 0.13976549 acc: 0.96
New set of weights found, iteration: 1241 loss: 0.13941061 acc: 0.9566666666666667
New set of weights found, iteration: 1250 loss: 0.13848363 acc: 0.9533333333333334
New set of weights found, iteration: 1257 loss: 0.13840272 acc: 0.95
New set of weights found, iteration: 1258 loss: 0.13705422 acc: 0.96
New set of weights found, iteration: 1259 loss: 0.13564675 acc: 0.9566666666666667
New set of weights found, iteration: 1264 loss: 0.13531914 acc: 0.96
New set of weights found, iteration: 1268 loss: 0.13478424 acc: 0.9533333333333334
New set of weights found, iteration: 1271 loss: 0.13286543 acc: 0.96
New set of weights found, iteration: 1284 loss: 0.13194227 acc: 0.96
New set of weights found, iteration: 1301 loss: 0.13167211 acc: 0.9533333333333334
New set of weights found, iteration: 1318 loss: 0.13067496 acc: 0.9533333333333334
New set of weights found, iteration: 1328 loss: 0.13014114 acc: 0.9533333333333334
New set of weights found, iteration: 1341 loss: 0.13009298 acc: 0.96
New set of weights found, iteration: 1342 loss: 0.12971078 acc: 0.96
New set of weights found, iteration: 1353 loss: 0.1292222 acc: 0.96
New set of weights found, iteration: 1377 loss: 0.1289623 acc: 0.9533333333333334
New set of weights found, iteration: 1379 loss: 0.1284198 acc: 0.9533333333333334
New set of weights found, iteration: 1380 loss: 0.12826477 acc: 0.96
New set of weights found, iteration: 1398 loss: 0.12822092 acc: 0.9533333333333334
New set of weights found, iteration: 1424 loss: 0.12801841 acc: 0.9533333333333334
New set of weights found, iteration: 1428 loss: 0.1273615 acc: 0.96
New set of weights found, iteration: 1437 loss: 0.12710948 acc: 0.9533333333333334
New set of weights found, iteration: 1462 loss: 0.1270865 acc: 0.9533333333333334
New set of weights found, iteration: 1463 loss: 0.12689057 acc: 0.96
New set of weights found, iteration: 1466 loss: 0.12684974 acc: 0.96
New set of weights found, iteration: 1472 loss: 0.12583242 acc: 0.9566666666666667
New set of weights found, iteration: 1489 loss: 0.12569143 acc: 0.9566666666666667
New set of weights found, iteration: 1519 loss: 0.1251728 acc: 0.96
New set of weights found, iteration: 1531 loss: 0.123804115 acc: 0.9533333333333334
New set of weights found, iteration: 1541 loss: 0.123375155 acc: 0.9566666666666667
New set of weights found, iteration: 1574 loss: 0.12318914 acc: 0.96
New set of weights found, iteration: 1589 loss: 0.12247859 acc: 0.9566666666666667
New set of weights found, iteration: 1592 loss: 0.12174063 acc: 0.9633333333333334
New set of weights found, iteration: 1605 loss: 0.12160721 acc: 0.9633333333333334
New set of weights found, iteration: 1606 loss: 0.12114216 acc: 0.96
New set of weights found, iteration: 1626 loss: 0.120632224 acc: 0.9533333333333334
New set of weights found, iteration: 1650 loss: 0.119857 acc: 0.9566666666666667
New set of weights found, iteration: 1664 loss: 0.11958987 acc: 0.9533333333333334
New set of weights found, iteration: 1668 loss: 0.11883352 acc: 0.96
New set of weights found, iteration: 1676 loss: 0.118607484 acc: 0.9633333333333334
New set of weights found, iteration: 1681 loss: 0.118410975 acc: 0.95
New set of weights found, iteration: 1693 loss: 0.11814265 acc: 0.9533333333333334
New set of weights found, iteration: 1696 loss: 0.117764406 acc: 0.96
New set of weights found, iteration: 1831 loss: 0.11759735 acc: 0.9533333333333334
New set of weights found, iteration: 1845 loss: 0.11692342 acc: 0.96
New set of weights found, iteration: 1908 loss: 0.11689401 acc: 0.9666666666666667
New set of weights found, iteration: 1917 loss: 0.11656797 acc: 0.9566666666666667
New set of weights found, iteration: 1926 loss: 0.116481856 acc: 0.96
New set of weights found, iteration: 1938 loss: 0.11599514 acc: 0.9533333333333334
New set of weights found, iteration: 1952 loss: 0.11564748 acc: 0.96
New set of weights found, iteration: 1962 loss: 0.11501895 acc: 0.9633333333333334
New set of weights found, iteration: 1989 loss: 0.11490126 acc: 0.9666666666666667
New set of weights found, iteration: 1992 loss: 0.11391325 acc: 0.9533333333333334
New set of weights found, iteration: 1996 loss: 0.11359161 acc: 0.96
New set of weights found, iteration: 2029 loss: 0.11337417 acc: 0.96
New set of weights found, iteration: 2030 loss: 0.11273825 acc: 0.9533333333333334
New set of weights found, iteration: 2038 loss: 0.1123399 acc: 0.9566666666666667
New set of weights found, iteration: 2079 loss: 0.11221922 acc: 0.9566666666666667
New set of weights found, iteration: 2080 loss: 0.11217795 acc: 0.9533333333333334
New set of weights found, iteration: 2153 loss: 0.11214544 acc: 0.9533333333333334
New set of weights found, iteration: 2173 loss: 0.11164228 acc: 0.96
New set of weights found, iteration: 2207 loss: 0.11161997 acc: 0.9566666666666667
New set of weights found, iteration: 2212 loss: 0.11161356 acc: 0.9533333333333334
New set of weights found, iteration: 2220 loss: 0.11145248 acc: 0.9566666666666667
New set of weights found, iteration: 2225 loss: 0.11140226 acc: 0.9566666666666667
New set of weights found, iteration: 2248 loss: 0.11132632 acc: 0.96
New set of weights found, iteration: 2297 loss: 0.11123787 acc: 0.96
New set of weights found, iteration: 2359 loss: 0.11118475 acc: 0.9566666666666667
New set of weights found, iteration: 2523 loss: 0.11089203 acc: 0.96
New set of weights found, iteration: 2542 loss: 0.1106587 acc: 0.96
New set of weights found, iteration: 2555 loss: 0.11062879 acc: 0.96
New set of weights found, iteration: 2571 loss: 0.11051885 acc: 0.96
New set of weights found, iteration: 2596 loss: 0.11049139 acc: 0.96
New set of weights found, iteration: 2597 loss: 0.11032727 acc: 0.96
New set of weights found, iteration: 2607 loss: 0.11016383 acc: 0.96
New set of weights found, iteration: 2614 loss: 0.11015165 acc: 0.9566666666666667
New set of weights found, iteration: 2617 loss: 0.11013978 acc: 0.96
New set of weights found, iteration: 2672 loss: 0.10986038 acc: 0.9533333333333334
New set of weights found, iteration: 2691 loss: 0.10984133 acc: 0.96
New set of weights found, iteration: 2722 loss: 0.109580666 acc: 0.96
New set of weights found, iteration: 2727 loss: 0.10943038 acc: 0.96
New set of weights found, iteration: 2730 loss: 0.10916651 acc: 0.9566666666666667
New set of weights found, iteration: 2758 loss: 0.109164655 acc: 0.96
New set of weights found, iteration: 2777 loss: 0.10914691 acc: 0.96
New set of weights found, iteration: 2809 loss: 0.108765386 acc: 0.96
New set of weights found, iteration: 2834 loss: 0.108760096 acc: 0.96
New set of weights found, iteration: 2883 loss: 0.108626045 acc: 0.96
New set of weights found, iteration: 2978 loss: 0.10858613 acc: 0.96
New set of weights found, iteration: 2982 loss: 0.10843824 acc: 0.96
New set of weights found, iteration: 2990 loss: 0.10836234 acc: 0.9633333333333334
New set of weights found, iteration: 3013 loss: 0.108347334 acc: 0.9566666666666667
New set of weights found, iteration: 3058 loss: 0.108091086 acc: 0.96
New set of weights found, iteration: 3092 loss: 0.10795685 acc: 0.96
New set of weights found, iteration: 3223 loss: 0.10789524 acc: 0.9633333333333334
New set of weights found, iteration: 3295 loss: 0.10787608 acc: 0.96
New set of weights found, iteration: 3455 loss: 0.10787282 acc: 0.9566666666666667
New set of weights found, iteration: 3623 loss: 0.107840426 acc: 0.96
New set of weights found, iteration: 3628 loss: 0.1076563 acc: 0.96
New set of weights found, iteration: 3689 loss: 0.107616045 acc: 0.96
New set of weights found, iteration: 3695 loss: 0.10758798 acc: 0.9566666666666667
New set of weights found, iteration: 3704 loss: 0.10751699 acc: 0.96
New set of weights found, iteration: 3714 loss: 0.10750296 acc: 0.9633333333333334
New set of weights found, iteration: 3716 loss: 0.107348725 acc: 0.96
New set of weights found, iteration: 3935 loss: 0.10733955 acc: 0.96
New set of weights found, iteration: 3953 loss: 0.107266285 acc: 0.96
New set of weights found, iteration: 4039 loss: 0.10723431 acc: 0.96
New set of weights found, iteration: 4148 loss: 0.10720879 acc: 0.96
New set of weights found, iteration: 5160 loss: 0.10716492 acc: 0.96
New set of weights found, iteration: 5583 loss: 0.107152954 acc: 0.9633333333333334
New set of weights found, iteration: 5601 loss: 0.10710914 acc: 0.96
New set of weights found, iteration: 5731 loss: 0.107095 acc: 0.96
New set of weights found, iteration: 5753 loss: 0.10709324 acc: 0.96
New set of weights found, iteration: 6113 loss: 0.10709178 acc: 0.9633333333333334
New set of weights found, iteration: 6157 loss: 0.10708496 acc: 0.96
New set of weights found, iteration: 6263 loss: 0.10707771 acc: 0.96
New set of weights found, iteration: 6334 loss: 0.10704773 acc: 0.96
New set of weights found, iteration: 6610 loss: 0.10700399 acc: 0.96
New set of weights found, iteration: 6753 loss: 0.10696361 acc: 0.96
New set of weights found, iteration: 7025 loss: 0.106910326 acc: 0.96
New set of weights found, iteration: 7371 loss: 0.106904425 acc: 0.96
New set of weights found, iteration: 7553 loss: 0.10685264 acc: 0.9633333333333334
New set of weights found, iteration: 7728 loss: 0.10681323 acc: 0.9633333333333334
New set of weights found, iteration: 7742 loss: 0.10678566 acc: 0.96
New set of weights found, iteration: 8978 loss: 0.106721625 acc: 0.96
New set of weights found, iteration: 8986 loss: 0.106717475 acc: 0.9633333333333334
New set of weights found, iteration: 9276 loss: 0.10668301 acc: 0.96
New set of weights found, iteration: 9308 loss: 0.1066614 acc: 0.96

loss descended decent amount this time and acc raised significant
almost call a solution
when tried with 100000 iterations, not progress further

not optimal way to find weights and biases because:
- no of possible combinations of weights and biases is infinte
- need smarter way
- each weight and bias may have different degress of influence (depends on parameters themselves and current samples) on loss


"""