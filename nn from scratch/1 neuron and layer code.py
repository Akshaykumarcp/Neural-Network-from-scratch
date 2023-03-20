"""
Trainable parameters:
- Weights
- Biases
    - The purpose of the bias is to offset the output positively or negatively, which can further
        help us map more real-world types of dynamic data.

why not just have biases or just weights?
- Biases and weights are both tunable parameters, and both will impact the neurons’ outputs, but
    they do so in different ways.
WEIGHTS
- Since weights are multiplied, they will only change the magnitude or
    even completely flip the sign from positive to negative, or vice versa.
- If weights and biases are thought as a straight line equation.
    - Output = weight·input+bias is not unlike the equation for a line y = mx+b
    - Adjusting the weight will impact the slope of the function
    - As we increase the value of the weight, the slope will get steeper. If we decrease the weight, the
        slope will decrease. If we negate the weight, the slope turns to a negative
BIAS
- The bias offsets the overall function.
- As we increase the bias, the function output overall shifts upward. If we decrease the bias, then
    the overall function output will move downward.

- weights and biases help to impact the outputs of neurons, but they do so in
    slightly different ways.
"""

# modelling single neuron with 3 inputs and 1 output
inputs = [1,2,3]
weights = [0.2, 0.8, -0.5]
bias = 2

output = inputs[0]*weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2] + bias
# 2.3

# modelling single neuron with 4 inputs and 1 output
inputs = [1, 2, 3, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2

output = inputs[0]*weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2]+ inputs[3]*weights[3] + bias
# 4.8

# modelling 3 neurons with 4 inputs and 3 outputs
inputs = [1.0, 2.0, 3.0, 2.5]
weights1 = [0.2, 0.8, -0.5, 1.0] # weights for 1st neuron
weights2 = [0.5, -0.91, 0.26, -0.5] # weights for 2nd neuron
weights3 = [-0.26, -0.27, 0.17, 0.87] # weights for 3rd neuron
bias1 = 2 # bias for 1st neuron
bias2 = 3 # bias for 2nd neuron
bias3 = 0.5 # bias for 3rd neuron

output = [inputs[0]*weights1[0] + inputs[1]*weights1[1] + inputs[2]*weights1[2]+ inputs[3]*weights1[3] + bias1, # output of 1st neuron
            inputs[0]*weights2[0] + inputs[1]*weights2[1] + inputs[2]*weights2[2]+ inputs[3]*weights2[3] + bias2, # output of 2nd neuron
            inputs[0]*weights3[0] + inputs[1]*weights3[1] + inputs[2]*weights3[2]+ inputs[3]*weights3[3] + bias3] # output of 3rd neuron
print(output) # [4.8, 1.21, 2.385]