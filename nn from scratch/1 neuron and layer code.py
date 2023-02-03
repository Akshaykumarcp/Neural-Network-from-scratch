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