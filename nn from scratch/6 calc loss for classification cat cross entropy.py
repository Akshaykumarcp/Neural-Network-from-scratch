# one hot encoding
# log

"""
log

log in general solving for x in equation of e raised to x = b

e ** x = b (input to log is b)

"""

# let's see what log is doing

import numpy as np
import math

b = 5.2

print(np.log(b)) # 1.6486586255873816
print(math.e ** 1.6486586255873816) # 5.199999999999999

# categorical cross entropy

softmax_output = [0.7, 0.1, 0.2]

# calc loss on above

target_output = [1, 0, 0]
# target_class = 0

loss = -(math.log(softmax_output[0]) * target_output[0] + #
        math.log(softmax_output[1]) * target_output[1]+ # 0
        math.log(softmax_output[2])* target_output[2]) # 0

print(loss) # 0.35667494393873245

# because line no 33 and 34 gives 0, formula can be
# complex formula, becomes simple
loss = -math.log(softmax_output[0])
print(loss) # 0.35667494393873245

# when confidence is high, loss is lower
print(-math.log(0.7)) # 0.35667494393873245

# when confidence is low, loss is higher
print(-math.log(0.5)) # 0.6931471805599453


