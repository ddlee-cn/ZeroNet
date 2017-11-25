## The expected api for zeronet

import numpy as np
import zeronet as zn

from zeronet.core.layer import Linear, ReLU
from zeronet.core.optimizer import sgd
from zeronet.core.model import model
from zeronet.core.loss import MSELoss

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold inputs and outputs, and wrap them in Variables.
x = np.random.normal()
y = np.random

# Define the network
net = [Linear(hidden_1), Linear(hidden_2), ReLU()]

# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.
loss_fn = MSELoss()
learning_rate = 1e-4
optimizer = sgd(lr = learning_rate)

model = model(net, input_data, optimizer, loss_fn)

model.fit()

