from zeronet.core.layer import *
from zeronet.core.net import net
from zeronet.core.function import softmax_loss
from zeronet.utils.data_utils import get_CIFAR10_data

import numpy as np

layers = [Conv(name="conv1", filter=6, kernel_size=5, stride=2, pad=3),
          ReLU(name="relu1"),
          Pool(name="pool1", pool_height=2, pool_width=2, stride=2),
          Linear(name="fc1", output_shape=256),
          ReLU(name="relu2"),
          Linear(name="fc2", output_shape=128)]

test_net = net(layers, softmax_loss)

data = get_CIFAR10_data(dir="data/cifar-10-batches-py/", num_training=1000, num_validation=100, num_test=100)

N, D = 4, 5
v = np.linspace(0.6, 0.9, num=N * D).reshape(N, D)

config = {'learning_rate': 1e-3, 'velocity': v}

test_net.warmup(data['X_train'], config)

assert test_net.params['conv1']['w'].shape == (6, 3, 5, 5)
assert test_net.params['conv1']['b'].shape == (6,)