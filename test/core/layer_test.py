import numpy as np
from zeronet.core.layer import *

# (batch_size, channels, height, width)
test_data_batch = np.random.normal(0, 1, (16, 3, 24, 36))

# linear
fc1 = Linear(output_shape=4)
fc1.warmup(test_data_batch)
assert fc1.params['w'].shape == (3 * 24 * 36, 4)
assert fc1.params['b'].shape == (4,)
fc1_out = fc1.forward(test_data_batch)
assert fc1_out.shape == (16, 4)

# conv
conv1 = Conv(filter=6, kernel_size=5, stride=2, pad=3)
conv1.warmup(test_data_batch)
assert conv1.params['w'].shape == (6, 3, 5, 5)
assert conv1.params['b'].shape == (6,)
conv1_out = conv1.forward(test_data_batch)
assert conv1_out.shape == (16, 6, int(
    1 + (24 + 2 * 3 - 5) / 2), int(1 + (36 + 2 * 3 - 5) / 2))
