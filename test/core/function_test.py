from zeronet.core.function import *
import numpy as np
from ..gradient_check import *

# linear_forward
num_inputs = 2
input_shape = (4, 5, 6)
output_dim = 3

input_size = num_inputs * np.prod(input_shape)
weight_size = output_dim * np.prod(input_shape)

x = np.linspace(-0.1, 0.5, num=input_size).reshape(num_inputs, *input_shape)
w = np.linspace(-0.2, 0.3,
                num=weight_size).reshape(np.prod(input_shape), output_dim)
b = np.linspace(-0.3, 0.1, num=output_dim)

out = linear_forward(x, (w, b))
correct_out = np.array([[1.49834967, 1.70660132, 1.91485297],
                        [3.25553199, 3.5141327, 3.77273342]])

assert rel_error(out, correct_out) < 1e-8, 'linear_foward failed'

# linear_backward
np.random.seed(231)
x = np.random.randn(10, 2, 3)
w = np.random.randn(6, 5)
b = np.random.randn(5)
dout = np.random.randn(10, 5)

dx_num = eval_numerical_gradient_array(
    lambda x: linear_forward(x, (w, b)), x, dout)
dw_num = eval_numerical_gradient_array(
    lambda w: linear_forward(x, (w, b)), w, dout)
db_num = eval_numerical_gradient_array(
    lambda b: linear_forward(x, (w, b)), b, dout)

out = linear_forward(x, (w, b))
grads = linear_backward(x, (w, b), dout)
dx = grads['x']
dw = grads['w']
db = grads['b']

assert rel_error(dx_num, dx) < 1e-9
assert rel_error(dw_num, dw) < 1e-9
assert rel_error(db_num, db) < 1e-9


# relu_forward
x = np.linspace(-0.5, 0.5, num=12).reshape(3, 4)

out = relu_forward(x)
correct_out = np.array([[0., 0., 0., 0., ],
                        [0., 0., 0.04545455, 0.13636364, ],
                        [0.22727273, 0.31818182, 0.40909091, 0.5, ]])

assert rel_error(out, correct_out) < 1e-7

# relu_backward
np.random.seed(231)
x = np.random.randn(10, 10)
dout = np.random.randn(*x.shape)

dx_num = eval_numerical_gradient_array(lambda x: relu_forward(x), x, dout)

out = relu_forward(x)
grads = relu_backward(out, dout)
dx = grads['x']

assert rel_error(dx_num, dx) < 1e-7


# Loss test
np.random.seed(231)
num_classes, num_inputs = 10, 50
x = 0.001 * np.random.randn(num_inputs, num_classes)
y = np.random.randint(num_classes, size=num_inputs)

dx_num = eval_numerical_gradient(lambda x: svm_loss(x, y)[0], x, verbose=False)
loss, dx = svm_loss(x, y)

assert rel_error(loss, 9) < 0.1
assert rel_error(dx_num, dx) < 1e-7

dx_num = eval_numerical_gradient(
    lambda x: softmax_loss(x, y)[0], x, verbose=False)
loss, dx = softmax_loss(x, y)

assert rel_error(loss, 2.3) < 0.1
assert rel_error(dx_num, dx) < 1e-7
