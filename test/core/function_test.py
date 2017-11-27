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


# Conv forward
x_shape = (2, 3, 4, 4)
w_shape = (3, 3, 4, 4)
x = np.linspace(-0.1, 0.5, num=np.prod(x_shape)).reshape(x_shape)
w = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w_shape)
b = np.linspace(-0.1, 0.2, num=3)

conv_param = {'stride': 2, 'pad': 1}
out = conv_forward(x, (w, b), conv_param)
correct_out = np.array([[[[-0.08759809, -0.10987781],
                          [-0.18387192, -0.2109216]],
                         [[0.21027089, 0.21661097],
                          [0.22847626, 0.23004637]],
                         [[0.50813986, 0.54309974],
                          [0.64082444, 0.67101435]]],
                        [[[-0.98053589, -1.03143541],
                          [-1.19128892, -1.24695841]],
                         [[0.69108355, 0.66880383],
                          [0.59480972, 0.56776003]],
                         [[2.36270298, 2.36904306],
                          [2.38090835, 2.38247847]]]])

assert rel_error(out, correct_out) < 1e-7

# Conv backward

np.random.seed(231)
x = np.random.randn(4, 3, 5, 5)
w = np.random.randn(2, 3, 3, 3)
b = np.random.randn(2,)
dout = np.random.randn(4, 2, 5, 5)
conv_param = {'stride': 1, 'pad': 1}

dx_num = eval_numerical_gradient_array(
    lambda x: conv_forward(x, (w, b), conv_param), x, dout)
dw_num = eval_numerical_gradient_array(
    lambda w: conv_forward(x, (w, b), conv_param), w, dout)
db_num = eval_numerical_gradient_array(
    lambda b: conv_forward(x, (w, b), conv_param), b, dout)

out = conv_forward(x, (w, b), conv_param)
grads = conv_backward(x, (w, b), conv_param, dout)
dx, dw, db = grads['x'], grads['w'], grads['b']


assert rel_error(dx, dx_num) < 1e-7
assert rel_error(dw, dw_num) < 1e-7
assert rel_error(db, db_num) < 1e-7


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
