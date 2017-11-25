## Author: David Lee, me@ddlee.cn
import numpy as np


def linear_forward(x, weights):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    w, b = weights
    out = None
    batch_size = x.shape[0]
    input_size = np.prod(x.shape[1:])
    x_ = x.reshape(batch_size, input_size)
    out = np.dot(x_, w) + b
    return out


def linear_backward(dout, input, weights):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x = input
    w, b = weights
    dx, dw, db = None, None, None
    dx = np.dot(dout, w.T)
    dx = dx.reshape(x.shape)
    batch_size = x.shape[0]
    input_size = np.prod(x.shape[1:])
    x = x.reshape(batch_size, input_size)
    dw = np.dot(x.T, dout)
    db = np.dot(dout.T, np.ones(batch_size))
    return dx, dw, db