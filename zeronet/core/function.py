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


def linear_backward(x, weights, dout):
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
    w, b = weights
    dx, dw, db = None, None, None
    dx = np.dot(dout, w.T)
    dx = dx.reshape(x.shape)
    batch_size = x.shape[0]
    input_size = np.prod(x.shape[1:])
    x = x.reshape(batch_size, input_size)
    dw = np.dot(x.T, dout)
    db = np.dot(dout.T, np.ones(batch_size))
    grads = (dx, dw, db)
    return grads


def conv_forward(x, weights, conv_params):
    '''
        The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    '''
    out = None
    w, b = weights
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']
    H_out = int(1 + (H + 2 * pad - HH) / stride)
    W_out = int(1 + (W + 2 * pad - WW) / stride)
    out = np.zeros((N , F , H_out, W_out))

    x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)), mode='constant', constant_values=0)
    for i in range(H_out):
        for j in range(W_out):
            x_pad_masked = x_pad[:, :, i*stride:i*stride+HH, j*stride:j*stride+WW] # slide window
            for k in range(F): # # of filters
                out[:, k, i, j] = np.sum(x_pad_masked * w[k, :, :, :], axis=(1,2,3))

    out = out + (b)[None, :, None, None] # add bias
    return out



def conv_backward(x, weights, conv_params, dout):
    '''
    Inputs:
    - dout: Upstream derivatives.

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    '''

    w, b = weights
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']
    H_out = int(1 + (H + 2 * pad - HH) / stride)
    W_out = int(1 + (W + 2 * pad - WW) / stride)
    x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)), mode='constant', constant_values=0)

    dx = np.zeros_like(x)
    dx_pad = np.zeros_like(x_pad)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    db = np.sum(dout, axis(0, 2, 3))

    for i in range(H_out):
        for j in range(W_out):
            x_pad_masked = x_pad[:, :, i*stride:i*stride+HH, j*stride:j*stride+WW]
            for k in range(F):
                dw[k, :, :, :] += np.sum(x_pad_masked * (dout[:, k, i, j])[:, None, None, None], axis=0) #?
            for n in range(N):
                dx_pad[n, :, i*stride:i*stride+HH, j*stride:j*stride+WW] += np.sum((w[:, :, :, :] * 
                                                                    (dout[n, :, i, j])[:, None, None, None]), axis=0)
    dx = dx_pad[:, :, pad:-pad, pad:-pad]
    grads = (dx, dw, db)
    return grads