# Author: David Lee, me@ddlee.cn
import numpy as np

__all__ = ['linear_forward', 'linear_backward',
           'conv_forward', 'conv_backward',
           'relu_forward', 'relu_backward',
           'sigmoid_backward', 'sigmoid_backward',
           'svm_loss', 'softmax_loss']


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
    grads = dict({'x': dx, 'w': dw, 'b': db})
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
    stride, pad = conv_params['stride'], conv_params['pad']
    H_out = int(1 + (H + 2 * pad - HH) / stride)
    W_out = int(1 + (W + 2 * pad - WW) / stride)
    out = np.zeros((N, F, H_out, W_out))

    x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)),
                   mode='constant', constant_values=0)
    for i in range(H_out):
        for j in range(W_out):
            x_pad_masked = x_pad[:, :, i * stride:i * stride +
                                 HH, j * stride:j * stride + WW]
            # slide window
            for k in range(F):  # of filters
                out[:, k, i, j] = np.sum(
                    x_pad_masked * w[k, :, :, :], axis=(1, 2, 3))

    out = out + (b)[None, :, None, None]  # add bias
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
    stride, pad = conv_params['stride'], conv_params['pad']
    H_out = int(1 + (H + 2 * pad - HH) / stride)
    W_out = int(1 + (W + 2 * pad - WW) / stride)
    x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)),
                   mode='constant', constant_values=0)

    dx = np.zeros_like(x)
    dx_pad = np.zeros_like(x_pad)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    db = np.sum(dout, axis=(0, 2, 3))

    for i in range(H_out):
        for j in range(W_out):
            x_pad_masked = x_pad[:, :, i * stride:i *
                                 stride + HH, j * stride:j * stride + WW]
            for k in range(F):
                dw[k, :, :, :] += np.sum(x_pad_masked * (dout[:, k, i, j])
                                         [:, None, None, None], axis=0)  # ?
            for n in range(N):
                dx_pad[n, :, i * stride:i * stride + HH, j * stride:j * stride + WW] += np.sum((w[:, :, :, :] *
                                                                                                (dout[n, :, i, j])[:, None, None, None]), axis=0)
    dx = dx_pad[:, :, pad:-pad, pad:-pad]
    grads = dict({"x": dx, "w": dw, "b": db})
    return grads


def max_pool_forward(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None
    N, C, H, W = x.shape
    HH, WW, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
    H_out = int((H-HH)/stride+1)
    W_out = int((W-WW)/stride+1)
    out = np.zeros((N, C, H_out, W_out))
    for i in range(H_out):
        for j in range(W_out):
            x_masked = x[:, :, i*stride:i*stride+HH, j*stride:j*stride+WW]
            out[:, :, i, j] = np.max(x_masked, axis=(2,3))
    cache = (x, pool_param)
    return out


def max_pool_backward(x, pool_param, dout):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    N, C, H, W = x.shape
    HH, WW, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
    H_out = int((H-HH)/stride+1)
    W_out = int((W-WW)/stride+1)
    dx = np.zeros_like(x)

    for i in range(H_out):
     for j in range(W_out):
        x_masked = x[:,:,i*stride : i*stride+HH, j*stride : j*stride+WW]
        max_x_masked = np.max(x_masked,axis=(2,3))
        temp_binary_mask = (x_masked == (max_x_masked)[:,:,None,None])
        dx[:,:,i*stride : i*stride+HH, j*stride : j*stride+WW] += temp_binary_mask * (dout[:,:,i,j])[:,:,None,None]
    grads = dict({"dx":dx})
    return grads



def relu_forward(x):
    out = x * (x > 0)
    return out


def relu_backward(x, dout):
    dx = dout * (x > 0)
    grads = dict({'x': dx})
    return grads


def sigmoid_forward(x):
    out = 1 / (1 + np.exp(-x))
    return out


def sigmoid_backward(x, dout):
    dx = dout * (1.0 - dout)
    grads = dict({'x': dx})
    return grads


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
