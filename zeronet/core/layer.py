# author: David Lee, me@ddlee.cn
# there are two special layers, input and loss,
# which stand for the start and the end of a network

import numpy as np
from collections import defaultdict
from core.function import *

__all__ = ['layer', 'Linear', 'Conv', 'ReLU', 'Sigmoid']


class layer(object):
    '''The base class for a layer in network.
    It contains parameters, forward and backward operation,
    as well as connection info(TODO)

    state dict:
    config should include super-params for this layer
    shape_dict contains names and shapes for params inside layer
    params contains names and value for params insied layer
    '''

    def __init__(self, name, *inputs):
        self.name = name
        self.shape_dict = defaultdict(tuple)
        self.params = defaultdict(np.array)

    def _infer_shape(self, warmup_data):
        '''
        infer param shape using warmup data and configs return
        a dict with name and shape for params initiation,
        should be called in a warmup phase after the net is defined
        '''
        raise NotImplementedError

    def _init_params(self):
        '''initiate weights inside the layer, layer.params should be a dict
        with param names and values'''
        mu = 0
        std = 0.5
        for name, shape in self.shape_dict.items():
            self.params[name] = np.random.normal(mu, std, size=shape)

    def warmup(self, warmup_data):
        '''wrapper to be called in warmup phase'''
        self._infer_shape(warmup_data)
        self._init_params()

    def forward(self, input):
        raise NotImplementedError

    def grad(self, input, dout):
        raise NotImplementedError

    def update(self, grads, optimizer, config):
        '''Update the params inside this layer,
            should be called after backward process.
            optimizer should be a partial function after configuration
            in the model class'''
        if len(self.params) == 0:
            # some layer have no params
            return config
        else:
            for name, param in self.params.items():
                # Keep a optim config dict for every param in the net
                param_grad = grads[name]
                cfg = config[self.name + '_' + name]
                next_param, next_config = optimizer(param, param_grad, cfg)
                self.params[name] = next_param
                config[self.name + '_' + name] = next_config
            return config


class Linear(layer):
    '''shape is a tuple that define the out shape of this FC layer,
    '''

    def __init__(self, output_shape):
        '''for linear(FC) layer, config is an int
        which specifies the output shape
        '''
        self.output_shape = output_shape
        super(Linear, self).__init__()

    def _infer_shape(self, warmup_data):
        self.batch_size = warmup_data.shape[0]
        self.input_shape = np.prod(warmup_data.shape[1:])
        self.shape_dict['w'] = (self.input_shape, self.output_shape)
        self.shape_dict['b'] = self.output_shape

    def forward(self, input):
        out = linear_forward(input,
                             (self.params['w'], self.params['b']))
        return out

    def grad(self, input, dout):
        grads = linear_backward(
            input, (self.params['w'], self.params['b']), dout)
        return grads


class Conv(layer):
    def __init__(self, filter=1, kernel_size=3, stride=1, pad=0):
        self.filter = filter
        self.kernel_size = kernel_size
        self.conv_params = defaultdict(int)
        self.conv_params['stride'] = stride
        self.conv_params['pad'] = pad
        super(Conv, self).__init__()

    def _infer_shape(self, warmup_data):
        '''
        the input should be in shape (N, C, H, W)
        (batch_size, channels, Height, Width)'''
        self.batch_size = warmup_data.shape[0]
        self.channels = warmup_data.shape[1]
        self.shape_dict['w'] = (
            self.filter, self.channels, self.kernel_size, self.kernel_size)
        self.shape_dict['b'] = (self.filter, )

    def forward(self, input):
        out = conv_forward(
            input, (self.params['w'], self.params['b']), self.conv_params)
        return out

    def grad(self, input, dout):
        grads = conv_backward(
            input, (self.params['w'], self.params['b']),
            self.conv_params, dout)
        return grads


class Pool(layer):
    def __init__(self, pool_height=2, pool_width=2, stride=2):
        self.pool_params = defaultdict(int)
        self.pool_params['pool_height'] = pool_height
        self.pool_params['pool_width'] = pool_width
        self.pool_params['stride'] = stride
        super(Conv, self).__init__()

    def _infer_shape(self, warmup_data):
        pass

    def forward(self, input):
        out = max_pool_forward(input, self.pool_params)
        return out

    def grad(self, input, dout):
        grads = max_pool_backward(input, self.pool_params, dout)
        return grads


class ReLU(layer):
    def __init__(self):
        pass

    def _infer_shape(self, warmup_data):
        pass

    def forward(self, input):
        return relu_forward(input)

    def grad(self, input, dout):
        return relu_backward(input, dout)


class Sigmoid(layer):
    def __init__(self):
        pass

    def _infer_shape(self, warmup_data):
        pass

    def forward(self, input):
        return sigmoid_forward(input)

    def grad(self, input, dout):
        return sigmoid_backward(input, dout)
