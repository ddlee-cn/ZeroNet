# Author: David Lee, me@ddlee.cn
from zeronet.core.layer import *
from zeronet.core.function import *
import numpy as np

class net(object):
    '''
    the net class is the abstraction of DAG composed by stacked layers
    it call forward and backward methods of layers recursively'''

    def __init__(self, layer_stack, loss_func, reg):
        '''
        layer stack is a list of layers with specific order
        layer stack start with a Input Layer, loss function not included
        '''
        self.layer_stack = layer_stack
        self.layer_mount = len(layer_stack)
        self.loss_func = loss_func
        self.reg = reg
        self.input_dict = None
        self.optim_configs = None
        self.params = None
        self.DEBUG = False

    def warmup(self, warmup_data, config):
        '''
        init params using warmup_data, build param dict and optim_config dict for every param
        '''
        self.optim_configs = {}
        self.params = {}
        self.input_dict = {}
        out = None
        for k, _layer in enumerate(self.layer_stack):
            self.input_dict[_layer] = None
            self.params[_layer.name] = {}
            if k != 0:
                _layer.warmup(out)
                out = _layer.forward(out)
            else:
                _layer.warmup(warmup_data)
                out = _layer.forward(warmup_data)
            for p, param in _layer.params.items():
                d = {k: v for k, v in config.items()}
                self.optim_configs[_layer.name + '_' + p] = d
                self.params[_layer.name][p] = param

    def forward(self, data_batch):
        '''
        perform computation layer by layer, recurstively.
        '''
        out = None
        for k, _layer in enumerate(self.layer_stack):
            if k != 0:
                self.input_dict[_layer.name] = out
                out = _layer.forward(out)
            else:
                self.input_dict[_layer.name] = data_batch
                out = _layer.forward(data_batch)
            if self.DEBUG:
                print(_layer.name)
                print('input_shape', self.input_dict[_layer.name].shape)
                print('output_shape', out.shape)
        if self.DEBUG:
            print(out)
        return out

    def loss(self, X_batch, y_batch):
        '''
        wrapper of forward
        '''
        out = self.forward(X_batch)
        loss, dout = self.loss_func(out, y_batch)
        # add regularization
        reg_loss = 0
        for layer_param in self.params.values():
            for value in layer_param.values():
                reg_loss += np.sum(value ** 2)
        loss += self.reg * 0.5 * reg_loss
        return (loss, dout)

    def backward(self, optimizer, dout):
        '''
        perform back propagation and update params recurstively
        currently, just reverse the layer stack and caculate from last layer
        '''
        # wrap dout from loss_function
        dout = dict({'x': dout})
        for _layer in self.layer_stack[::-1]:
            dout = _layer.grad(self.input_dict[_layer.name], dout['x'])
            self.optim_configs = _layer.update(dout, optimizer, self.optim_configs)