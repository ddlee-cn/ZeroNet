# Author: David Lee, me@ddlee.cn
from core.layer import *
from core.function import *

class net(object):
    '''
    the net class is the abstraction of DAG composed by stacked layers
    it call forward and backward methods of layers recursively'''

    def __init__(self, layer_stack, loss_func):
        '''
        layer stack is a list of layers with specific order
        layer stack start with a Input Layer, loss function not included
        '''
        self.layer_stack = layer_stack
        self.layer_mount = len(layer_stack)
        self.loss_func = loss_func
        self.optim_configs = None
        self.params = None
        self.check()

    def check(self):
        '''
        check layers
        '''
        assert isinstance(self.layer_stack[0], InputLayer), 'The first layer must be InputLayer'
        for _layer in self.layer_stack:
            assert isinstance(_layer, layer)

    def warmup(self, warmup_data, config):
        '''
        init params using warmup_data, build param dict and optim_config dict for every param
        '''
        self.optim_configs = {}
        self.params = {}
        for _layer in self.layer_stack:
            _layer.warmup(warmup_data)
            for p in _layer.params.keys():
                d = {k: v for k, v in config.items()}
                self.optim_configs[_layer.name + '_' + p] = d
                self.params[_layer.name + '_' + p] = None

    def forward(self, data_batch):
        '''
        perform computation layer by layer, recurstively.
        '''
        out = None
        for _layer in self.layer_stack:
            if not isinstance(layer, InputLayer):
                out = _layer.forward(out)
            else:
                out = _layer.forward(data_batch)
        return out

    def loss(self, X_batch, y_batch):
        '''
        compute loss, wrapper of forward
        '''
        out = self.forward(X_batch)
        loss = self.loss_func(out, y_batch)
        return loss

    def backward(self, optimizer, loss):
        '''
        perform back propagation and update params recurstively
        currently, just reverse the layer stack and caculate from last layer
        '''
        dout = None
        for k, _layer in enumerate(self.layer_stack[::-1]):
            if k != 0:
                dout = _layer.grad(dout)
            else:
                dout = _layer.grad(loss)
            self.optim_configs = _layer.update(dout, optimizer, self.optim_configs)