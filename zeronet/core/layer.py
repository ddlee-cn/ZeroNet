## author: David Lee, me@ddlee.cn
## there are two special layers, input and loss, 
## which stand for the start and the end of a network

import numpy as np
import defaultdict
from zeronet.core.function import *

class layer(object):
	'''The base class for a layer in network.
	It contains parameters, forward and backward operation,
	as well as connection info(TODO)

	state dict:
	config should include super-params for this layer
	shape_dict contains names and shapes for params inside layer
	params contains names and value for params insied layer
	'''

	def __init__(config):
		self.config = config
		self.shape_dict = defaultdict()
		self.params = defaultdict()

	def _infer_shape(self, warmup_data):
		'''
		infer param shape using warmup data and configs
		return a dict with name and shape for
		params initiation'''
		raise NotImplementedError


	def _init_params(self):
		'''initiate weights inside the layer, layer.params should be a dict
		with param names and values'''
		for name, shape in self.shape_dict.iter_items():
			self.params[name] = np.random.normal(mu, std, size=shape)


	def warm_up(self, warmup_data):
		'''wrapper to be called
		'''
		self._infer_shape(warmup_data)
		self._init_params()



    def forward(self, input):
        '''Defines the computation performed at every call.

        Should be overriden by all subclasses.
        '''
        raise NotImplementedError


    def grad(self, input, dout):
	    '''Return the gradients of this layer, in a dict

	    Should be overriden by all subclasses.
	    '''
    	raise NotImplementedError

    def update(self, optimizer):
    	'''Update the params inside this layer,
    	should be called after backward process.
    	optimizer should be a partial function after configuration
    	in the model class'''
    	for name, param in self.params.iter_items():
    		param_grad = grads[name]
    		next_param = optimizer(param, param_grad)
    		self.params[name] = next_param





class Linear(layer):
	'''shape is a tuple that define the out shape of this FC layer,
	'''
	def __init__(self, config):
		'''for linear(FC) layer, config is an int 
		which specifies the output shape
		'''
		self.output_shape = self.config



	def _infer_shape(self, warmup_data):
		'''infer input and output shape to initiate weights,
		before the layer sees input data
		??? how to
		solution: recursively call this method in a init forward pass
		'''
		self.batch_size = self.warmup_data.shape[0]
		self.input_shape = np.prod(self.warmup_data.shape[1:])
		self.shape_dict['w'] = (self.input_shape, self.output_shape)
		self.shape_dict['b'] = self.output_shape


	def forward(self, input):
    	self.out = linear_foward(input, self.weights)


	def grad(self, input, dout):
		self.grads = linear_backward(input, self.weights, dout)