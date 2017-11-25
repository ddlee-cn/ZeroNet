## author: David Lee, me@ddlee.cn
## there are two special layers, input and loss, 
## which stand for the start and the end of a network

import numpy as np
from zeronet.core.function import *

class layer(object):
	'''The base class for a layer in network.
	It contains parameters, forward and backward operation,
	as well as connection info

	state dict:
	batch_size, input_shape, output_shape, 
	input, out, grads
	'''

	def __init__():
		pass

	def _infer_shape():
		pass

	def _init_params():
		'''initiate weights inside the layer'''

    def forward(self):
        '''Defines the computation performed at every call.

        Should be overriden by all subclasses.
        '''
        raise NotImplementedError


    def grad(self):
	    '''Return the gradients of this layer

	    Should be overriden by all subclasses.
	    '''
    	raise NotImplementedError

    def update(self, optimizer):
    	'''Update the params inside this layer,
    	should be called after backward process.'''



class Linear(layer):
	'''shape is a tuple that define the out shape of this FC layer,
	'''
	def __init__(self, shape):
		'''nearly do nothing when creating layer class
		'''
		self.input = input
		self.output_shape = shape



	def _infer_shape(self, warmup_data):
		'''infer input and output shape to initiate weights,
		before the layer sees input data
		??? how to
		solution: recursively call this method in a init forward pass
		'''
		self.batch_size = self.warmup_data.shape[0]
		self.input_shape = np.prod(self.warmup_data.shape[1:])

	def _init_weights(self):
		'''
		w: A numpy array of weights, of shape (D, M)
	    b: A numpy array of biases, of shape (M,)'''
	    mu = 0
	    std = 0.5
		w_shape = (self.input_shape, self.output_shape)
		w = np.random.normal(mu, std, w_shape)
		b_shape = self.output_shape
		b = np.random.normal(mu, std, b_shape)
		self.weights = (w, b)

	def forward(self, input):
		'''
		perform forward pass. 
		after forward pass, the input data is contained in layer 
		for gradient caculation
		'''
		self.input = input
    	self.out = linear_foward(self.input, self.weights)


	def grad(self, dout):
		dx, dw, db = linear_backward(dout, self.input, self.weights)
	    self.grads = (dx, dw, db)