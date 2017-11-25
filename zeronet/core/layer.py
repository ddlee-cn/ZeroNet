## author: David Lee, me@ddlee.cn

import numpy as np

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

	def _init_weights():
		'''initiate weights inside the layer'''

    def forward(self):
        '''Defines the computation performed at every call.

        Should be overriden by all subclasses.
        '''
        raise NotImplementedError


    def backward(self):
	    '''Return the gradients of this layer

	    Should be overriden by all subclasses.
	    '''
    	raise NotImplementedError



class Linear(layer):
	'''shape is a tuple that define the out shape of this FC layer,
	'''
	def __init__(self, input, shape):
		'''input shape will be inferred in init process
		'''
		self.input = input
		self.output_shape = shape
		self.batch_size = self.input.shape[0]
		self.input_shape = np.prod(self.input.shape[1:])

		self._init_weights()

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

	def forward(self):
		'''The weights cotains w and b,

	    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
	    examples, where each example x[i] has shape (d_1, ..., d_k). We will
	    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
	    then transform it to an output vector of dimension M.

	    Inputs:
	    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)

	    Outputs:
	    - out should have shape (N, M)
		'''
		x = self.input
		x_ = x.reshape(self.batch_size, self.input_shape)
    	w, b = self.weights
    	self.out = np.dot(x_, w) + b


	def backward(self, dout):
		'''Computes the backward pass for FC layer.

	    Inputs:
	    - dout: Upstream derivative, of shape (N, M)
	    - x: Input data, of shape (N, d_1, ... d_k)
	    - w: Weights, of shape (D, M)

	    Returns a tuple of:
	    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
	    - dw: Gradient with respect to w, of shape (D, M)
	    - db: Gradient with respect to b, of shape (M,)

	    TODO: where does dout come from?
	    '''
	    x = self.input
		w, b = self.weights
		dx = np.dot(dout, w.T)
	    dx = dx.reshape(x.shape)
	    x = x.reshape(self.batch_size, self.input_shape)
	    dw = np.dot(x.T, dout)
	    db = np.dot(dout.T, np.ones(self.batch_size))
	    self.grads = (dx, dw, db)