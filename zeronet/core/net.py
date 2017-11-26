## Author: David Lee, me@ddlee.cn
from layer import *

class net(object):
	'''
	the net class is the abstraction of DAG composed by stacked layers
	it call forward and backward methods of layers recursively'''
	def __init__(self, layer_stack):
		'''
		layer stack is a list of layers with specific order'''
		self.layer_stack = layer_stack
		self.layer_mount = len(layer_stack)
		self.check()

	def check():
		'''
		check layers
		'''
		assert is_instance(layer_stack[0], InputLayer)
		assert is_instance(layer_stack[self.layer_mount], LossLayer)
		for _layer in layer_stack:
			assert is_instacne(_layer, layer)


	def warmup(self, warmup_data):
		'''
		init params using warmup_data
		'''
		for _layer in self.layer_stack:
			_layer.warmup(warmup_data)

	def forward(self, data_batch):
		'''
		perform computation layer by layer, recurstively.
		'''
		for _layer in self.layer_stack:
			if not is_instance(layer, InputLayer):
				out = _layer.forward(out)
			else:
				out = _layer.forward(data_batch)
		return out


	def backward(self, optimizer, loss):
		'''
		perform back propagation and update params recurstively
		'''
		for _layer in self.layer_stack:
			if not is_instance(layer, LossLayer):
				dout = _layer.grad(dout)
			else:
				dout = _layer.grad(loss)
			_layer.update(dout, optimizer)

		return dout