## Function
`function`函数位于`zeronet.core.function`，是完成计算的幕后英雄。

每个函数成对出现，`_forward()`负责前向计算，`_backward()`负责后向传播梯度。

### `_foward()`

前向函数完成网络计算中的前向部分，输入是来自上层的输出`x`，本层参数`weights`和超参数。

线性层的超参已经蕴含在`weights`中（即输出的维度），而向卷积层的超参，`kernel_size`蕴含在`weights`中，而`stride`和`pad`则组成一个字典，传入`_forward()`函数。

### `_backward()`

后向函数完成反向传播的核心部分：接受后一层传来的数据梯度，计算本层数据的梯度、参数的梯度。

与前向函数类似，输入包括后一层传来的数据梯度`dout`、本层数据输入`x`、本层参数`weights`和本层超参。

### 目前支持的函数列表
- `linear_forward(x, weights)`, `linear_backward(x, weights, dout)
- `conv_forward(x, weights, conv_params)`, `conv_backward(x, weights, conv_params, dout)`
- `max_pool_forward(x, pool_param)`, `max_pool_backward(x, pool_param, dout)`
- `relu_forward(x)`, `relu_backward(x, dout)`
- `sigmoid_forward(x)`, `sigmoid_backward(x, dout)`

更多关于这些函数实现的信息请参见[博文](https://blog.ddlee.cn/)和[源码](https://github.com/ddlee96/ZeroNet/blob/master/zeronet/core/function.py)。