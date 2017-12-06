## Function
`function`函数位于`zeronet.core.function`，是完成计算的幕后英雄。

每个函数成对出现，`_forward()`负责前向计算，`_backward()`负责后向传播梯度。

### `_foward()`

### `_backward()`

### 目前支持的函数列表
- Linear
- Conv
- Pool
- ReLU
- Sigmoid
- SoftmaxLoss