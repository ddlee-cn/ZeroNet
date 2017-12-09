## Layer

`Layer`类是对网络中单独层的抽象，内聚的数据是习得参数，主要作用是包装`function`的计算功能，完成当前层习得参数的初始化和更新。

### 基类`layer(name='layer')`

初始化时，`layer`需要一个名字标识`name`，内建两个字典`shape_dict`和`params`，前者用于存储习得参数的维度，后者存储习得参数的值。

##### `warmup(self, warmup_data)`

`layer`类提供`warmup()`方法来推断习得参数的维度（`_infer_shape()`）并将其初始化（`_init_params()`），这一过程在模型建立后被调用。

##### `forward(self, input)`和`grad(self, input, dout)`

这两个方法分别完成当前层的前向计算和反向传播过程，是对`function()`的包装。

##### `update(self, grads, optimizer, config)`

`update()`方法是`layer`类的核心功能，其在完成反向传播后被调用，接收`grad()`方法得到的梯度、模型传来的优化器函数（`optimizer`）及其配置，对当前层内聚的习得参数进行一步更新。

### 目前支持的层

#### `Linear(name="linear", output_shape=256)`

`Linear`层即为全连接层，接收数据维度为`(batch_size, X, Y, Z, .....)`，习得参数为权重`w`和偏置`b`。

超参：

- 输出维度`output_shape`


#### `Conv(name='conv', filter=1, kernel_size=3, stride=1, pad=0)`

`Conv`为卷积层，接收数据维度为`(batch_size, channels, height, width)`，内聚的习得参数为卷积权重`w`（维度`(filters, channels, kernel_size, kernel_size)`）和偏置`b`。

超参：

- 过滤器个数`filter`
- 卷积核大小`kernel_size`
- 步长`stride`
- pad长度`pad`

#### `Pool(name="pool", pool_height=2, pool_width=2, stride=2)`

`Pool`层为极大下采样层，不含习得参数。

超参：

- 采样高度`pool_height`
- 采样宽度`pool_width`
- 步长`stride`


#### `ReLU(name='relu')`和`Sigmoid(name='sigmoid')`

`ReLU`和`Sigmoid`为非线性激活层，不含习得参数。


了解更多有关`layer`的作用，参见[边界划分与设计理念](./overview.html)。