## 网络Net

`net`类是对网络结构的抽象，完成对层的组织和数据流的控制，目前尽支持单方向无分叉的序列结构。

#### 初始化 `net(layer_stack=list(), loss_func, reg)`

建立`net`对象需要层的顺序列表（每一项都是`layer`对象），损失函数`loss_func`和正则化参数`reg`。

`loss_func`为`function`模块中定义的函数，目前支持`svm_loss(x, y)`和`softmax_loss(x, y)`。

`loss_func`同时返回损失和反向梯度，是前向计算的终点和反向传播的起点。

#### `forward(data_batch)`

`forward()`方法完成前向计算部分，递归调用每一层的`forward()`方法并缓存每一层的输入。

#### `loss(X_batch, y_batch)`

`loss()`方法是对`forward()`的包装，接收`label`数据计算损失并返回反向梯度。

训练时调用`loss()`方法，推断时调用`forward()`方法。

#### `backward(optimizer, dout)`

`backward()`方法完成反向传播过程，流数据为`loss()`传过来的反向梯度和`forward()`过程缓存的每一层输入，调用`layer`对象的`grad()`和`update()`方法来计算梯度并更新习得参数。

了解更多有关`net`的作用，参见[设计理念](./overview.html)。