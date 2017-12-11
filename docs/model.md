## 模型 model

`model`类封装了训练和推断的逻辑。主要功能是准备数据（划分batch）、初始化网络（根据传入的`net`对象）和管理训练过程。

#### 初始化 `model(net, data, **kargs)`

必选参数：

-  `net`： 定义网络结构的`net`对象
- `data`： 训练或测试用数据，`dict`对象，包括`X_train`, `X_val`, `y_train`, `y_val`

可选参数：

- `update_rule`：优化器函数
- `optim_config`：Dict, 优化器的初始配置
- `lr_decay`：Float, 学习率调控
- `batch_size`：Int, 批次大小，默认100
- `num_epochs`：Int, 训练批次，默认10
- `print_every`：Int, 打印频次，默认10
- `verbose`：Boolean, 是否显示进度，默认`True`
- `num_train_sample`： Int, 训练用样本数，默认是1000，设置为`None`则使用传入的全部数据
- `num_val_sample`： Int，验证集样本数，默认`None`，即使用全部验证样本
- `checkpoint_name`：存档点路径及名称

#### `warmup()`

`model`对象建立后即可执行，传入一个batch的数据来初始化网络的习得参数。

#### `train()`

`train`方法完成训练过程的逻辑，首先是划分数据的批次，完成模型的一次参数更新，在一个epoch结束后在验证集上检测结果，保存存档点和当前最佳参数。

参数更新用到内部方法`_step()`，核心逻辑如下：

```
# foward pass
loss, dout = self.net.loss(X_batch, y_batch)
# backward pass
self.net.backward(self.optimizer, dout)
```

#### `predict(X)` 和 `check_accuracy(y_pred, y)`

推断时，传入测试数据后，由`predict()`方法可以得到`y_pred`，之后连同真实值调用`check_accuracy()`方法即可得到正确率。

了解更多有关`model`的作用，参见[边界划分与设计理念](./overview.html)。