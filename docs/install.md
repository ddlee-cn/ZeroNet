## 安装ZeroNet

现阶段，`ZeroNet`可由github源码调用。

```
git clone https://github.com/ddlee96/ZeroNet.git
```

在`demo`文件夹下建立Jupyter Notebook或者Python脚本，加入如下内容：

```
import os
import sys
sys.path.insert(0, os.path.join("..\"))
```

之后，便可以用`from zeronet.core.layer import Linear`的形式使用。更多细节请参见文档的其他部分