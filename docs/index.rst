.. ZeroNet documentation master file, created by
   sphinx-quickstart on Tue Dec  5 21:49:39 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

ZeroNet: 基于NumPy的最小可用神经网络框架
================================================

.. toctree::
   :maxdepth: 1
   :caption: 目录：

   about.md
   install.md
   overview.md
   function.md
   layer.md
   net.md
   model.md

=============
简介
=============

ZeroNet是一个完全基于numpy的纯手打、轻量级、几乎什么功能都没有的神经网络框架。它可以作为你了解深度学习库的简易起点。麻雀虽小，五脏俱全。

特点
------------

- 基本网络层的numpy朴素实现（全连接、卷积、ReLU等）
- 网络结构构建类和数据组织
- 训练逻辑组织

缺点
-----------

- 无GPU支持，朴素实现，效率低下
- 无计算资源的抽象，资源利用低
- 目前仅支持单序列的网络结构，不支持RNN等

演示
--------------

1.拷贝代码到本地： ``git clone https://github.com/ddlee96/Zeronet.git``

2.安装依赖

需要： Numpy, Jupyter(演示用), Matplotlib(演示用)

(可选) 使用 Pipenv_ 安装依赖：

.. _Pipenv: https://docs.pipenv.org 

.. code-block:: bash

    # install Pipenv
    pip install pipenv

    # install dependencies, pipenv will install packages based on Pipfile.lock
    cd path/to/zeronet
    pipenv install


3.准备数据集(CIFAR-10)：``bash data/get_cifar.sh``

4.演示

用 ``Jupyter notebook`` 打开 ``demo/demo.ipynb`` 即可。

源码
-------------
源码地址： https://github.com/ddlee96/ZeroNet
问题反馈： https://github.com/ddlee96/ZeroNet/issues/new
