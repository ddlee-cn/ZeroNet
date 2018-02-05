.. ZeroNet documentation master file, created by
   sphinx-quickstart on Tue Dec  5 21:49:39 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

ZeroNet: 完全基于NumPy、轻量级的神经网络框架
================================================

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   about.md
   install.md
   overview.md
   function.md
   layer.md
   net.md
   model.md

=============
Introduction
=============

ZeroNet是一个基于numpy的纯手打、轻量级、几乎什么功能都没有的神经网络框架。

## 演示


1.Clone repo
```
git clone https://github.com/ddlee96/Zeronet.git
```
2.Prepare enviroment

Requirement: Numpy, Jupyter(for demo), Matplotlib(for demo)

(optional) Prepare env using [Pipenv](https://docs.pipenv.org)

```
# install Pipenv
pip install pipenv

# install dependencies, pipenv will install packages based on Pipfile.lock
cd path/to/zeronet
pipenv install
```

3.Get Dataset(CIFAR-10)
```
bash data/get_cifar.sh
```

4.Try Demo
Start Jupyter notebook and open `demo/demo.ipynb`.

