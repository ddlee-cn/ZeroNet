# ZeroNet

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat)](http://makeapullrequest.com)[![Documentation Status](https://readthedocs.org/projects/zeronet-docs/badge/?version=latest)](http://zeronet-docs.readthedocs.io/?badge=docs)

A minimal framework for nerual network learning based on numpy.

[中文简介](https://github.com/ddlee96/ZeroNet/wiki/ZeroNet:-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%AD%A6%E4%B9%A0%E7%9A%84%E6%9C%80%E5%B0%8F%E5%8F%AF%E7%94%A8%E8%8C%83%E4%BE%8B)

## Features

- Naive implementation of basic layers like FC, Conv, ReLU, etc
- Four-layer abstraction for nerual network model: Function, Layer, Net, Model
- Demo scripts with Jupyter Notebook

## Usage


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
