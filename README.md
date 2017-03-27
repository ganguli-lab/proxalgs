## Proximal Algorithms
Proximal algorithms and operators in python

[![Build Status](https://travis-ci.org/ganguli-lab/proxalgs.svg?branch=master)](https://travis-ci.org/ganguli-lab/proxalgs)
[![Coverage Status](https://coveralls.io/repos/ganguli-lab/proxalgs/badge.svg?branch=master&service=github)](https://coveralls.io/github/ganguli-lab/proxalgs?branch=master)
![PyPI](https://img.shields.io/pypi/v/proxalgs.svg)

by [Niru Maheswaranathan](http://niru.org/) :soccer:

## Installation
```bash
pip install proxalgs
```

## Overview
Proxalgs is a package for performing convex optimization in python.

Example code for solving l1-regularized least-squares:
```python
>>> from proxalgs import Optimizer
>>> # we want to solve: min ||Ax - b||_2^2 + \gamma ||x||_1
>>> opt = Optimizer('linsys', P=(A.T @ A), q=(A.T @ b))       # main objective (least squares linear system)
>>> opt.add_regularizer('sparse', gamma=0.1)                  # regularizer (l1-norm) with penalty of 0.1
>>> x_hat = opt.minimize(x_init)                              # x_init can be any initialization (e.g. random)
```
