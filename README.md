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
>>> opt = Optimizer('squared_error', x_obs=x_obs)
>>> opt.add_regularizer('sparse', gamma=0.1)
>>> x_hat = opt.minimize(x_init)
```
