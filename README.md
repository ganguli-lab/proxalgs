# Proximal Algorithms

_Note: this project has been folded into the more comprehensive [descent](https://github.com/nirum/descent) package._

[![Build Status](https://travis-ci.org/ganguli-lab/proxalgs.svg?branch=master)](https://travis-ci.org/ganguli-lab/proxalgs)
[![Coverage Status](https://coveralls.io/repos/ganguli-lab/proxalgs/badge.svg?branch=master&service=github)](https://coveralls.io/github/ganguli-lab/proxalgs?branch=master)

## Installation
```bash
git clone git@github.com:ganguli-lab/proxalgs.git
cd proxalgs
pip install -r requirements.txt
python setup.py install
```

## Dependencies

Required:
- python 2.7 or higher
- numpy
- scipy
- toolz

Optional:
- scikit-image
- tableprint

## Development
Pull requests welcome! Please stick to the [NumPy/SciPy documentation standards](https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt#docstring-standard)
We use `sphinx` for documentation and `nose` for testing.

## Todo
- more support for operations on unfolded tensors
- parallelization of the various proximal updates

## Contact
Niru Maheswaranathan (nirum@stanford.edu)
