"""
Proxalgs
========

A python package for using proximal algorithms

Modules
-------
core        - Main optimizer class
operators   - A list of functions to use as proximal maps

"""

__version__ = '0.2.0'
__author__ = 'nirum'
__all__ = ['core', 'operators']

from .core import Optimizer
