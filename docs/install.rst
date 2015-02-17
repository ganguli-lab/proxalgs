============
Installation
============

Basic
-----

The fastest way to install is by grabbing the code from Github:

.. code-block:: bash

    git clone https://github.com/ganguli-lab/proxalgs.git
    cd proxalgs
    pip install -r requirements.txt
    python setup.py install

Dependencies
------------

Proxalgs requires that you have ``numpy``, ``scipy`` and ``hyperopt`` installed. The first two are part of the standard
scientific python stack. ``hyperopt`` (https://github.com/hyperopt/hyperopt) is a package for performing optimization over hyperparameters.

Development
-----------

To contribute to ``proxalgs``, you'll need to also install ``sphinx`` for documentation and
``nose`` for testing. We adhere to the `NumPy/SciPy documentation standards <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt#docstring-standard>`_.


Bugs
----

Please report any bugs you encounter through the github `issue tracker
<https://github.com/ganguli-lab/proxalgs/issues/new>`_.
