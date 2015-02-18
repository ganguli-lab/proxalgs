==========
Quickstart
==========

Background
----------

This package provides a set of python modules for solving convex optimization problems. Specifically, it is useful for solving problems of the form:

.. math::

    \text{minimize} f(x) + \sum_i g_i(x)

Where :math:`f(x)` and :math:`g_i(x)` are (potentially non-smooth), closed convex functions. Many relevant problems can be formulated
this way where :math:`f(x)` is a data fidelity or log-likelihood term and each :math:`g_i(x)` is a regularization term (or log-prior).

Note that many desirable convex constraints (e.g. restriction to non-negative values) can be incorporated into the objective
by defining a function that is zero over the desired domain and :math:`+\infty` elsewhere.

For more information on convex optimization, see the textbook by Boyd and Vandenberghe [cite].
For more information on proximal algorithms and their applications, see [cite].

Overview
--------
The main class to worry about is ``proxalgs.Optimizer``. It exposes a function, ``minimize``, that will solve your
instantiated problem using a proximal consensus algorithm.

You initialize the optimizer with the name of a function to minimize. A list of compatible functions are in the ``operators`` module.

For example, if the first function in your objective is a squared error term, you can initialize the optimzier as follows:

.. code-block:: python

    >> import proxalgs
    >> opt = proxalgs.Optimzier('squared_error', x_obs=data)

Where ``data`` is some observed data that you want to approximate. Different strings (first argument) correspond to different functions
in the ``operators`` module, and the keyword arguments are necessary keyword arguments for those functions.

We can add as many regularization terms (additional objectives) as we would like using the following syntax:

.. code-block:: python

    >> opt.add_regularizer('sparse', gamma=0.1)
    >> opt.add_regularizer('nonneg')

This adds an :math:`\ell_1`-penalty and restricts the solution to be non-negative.
For more information on the available operators, see the ``operators`` module.

We have now created our optimizer, and want to minimize it. We can do this by calling ``minimize``:

.. code-block:: python

    >> x_hat = opt.minimize(x_init)

Where the argument we passed in is the initial parameter values. The solution found is returned, and additional information
about convergence is stored in the ``opt.results`` dictionary.

Demo - sparse regression
------------------------

.. note:: This example follows the test available in ``tests/test_sparse_regression.py``. See the code for more detail!

Let's say we want to solve a problem of the following form:

.. math::

    \text{minimize} |Ax-b|_2^2 + \gamma |x|_1

The first term in the problem is a squared error term, and we tack on a sparsity penalty term (via the :math:`\ell_1` norm).

We instantiate the optimizer using a least squares solver, and add a sparsity penalty before minimizing:

.. code-block:: python

    >> opt = Optimizer('linsys', P=A.T.dot(A), q=A.T.dot(x_obs))
    >> opt.add_regularizer('sparse', gamma=0.1)
    >> x_hat = opt.minimize(np.random.randn(x_obs.size), num_iter=100, disp=2)

We can check for convergence by making sure the primal and dual residuals in ``opt.results['residuals']`` go to zero.