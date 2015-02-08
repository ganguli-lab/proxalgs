"""
Test suite for sparse regression

"""

# imports
from nose import with_setup
from nose.tools import ok_
from proxalgs import Optimizer
import numpy as np


def generate_sparse_vector(n=100, m=50, p=0.1, eta=0.05, seed=1234):
    """
    Generate a sparse, noisy vector

    """
    print("Generating data for sparse regression")
    global x_true, x_obs, A

    # define the seed
    np.random.seed(seed)

    # the true sparse signal
    x_true = 10 * np.random.randn(n) * (np.random.rand(n) < p)

    # the noisy, observed signal
    A = np.random.randn(m,n)
    x_obs = A.dot(x_true)


@with_setup(generate_sparse_vector)
def test_sparse_regression():
    """
    Test sparse regression

    """

    # least squares solution
    xls = np.linalg.lstsq(A, x_obs)[0]

    # proximal algorithm for sparse regression
    opt = Optimizer('linsys', P=A.T.dot(A), q=A.T.dot(x_obs))
    opt.add_regularizer('sparse', gamma=1)
    x_hat = opt.minimize(xls, num_iter=100)

    test_err = np.linalg.norm(x_hat - x_true, 2)
    naive_err = np.linalg.norm(xls - x_true, 2)
    err_ratio = test_err / naive_err
    print("The error ratio is: %5.4f" % err_ratio)

    ok_(err_ratio <= 0.01)