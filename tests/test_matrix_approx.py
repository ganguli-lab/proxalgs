"""
Test suite for matrix approximation using proximal algorithms

"""

# imports
from nose import with_setup
from nose.tools import ok_
from proxalgs import Optimizer
import numpy as np

def generate_lowrank_matrix(n=10, m=20, k=3, eta=0.05, seed=1234):
    """
    Generate an n-by-m noisy low-rank matrix

    """
    print("Generating data for low rank matrix approximation")
    global x_true, x_obs

    # define the seed
    np.random.seed(seed)

    # the true low-rank matrix
    x_true = np.sin(np.linspace(0, 2 * np.pi, n)).reshape(-1, 1).dot(
             np.cos(np.linspace(0, 2 * np.pi, m)).reshape(1, -1))

    # the noisy, observed matrix
    x_obs = x_true + eta * np.random.randn(n, m)

@with_setup(generate_lowrank_matrix)
def test_lowrank_matrix_approx():
    """
    Test low rank matrix approximation

    """

    # proximal algorithm for low rank matrix approximation
    opt = Optimizer('fronorm', x_obs=x_obs)
    opt.add_regularizer('nucnorm', gamma=0.2)
    x_hat = opt.minimize(x_obs, num_iter=100)

    test_err = np.linalg.norm(x_hat - x_true, 'fro')
    naive_err = np.linalg.norm(x_obs - x_true, 'fro')
    err_ratio = test_err / naive_err
    print("The error ratio is: %5.4f" % err_ratio)

    ok_(err_ratio <= 0.5)