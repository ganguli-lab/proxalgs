"""
Proximal operators

Evaluates proximal operators for various functions.

Notes
-----
evaluates expressions of the form:

.. math:: \mathrm{prox}_{f,rho} (x0) = \mathrm{argmin}_x ( f(x) + (rho / 2) ||x-x0||_2^2 )

"""

# imports
import numpy as np
from scipy.optimize import minimize
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve
from .tensor import Tensor

# python 2/3 compatibility
import sys
if sys.version_info < (3,0):
    from builtins import super


def tensor_wrapper(axis):
    def decorate(func):
        def wrapper(x, *args, **kwargs):
            xu = Tensor(x).unfold(axis)
            return func(xu, *args, **kwargs).fold()
        return wrapper
    return decorate


class Operator(object):

    def __init__(self, penalty):
        # TODO: deal with None
        self.penalty = penalty #float(penalty)

    def __str__(self):
        # TODO: check for name and penalty
        return '{} proximal operator (penalty = {})'.format(self.name, self.penalty)

    def __call__(x0, rho):
        raise NotImplementedError


class sfo(Operator):

    def __init__(self, optimizer, num_steps=50):
        """
        Proximal operator for an arbitrary function minimized via the Sum-of-Functions optimizer (SFO)

        Notes
        -----
        SFO is a function optimizer for the
        case where the target function breaks into a sum over minibatches, or a sum
        over contributing functions. It is
        described in more detail in [1]_.

        Parameters
        ----------
        x0 : array_like
            The starting or initial point used in the proximal update step

        rho : float
            Momentum parameter for the proximal step (larger value -> stays closer to x0)

        optimizer : SFO instance
            Instance of the SFO object in `SFO_admm.py`

        num_steps : int, optional
            Number of SFO steps to take

        Returns
        -------
        theta : array_like
        The parameter vector found after running `num_steps` iterations of the SFO optimizer

        References
        ----------
        .. [1] Jascha Sohl-Dickstein, Ben Poole, and Surya Ganguli. Fast large-scale optimization by unifying stochastic
            gradient and quasi-Newton methods. International Conference on Machine Learning (2014). `arXiv preprint
            arXiv:1311.2115 (2013) <http://arxiv.org/abs/1311.2115>`_.

        """

        self.name = 'SFO'
        self.optimizer = optimizer
        super().__init__(None)

    def __call__(self, x0, rho):

        # set the current parameter value of SFO to the given value
        self.optimizer.set_theta(x0, float(rho))

        # set the previous ADMM location as the flattened paramter array
        self.optimizer.theta_admm_prev = self.optimizer.theta_original_to_flat(x0)

        # run the optimizer for n steps
        return self.optimizer.optimize(num_steps=self.num_steps)


class poissreg(Operator):

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.n = float(x.shape[0])
        self.proj = x.T.dot(y)

    def f_df(self, w):

        u = self.x.dot(w)
        r = np.exp(u)

        obj = np.mean(r - self.y * u)
        grad = (self.x.T.dot(r) - self.proj) / self.n
        return obj, grad

    def __call__(self, x0, rho):

        raise NotImplementedError


class bfgs(Operator):

    def __init__(self, f_df, method='BFGS', maxiter=50, disp=False):
        self.name = 'Custom objective with BFGS'
        self.f_df = f_df
        self.opts = {'maxiter': maxiter, 'disp': disp}
        super().__init__(None)

    def build_objective(self, x0, rho):

        def wrapper(x):
            f, df = self.f_df(x)

            xdiff = x.reshape(x0.shape) - x0
            obj = f + (rho / 2) * np.sum(xdiff ** 2)
            grad = df + rho * xdiff

            return obj, grad

        return wrapper

    def __call__(self, x0, rho):

        # specify the objective function and gradient for the proximal operator
        obj = self.build_objective(x0, rho)

        # minimize via BFGS
        res = minimize(obj, x0, method='BFGS', jac=True, options=self.opts)
        return res.x


class smooth(Operator):

    def __init__(self, penalty):
        self.name = 'Smooth'
        super().__init__(penalty)

    def __call__(self, x0, rho):

        # Apply Laplacian smoothing
        n = x0.shape[0]
        lap_op = spdiags([(2 + rho / self.penalty) * np.ones(n), -1 * np.ones(n),
                          -1 * np.ones(n)], [0, -1, 1], n, n, format='csc')
        x_out = spsolve(self.penalty * lap_op, rho * x0)

        return x_out


class nucnorm(Operator):

    def __init__(self, penalty):
        self.name = 'Nuclear norm'
        super().__init__(penalty)

    def __call__(self, x0, rho):

        # compute SVD
        u, s, v = np.linalg.svd(x0, full_matrices=False)

        # soft threshold the singular values
        sthr = np.maximum(s - (self.penalty / float(rho)), 0)

        # reconstruct
        return u.dot(np.diag(sthr)).dot(v)


class squared_error(Operator):

    def __init__(self, x_obs):
        self.name = 'Squared error'
        self.x_obs = x_obs
        super().__init__(None)

    def __call__(self, x0, rho):
        return (x0 + self.x_obs / rho) / (1 + 1 / rho)


class tvd(Operator):

    def __init__(self, penalty):
        self.name = 'Total variation denoising'
        super().__init__(penalty)

    def __call__(self, x0, rho):

        try:
            from skimage.restoration import denoise_tv_bregman
        except ImportError:
            print('Error: scikit-image not found. TVD will not work.')
            return x0

        return denoise_tv_bregman(x0, rho / self.penalty)


class sparse(Operator):

    def __init__(self, penalty):
        self.name = 'Sparse'
        super().__init__(penalty)

    def __call__(self, x0, rho):
        lmbda = float(self.penalty) / rho
        return (x0 - lmbda) * (x0 >= lmbda) + (x0 + lmbda) * (x0 <= -lmbda)


class nonneg(Operator):

    def __init__(self):
        self.name = 'Non-negative'
        super().__init__(None)

    def __call__(self, x0, rho):
        return np.maximum(x0, 0)


class linsys(Operator):

    def __init__(self, P, q):
        self.name = 'Quadratic'
        self.P = P
        self.q = q
        super().__init__(None)

    def __call__(self, x0, rho):
        return np.linalg.solve(rho * np.eye(self.q.size) + self.P,
                               rho * x0.copy() + self.q)
