"""
Proximal algorithms

A proximal consensus optimization algorithm

"""

# imports
import time
import numpy as np
import operators

# exports
__all__ = ['minimize', 'add_regularizer']

class Optimizer(object):

    def __init__(self, objfun, **kwargs):

        self.objectives = list()
        self.add_regularizer(objfun, **kwargs)

    def __str__(self):
        return "foobaz"

    def __repr__(self):
        return "foobaz"

    def add_regularizer(self, proxfun, **kwargs):

        def wrapper(theta, rho):
            return getattr(operators, proxfun)(theta.copy(), float(rho), **kwargs)

        self.objectives.append(wrapper)

    def minimize(self, theta_init, num_iter=20, callback=None, **kwargs):
        """
        Minimize a list of objectives using a proximal consensus algorithm

        Parameters
        ----------
        objectives : list
            List of objectives, each objective is a function that computes a proximal update.

        theta_init : ndarray
            Initial parameter vector (numpy array)

        num_iter : int, optional
            number of iterations to run (default: 20)

        callback : function, optional
            a function that gets called on each iteration with the following arguments: the current parameter
            value (ndarray), and a dictionary that contains a information about the status of the algorithm

        Returns
        -------
        theta : ndarray
            The parameters found after running the optimization procedure

        res : dict
            A dictionary containing results and other information about convergence of the algorithm

        Other Parameters
        ----------------
        rho_init : int, optional
            initial value of the momentum term, larger values take smaller steps (default: 10)

        tau_inc : int, optional
            increment parameter for the momentum scheduler (default: 2)

        tau_dec : int, optional
            decrement parameter for the momentum scheduler (default: 2)

        """

        # default options / parameter values
        opt = {'rho_init': 10, 'tau_inc': 2, 'tau_dec': 2}
        opt.update(kwargs)

        # get list of objectives for this parameter
        num_obj = len(self.objectives)
        assert num_obj >= 1, "There must be at least one objective!"

        # initialize lists of primal and dual variable copies, one for each objective
        orig_shape = theta_init.shape
        theta = [theta_init.flatten() for _ in range(num_obj)]
        duals = [np.zeros(theta_init.size) for _ in range(num_obj)]
        mu = np.mean(theta, axis=0).ravel()

        # store primal and dual residuals
        resid = dict()
        resid['primal'] = np.zeros((num_iter, num_obj))
        resid['dual'] = np.zeros(num_iter)

        # penalty parameter scheduling (see sect. 3.4.1 of the Boyd and Parikh ADMM paper)
        rho = np.zeros(num_iter + 1)
        rho[0] = opt['rho_init']

        # store runtimes of each iteration
        runtimes = list()
        tstart = time.time()

        for k in range(num_iter):

            # iter
            print('[Iteration %i of %i]' % (k + 1, num_iter))

            # update each variable copy by taking a proximal step via each objective (TODO: in parallel?)
            theta = [obj((mu - duals[idx]).reshape(orig_shape), rho[k]).ravel() for idx, obj in enumerate(self.objectives)]

            # average local variables
            mu_prev = mu.copy()
            mu = np.mean(theta, axis=0).copy()

            # dual update
            for objective_index, theta_idx in enumerate(theta):
                duals[objective_index] += theta_idx - mu

            # compute primal and dual residuals
            resid['primal'][k, :] = np.linalg.norm(np.vstack(theta) - mu, axis=1)
            resid['dual'][k] = np.sqrt(num_obj) * rho[k] * np.linalg.norm(mu - mu_prev)

            # store runtime
            runtimes.append(time.time() - tstart)

            # update penalty parameter according to primal and dual residuals
            rk = np.linalg.norm(resid['primal'][k,:])
            sk = resid['dual'][k]
            if rk > opt['rho_init'] * sk:
                rho[k + 1] = opt['tau_inc'] * rho[k]
            elif sk > opt['rho_init'] * rk:
                rho[k + 1] = rho[k] / opt['tau_dec']
            else:
                rho[k + 1] = rho[k]

            # call the callback function
            if callback is not None:
                res = {'resid': resid, 'rho': rho, 'duals': duals, 'runtimes': runtimes, 'primals': theta}
                callback(mu.reshape(orig_shape), res)

        self.res = {'resid': resid, 'rho': rho, 'duals': duals, 'runtimes': runtimes, 'primals': theta}
        self.theta = mu.reshape(orig_shape)
        return self.theta
