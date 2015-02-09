"""
Proximal algorithms

A proximal consensus optimization algorithm

"""

# imports
import time
import numpy as np
import operators, tensors

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

    def add_tensor_regularizer(self, proxfun, **kwargs):

        def wrapper(theta, rho):
            return getattr(tensors, proxfun)(theta.copy(), float(rho), **kwargs)

        self.objectives.append(wrapper)

    def minimize(self, theta_init, num_iter=20, callback=None, disp=0, **kwargs):
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

        disp : integer, optional
            determines how much information to display when running. 0 (default): nothing, 2: lots of information

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
        primals = [theta_init.flatten() for _ in range(num_obj)]
        duals = [np.zeros(theta_init.size) for _ in range(num_obj)]
        mu = [np.mean(primals, axis=0).ravel()]

        # store primal and dual residuals
        resid = dict()
        resid['primal'] = list()
        resid['dual'] = list()

        # penalty parameter scheduling (see sect. 3.4.1 of the Boyd and Parikh ADMM paper)
        rho = np.zeros(num_iter + 1)
        rho[0] = opt['rho_init']

        # store cumulative runtimes of each iteration
        runtimes = list()
        tstart = time.time()

        for k in range(num_iter):

            # iter
            if disp > 0:
                print('[Iteration %i of %i]' % (k + 1, num_iter))

            # update each variable copy by taking a proximal step via each objective (TODO: in parallel?)
            for idx, x in enumerate(primals):

                # unpack objective
                obj = self.objectives[idx]

                # evaluate objective (proximity operator) to update primals
                primals[idx] = obj((-duals[idx] + mu[-1]).reshape(orig_shape), rho[k]).ravel()

            # average primal copies
            mu.append(np.mean(primals, axis=0).copy())

            # update the dual variables (after primal update has finished!)
            for idx, x in enumerate(primals):
                duals[idx] += x - mu[-1]

            # compute primal and dual residuals
            rk = np.sum([np.linalg.norm(x - mu[-1]) for x in primals])
            sk = num_obj * rho[k] ** 2 * np.linalg.norm(mu[-1] - mu[-2])
            resid['primal'].append(rk)
            resid['dual'].append(sk)

            # store runtime
            runtimes.append(time.time() - tstart)

            # update penalty parameter according to primal and dual residuals
            if rk > opt['rho_init'] * sk:
                rho[k + 1] = opt['tau_inc'] * rho[k]
            elif sk > opt['rho_init'] * rk:
                rho[k + 1] = rho[k] / opt['tau_dec']
            else:
                rho[k + 1] = rho[k]

            # more to display?
            if disp > 1:
                print('> Elapsed time: %5.4f s' % runtimes[-1])
                print('> Primal residual: %5.4f s' % rk)
                print('> Dual residual: %5.4f s' % sk)

            # call the callback function
            if callback is not None:
                results = {'residuals': resid, 'rho': rho, 'duals': duals, 'runtimes': runtimes, 'primals': primals}
                callback(mu[-1].reshape(orig_shape), results)

        self.results = {'residuals': resid, 'rho': rho, 'duals': duals, 'runtimes': runtimes, 'primals': primals}
        self.theta = mu[-1].reshape(orig_shape)
        return self.theta
