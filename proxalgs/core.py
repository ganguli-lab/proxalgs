"""
Proximal algorithms

A proximal consensus optimization algorithm

"""

# imports
import time
import numpy as np
from . import operators
from collections import defaultdict
import tableprint
from toolz import last, valmap

# exports
__all__ = ['Optimizer']


class Optimizer(object):
    """
    Optimizer class for running proximal algorithms

    Examples
    --------
    To initialize an Optimizer object, pass the name of the desired objective function from the operators
    module (and any additional arguments needed for that function). Then, add any desired regularizers along
    with the necessary hyperparameters for those functions. Finally, use the minimize() function to run
    a proximal consensus algorithm for your problem.

    >>> opt = Optimizer('squared_error', x_obs=x_obs)
    >>> opt.add_regularizer('sparse', gamma=0.1)
    >>> opt.add_regularizer('nucnorm', gamma=0.5)
    >>> x_hat = opt.minimize(x_init)

    Notes
    -----
    - TODO: Add citations
    - TODO: Add a demo notebook

    """

    def __init__(self, objfun, **kwargs):

        self.objectives = list()
        self.add_regularizer(objfun, **kwargs)
        self.converged = False
        self.theta = None

    def __str__(self):
        return "Optimizer object with %i objectives." % len(self.objectives)

    def add_regularizer(self, proxfun, **kwargs):
        """
        Add a regularizer from the operators module to the list of objectives

        Parameters
        ----------
        proxfun : string or function
            If a string, then it must be the name of a corresponding function in the `operators` module.
            If a function, then it must apply a proximal update given an initial point x0, momentum parameter
            rho, and optional arguments given in `**kwargs`.

        \\*\\*kwargs : keyword arguments
            Any optional arguments required for the given function

        """

        # if proxfun is a string, grab the corresponding function from operators.py
        if isinstance(proxfun, str):
            try:
                self.objectives.append(lambda theta, rho: getattr(operators, proxfun)(theta.copy(), float(rho), **kwargs))

            except AttributeError as e:
                print(str(e) + '\n' + 'Could not find the function ' + proxfun + ' in the operators module!')

        # if proxfun is a function, add it as its own proximal operator
        elif hasattr(proxfun, '__call__'):
            self.objectives.append(lambda theta, rho: proxfun(theta.copy(), float(rho)))

        # type of proxfun must be a string or a function
        else:
            raise TypeError('The argument "proxfun" must be a string or a function!')

    def set_regularizers(self, regularizers, clear=True):
        """
        Adds a set of regularizers

        Parameters
        ----------
        regularizers : dict
            Each key is the name of a corresponding proximal operator, and the
            value associated with that key is a set of keyword arguments

        clear : boolean, optional
            Whether or not to clear the existing regularizers. (Default: True)

        """

        # clear existing operators
        if clear:
            self.clear()

        # add new regularizers
        list([self.add_regularizer(proxfun, **regularizers[proxfun])
              for proxfun in regularizers.keys()])

    def clear(self):
        """
        Clear any added regularizers (only retains the first objective)

        """
        self.objectives = [self.objectives[0]]

    def minimize(self, theta_init, max_iter=50, callback=None, disp=0, mu=10.0, tau_inc=2.0, tau_dec=2.0, tol=1e-3):
        """
        Minimize a list of objectives using a proximal consensus algorithm

        Parameters
        ----------
        theta_init : ndarray
            Initial parameter vector (numpy array)

        max_iter : int, optional
            Maximum number of iterations to run (default: 50)

        callback : function, optional
            a function that gets called on each iteration with the following arguments: the current parameter
            value (ndarray), and a dictionary that contains a information about the status of the algorithm

        disp : integer, optional
            determines how much information to display when running. Ranges from 0 (nothing) to 3 (lots of information)

        Returns
        -------
        theta : ndarray
            The parameters found after running the optimization procedure

        res : dict
            A dictionary containing results and other information about convergence of the algorithm

        Other Parameters
        ----------------
        mu : int, optional
            initial value of the momentum term, larger values take smaller steps (default: 10)

        tau_inc : int, optional
            increment parameter for the momentum scheduler (default: 2)

        tau_dec : int, optional
            decrement parameter for the momentum scheduler (default: 2)

        tol : float, optional
            residual tolerance for assessing convergence. if both the primal and dual residuals are less
            than this value, then the algorithm has converged

        """

        # get list of objectives for this parameter
        num_obj = len(self.objectives)
        assert num_obj >= 1, "There must be at least one objective!"

        # initialize lists of primal and dual variable copies, one for each objective
        orig_shape = theta_init.shape
        primals = [theta_init.flatten() for _ in range(num_obj)]
        duals = [np.zeros(theta_init.size) for _ in range(num_obj)]
        theta_avg = np.mean(primals, axis=0).ravel()

        # penalty parameter
        rho = mu

        # store cumulative runtimes of each iteration, starting now
        tstart = time.time()

        # clear metadata
        self.metadata = defaultdict(list)

        # run ADMM iterations
        self.converged = False
        for cur_iter in range(max_iter):

            # store the parameters from the previous iteration
            theta_prev = theta_avg

            # update each primal variable copy by taking a proximal step via each objective
            for varidx, dual in enumerate(duals):
                primals[varidx] = self.objectives[varidx]((theta_prev - dual).reshape(orig_shape), rho).ravel()

            # average primal copies
            theta_avg = np.mean(primals, axis=0)

            # update the dual variables (after primal update has finished)
            for varidx, primal in enumerate(primals):
                duals[varidx] += primal - theta_avg

            # compute primal and dual residuals
            primal_resid = float(np.sum([np.linalg.norm(primal - theta_avg) for primal in primals]))
            dual_resid = num_obj * rho ** 2 * np.linalg.norm(theta_avg - theta_prev)

            # update penalty parameter according to primal and dual residuals
            # (see sect. 3.4.1 of the Boyd and Parikh ADMM paper)
            if primal_resid > mu * dual_resid:
                rho *= float(tau_inc)
            elif dual_resid > mu * primal_resid:
                rho /= float(tau_dec)

            # update metadata for this iteration
            self.metadata['Primal resid'].append(primal_resid)
            self.metadata['Dual resid'].append(dual_resid)
            self.metadata['Time (s)'].append(time.time() - tstart)
            self.metadata['Momentum'].append(rho)

            # call the callback function with the current parameters and metadata from the last iteration
            if callback is not None:
                callback(theta_avg.reshape(orig_shape), self.metadata.tail(1).irow(0).to_dict())

            # update the display
            self.update_display(cur_iter + 1, disp)

            # check for convergence
            if (primal_resid <= tol) & (dual_resid <= tol):
                self.converged = True
                break

        # clean up display
        self.update_display(-1, disp)

        # store and return final parameters
        self.theta = theta_avg.reshape(orig_shape)
        return self.theta

    def update_display(self, iteration, disp_level, col_width=12):
        """
        Prints information about the optimization procedure to standard output

        Parameters
        ----------
        iteration : int
            The current iteration. Must either a positive integer or -1, which indicates the end of the algorithm

        disp_level : int
            An integer which controls how much information to display, ranging from 0 (nothing) to 3 (lots of stuff)

        col_width : int
            The width of each column in the data table, used if disp_level > 1

        """

        # exit and print nothing if disp_level is zero
        if disp_level == 0:
            return

        else:

            # simple update, no table
            if disp_level == 1 and iteration >= 0:
                print('[Iteration %i]' % iteration)

            # fancy table updates
            if disp_level > 1:

                # get the metadata from this iteration
                data = valmap(last, self.metadata)

                # choose what keys to use
                keys = ['Time (s)', 'Primal resid', 'Dual resid', 'Momentum']

                # initial update. print out table headers
                if iteration == 1:
                    hr = tableprint.hr(len(keys), column_width=col_width)
                    print('\n' + hr)
                    print(tableprint.header(keys, column_width=col_width))
                    print(hr)

                # print data
                tabledata = map(float, [np.array(data[key]) for key in keys])
                print(tableprint.row(tabledata, column_width=col_width, precision='4g'))

                if iteration == -1:
                    print(tableprint.hr(len(keys), column_width=col_width) + '\n')

            # print convergence statement
            if iteration == -1 and self.converged:
                print('Converged after %i iterations!' % len(self.metadata.values()[0]))
