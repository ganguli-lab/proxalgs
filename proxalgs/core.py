"""
Proximal algorithms

A proximal consensus optimization algorithm

"""

# imports
import time
import numpy as np
import operators
import hyperopt
import types

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
                def wrapper(theta, rho):
                    return getattr(operators, proxfun)(theta.copy(), float(rho), **kwargs)

                self.objectives.append(wrapper)

            except AttributeError as e:
                print(str(e) + '\n' + 'Could not find the function ' + proxfun + ' in the operators module!')

        # if proxfun is a function, add it as its own proximal operator
        elif isinstance(proxfun, types.FunctionType):
            def wrapper(theta, rho):
                return proxfun(theta.copy(), float(rho), **kwargs)

            self.objectives.append(wrapper)

        # type of proxfun must be a string or a function
        else:
            raise TypeError('The argument "proxfun" must be a string or a function. See the documentation for more details.')


    def clear_regularizers(self):
        """
        Clear any added regularizers (only retains the first objective)

        """
        self.objectives = [self.objectives[0]]


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

        tol : float, optional
            residual tolerance for assessing convergence. if both the primal and dual residuals are less
            than this value, then the algorithm has converged

        """

        # default options / parameter values
        opt = {'rho_init': 10, 'tau_inc': 2, 'tau_dec': 2, 'tol': 1e-3}
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

        # udpate each iteration
        if disp > 1:
            print('-------------------------------------------------------------------------')
            print('|  ELAPSED TIME (s) \t|  PRIMAL RESIDUAL \t|  DUAL RESIDUAL \t|')
            print('-------------------------------------------------------------------------')

        # run ADMM iterations
        self.converged = False
        for k in range(num_iter):

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
            rk = float(np.sum([np.linalg.norm(x - mu[-1]) for x in primals]))
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

            # display
            if disp == 1:
                print('Iteration %i of %i' % (k + 1, num_iter))

            elif disp > 1:
                print('| %10.4f \t\t| %16.8f \t| %16.8f \t|' % (runtimes[-1], rk, sk))

            # call the callback function
            if callback is not None:
                results = {'residuals': resid, 'rho': rho, 'duals': duals, 'runtimes': runtimes, 'primals': primals}
                callback(mu[-1].reshape(orig_shape), results)

            # check for convergence
            if (rk <= opt['tol']) & (sk <= opt['tol']):
                self.converged = True
                break

        # clean up
        if disp > 1:
            print('-------------------------------------------------------------------------\n')

        if self.converged and disp > 0:
            print('Converged after %i iterations!' % (k+1))

        self.results = {'residuals': resid, 'rho': rho, 'duals': duals,
                        'runtimes': runtimes, 'primals': primals, 'numiter': k+1}
        self.theta = mu[-1].reshape(orig_shape)
        return self.theta


    def hyperopt(self, regularizers, validation_loss, theta_init, num_runs, num_iter=50):
        """
        Learn hyperparameters

        Parameters
        ----------
        regularizers : list
            A list of tuples. Each tuple contains four items. The first is a string (the name of the regularizer,
            which can be anything you want). The second is EITHER a string and the name of a function in the operators
            module which has three arguments (x0, rho, gamma) OR a custom function that accepts three arguments
            (x0, rho, and gamma) and applies a proximal operator to the point x0. The final two parameters are floats
            which define the bounds of the log-transformed search space for the regularization parameters (e.g bounds
            of -3 and 0 would correspond to searching values from 0.001 to 1)

        validation_loss : function
            A callback function that takes a single argument, a value for the parameters, and evaluates the error or
            loss on some held out data.

        theta_init : array_like
            An array corresponding to the initial parameter values for optimization

        num_runs : int
            The number of different hyperparameter combinations to search through

        Returns
        -------
        gamma_opt : dict
            a dictionary containing the optimal hyperparameters found. Each key is a regularizer with a
            corresponding function in the operators module, and each value is the learned hyperparameter value

        trials : dict
            Results object from hyperopt.fmin()

        """

        # verify inputs are the right type
        for (name, proxfun, lb, ub) in regularizers:

            # name must be a string
            assert isinstance(name, str), "Name must be a string"

            # proxfun must exist in operators if given as a string
            if isinstance(proxfun, str):
                assert getattr(operators, proxfun, None) is not None, "Could not find " + proxfun + "() in operators.py"

            # lower and upper bounds must be numbers
            assert isinstance(lb, (int, float)), "Lower bound must be a number"
            assert isinstance(ub, (int, float)), "Upper bound must be a number"

        # define the meta-objective over the hyperparameters
        def metaobjective(gammas):

            # clear previous regularizers
            self.clear_regularizers()

            # concatenate regularizers with gamma values
            regs = [(reg[0][1], reg[1]) for reg in zip(regularizers, gammas)]

            # add regularizers with the given hyperparameters (given names of functions in operators.py)
            map(lambda x: self.add_regularizer(x[0], gamma=x[1]),
                filter(lambda v: isinstance(v[0], str), regs))

            # add regularizers with the given hyperparameters (given custom functions)
            map(lambda x: self.add_custom_regularizer(x[0], gamma=x[1]),
                filter(lambda v: not isinstance(v[0], str), regs))

            # run the minimizer
            x_hat = self.minimize(theta_init, num_iter=num_iter, disp=2)

            # test on the validation set
            loss = validation_loss(x_hat)

            return {
                'loss': loss,
                'runtime': self.results['runtimes'][-1],
                'primal_residual': self.results['residuals']['primal'][-1],
                'dual_residual': self.results['residuals']['dual'][-1],
                'num_iter': self.results['numiter']
            }

        # build the search space consisting of loguniform ranges of the given values in the regularizers dictionary
        searchspace = map(lambda x: hyperopt.hp.loguniform(x[0], x[2], x[3]), regularizers)

        # store results in hyperopt trials object
        self.hyperopt_trials = hyperopt.Trials()

        # search over different hyperparameters
        gamma_opt = hyperopt.fmin(metaobjective,
                   space=searchspace,
                   algo=hyperopt.tpe.suggest,
                   max_evals=num_runs,
                   trials=self.hyperopt_trials)

        return gamma_opt