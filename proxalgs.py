"""
Proximal algorithms

A proximal consensus optimization algorithm

"""

# imports
import time
import numpy as np
import operators
import hyperopt

# exports
__all__ = ['minimize', 'add_regularizer']

class Optimizer(object):
    """
    Optimizer class for running proximal algorithms

    Usage
    -----
    To initialize an Optimizer object, pass the name of the desired objective function from the operators
    module (and any additional arguments needed for that function). Then, add any desired regularizers along
    with the necessary hyperparameters for those functions. Finally, use the minimize() function to run
    a proximal consensus algorithm for your problem.

    >> opt = Optimizer('squared_error', x_obs=x_obs)
    >> opt.add_regularizer('sparse', gamma=0.1)
    >> opt.add_regularizer('nucnorm', gamma=0.5)
    >> x_hat = opt.minimize(x_init)

    Notes
    -----
    TODO: Add citations
    TODO: Add a demo notebook

    """

    def __init__(self, objfun, **kwargs):

        self.objectives = list()
        self.add_regularizer(objfun, **kwargs)
        self.converged = False

    def __str__(self):
        return "foobaz"

    def __repr__(self):
        return "foobaz"

    def add_regularizer(self, proxfun, **kwargs):

        def wrapper(theta, rho):
            return getattr(operators, proxfun)(theta.copy(), float(rho), **kwargs)

        self.objectives.append(wrapper)

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

    def hyperopt(self, regularizers, validation_loss, theta_init, num_runs):
        """
        Learn hyperparameters

        Parameters
        ----------
        regularizers : dict
            The set of regularizers to search over. Each key in the dictionary needs to be the name of a
            corresponding function in the operators module, and the value associated with that key is a tuple
            containing the bounds for the search space on a log scale. E.g. (-3,0) will search the space from
            for the regularizer from 0.001 to 1.

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

        # make sure things are initialized correctly
        for proxfun in regularizers:
            assert getattr(operators, proxfun, None) is not None, "Could not find function " + proxfun + "() in operators.py"
            assert len(regularizers[proxfun]) == 2, "Each key in regularizers must be associated with a length 2 tuple"

        # define the meta-objective over the hyperparameters
        def metaobjective(gammas):

            # clear previous regularizers
            self.clear_regularizers()

            # add regularizers with the given hyperparameters
            map(lambda v: self.add_regularizer(v[0], gamma=v[1]), zip(regularizers.keys(), gammas))

            # run the minimizer
            x_hat = self.minimize(theta_init, num_iter=100, disp=2)

            # test on the validation set
            loss = validation_loss(x_hat)

            return {
                'loss': loss,
                'runtime': self.results['runtimes'][-1],
                'primal_residual': self.results['residuals']['primal'],
                'dual_residual': self.results['residuals']['dual'],
                'num_iter': self.results['numiter']
            }

        # build the search space consisting of loguniform ranges of the given values in the regularizers dictionary
        searchspace = map(lambda v: hyperopt.hp.loguniform(v[0], v[1][0], v[1][1]), regularizers.items())

        # store results in hyperopt trials object
        self.hyperopt_trials = hyperopt.Trials()

        # search over different hyperparameters
        gamma_opt = hyperopt.fmin(metaobjective,
                   space=searchspace,
                   algo=hyperopt.tpe.suggest,
                   max_evals=num_runs,
                   trials=self.hyperopt_trials)

        return gamma_opt