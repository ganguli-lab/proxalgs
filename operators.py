"""
Proximal operators

Evaluates proximal operators

Notes
-----
evaluates expressions of the form:

.. math:: \mathrm{prox}_{f,rho} (v) = \mathrm{argmin}_x ( f(x) + (rho / 2) ||x-v||_2^2 )

"""

def sfo(v, rho, optimizer, num_steps=5):
    """
    Proximal operator for an arbitrary function minimized via the Sum-of-Functions optimizer (SFO)

    Notes
    -----
    SFO is a function optimizer for the case where the target function breaks into a sum over minibatches, or a sum
    over contributing functions. It is described in more detail in [1]_.

    Parameters
    ----------
    v : ndarray
        The starting or initial point used in the proximal update step
    rho : float
        Momentum parameter for the proximal step (larger value -> stays closer to v)
    optimizer : SFO instance
        Instance of the SFO object in `SFO_admm.py`
    num_steps : int, optional
        Number of SFO steps to take

    Returns
    -------
    theta : ndarray
        The parameter vector found after running `num_steps` iterations of the SFO optimizer

    References
    ----------
    .. [1] Jascha Sohl-Dickstein, Ben Poole, and Surya Ganguli. Fast large-scale optimization by unifying stochastic
        gradient and quasi-Newton methods. International Conference on Machine Learning (2014). `arXiv preprint
        arXiv:1311.2115 (2013) <http://arxiv.org/abs/1311.2115>`_.

    """

    # set the current parameter value of SFO to the given value
    optimizer.set_theta(v, float(rho))

    # set the previous ADMM location as the flattened paramter array
    optimizer.theta_admm_prev = optimizer.theta_original_to_flat(v)

    # run the optimizer for n steps
    return optimizer.optimize(num_steps=num_steps)

def poissreg(v, rho, X, y):
    """
    Proximal operator for Poisson regression

    Computes the proximal operator of the negative log-likelihood loss assumping a Poisson noise distribution.

    Parameters
    ----------
    v : ndarray
        The starting or initial point used in the proximal update step
    rho : float
        Momentum parameter for the proximal step (larger value -> stays closer to v)
    X : (N, k) ndarray
        A design matrix consisting of N examples of k-dimensional features (or input).
    y : (N,) ndarray
        A vector containing the responses (outupt) to the N features given in X.

    Returns
    -------
    theta : ndarray
        The parameter vector found after running the proximal update step


    """

    # objective and gradient
    N = float(X.shape[0])
    f = lambda w: np.mean(np.exp(X.dot(w)) - y * X.dot(w))
    df = lambda w: (X.T.dot(np.exp(X.dot(w))) - X.T.dot(y)) / N

    # minimize via BFGS
    return bfgs(v, rho, f, df)

def bfgs(v, rho, f, fgrad):
    """
    Proximal operator for minimizing an arbitrary function using BFGS

    Uses the BFGS algorithm to find the proximal update for an arbitrary function, `f`, whose gradient is known.

    Parameters
    ----------
    v : ndarray
        The starting or initial point used in the proximal update step
    rho : float
        Momentum parameter for the proximal step (larger value -> stays closer to v)
    f : function
        The function to use when applying the proximal operator. Must take as input a parameter vector (ndarray) and
        return a real number (floating point value)
    df : function
        A function that computes the gradient of `f` with respect to the parameters. Must take as input a parameter
        vector (ndarray) and returns another ndarray of the same size.

    Returns
    -------
    theta : ndarray
        The parameter vector found after running the proximal update step

    """

    # specify the objective function and gradient for the proximal operator
    g = lambda x: f(x) + (rho / 2) * np.sum((x.reshape(v.shape) - v) ** 2)
    dg = lambda x: fgrad(x) + rho * (x.reshape(v.shape) - v)

    # minimize via BFGS
    return opt.fmin_bfgs(g, v, dg, disp=False)

def smooth_l2(v, rho, gamma):
    """
    Proximal operator for a smoothing function enforced as an l2-penalty on pairwise differences in a vector

    Parameters
    ----------
    v : ndarray
        The starting or initial point used in the proximal update step
    rho : float
        Momentum parameter for the proximal step (larger value -> stays closer to v)
    gamma : float
        A constant that weights how strongly to enforce the constraint

    Returns
    -------
    theta : ndarray
        The parameter vector found after running the proximal update step

    """
    # objective and gradient
    N = float(v.size)
    f = lambda w: 0.5 * gamma * np.sum(np.diff(w) ** 2)
    df = lambda w: gamma * (np.append(0, np.diff(w)) - np.append(np.diff(w), 0))

    # minimize via BFGS
    return bfgs(v, rho, f, df)

def nucnorm(v, rho, gamma):
    """
    Proximal operator for the nuclear norm (sum of the singular values of a matrix)

    Parameters
    ----------
    v : ndarray
        The starting or initial point used in the proximal update step
    rho : float
        Momentum parameter for the proximal step (larger value -> stays closer to v)
    gamma : float
        A constant that weights how strongly to enforce the constraint

    Returns
    -------
    theta : ndarray
        The parameter vector found after running the proximal update step

    """

    # compute SVD
    u, s, v = np.linalg.svd(v, full_matrices=False)

    # soft threshold the singular values
    sthr = np.maximum(s-(gamma/float(rho)),0)

    # reconstruct
    return (u.dot(np.diag(sthr)).dot(v))

def fronorm(v, rho, v_star):
    """
    Proximal operator for the pairwise difference between two matrices (Frobenius norm)

    Parameters
    ----------
    v : ndarray
        The starting or initial point used in the proximal update step
    rho : float
        Momentum parameter for the proximal step (larger value -> stays closer to v)
    v_star : ndarray
        The true matrix that we want to approximate. The error between the parameters and this matrix is minimized.

    Returns
    -------
    theta : ndarray
        The parameter vector found after running the proximal update step

    """
    return (v.ravel() + v_star.ravel() / rho).reshape(v.shape) / (1 + 1/rho)

def tvd(v, rho, gamma):
    """
    Proximal operator for the total variation denoising penalty

    Requires scikit-image be installed

    Parameters
    ----------
    v : ndarray
        The starting or initial point used in the proximal update step
    rho : float
        Momentum parameter for the proximal step (larger value -> stays closer to v)
    gamma : float
        A constant that weights how strongly to enforce the constraint

    Returns
    -------
    theta : ndarray
        The parameter vector found after running the proximal update step

    Raises
    ------
    ImportError
        If scikit-image fails to be imported

    """

    try:
        from skimage.restoration import denoise_tv_bregman
        return denoise_tv_bregman(v, rho / gamma)
    except ImportError:
        print("Must have scikit-image installed.")

def sparse(v, rho, gamma):
    """
    Proximal operator for the l1 norm (induces sparsity)

    Parameters
    ----------
    v : ndarray
        The starting or initial point used in the proximal update step
    rho : float
        Momentum parameter for the proximal step (larger value -> stays closer to v)
    gamma : float
        A constant that weights how strongly to enforce the constraint

    Returns
    -------
    theta : ndarray
        The parameter vector found after running the proximal update step

    """

    return (v * gamma - 1./rho) * (v * gamma >= 1./rho) \
           + (v * gamma + 1./rho) * (v * gamma <= -1./rho)

def nonneg(v, rho):
    """
    Proximal operator for enforcing non-negativity (indicator function over the set x >= 0)

    Parameters
    ----------
    v : ndarray
        The starting or initial point used in the proximal update step
    rho : float
        Unused parameter

    Returns
    -------
    theta : ndarray
        The parameter vector found after running the proximal update step

    """
    return np.maximum(v, 0)

def linsys(v, rho, P, q):
    """
    Proximal operator for the linear approximation Ax = b

    Minimizes the function:

    .. math:: f(x) = (1/2)||Ax-b||_2^2 = (1/2)x^TA^TAx - (b^TA)x + b^Tb

    Parameters
    ----------
    v : ndarray
        The starting or initial point used in the proximal update step
    rho : float
        Momentum parameter for the proximal step (larger value -> stays closer to v)
    P : ndarray
        The symmetric matrix A^TA, where we are trying to approximate Ax=b
    q : ndarray
        The vector A^Tb, where we are trying to approximate Ax=b

    Returns
    -------
    theta : ndarray
        The parameter vector found after running the proximal update step

    """
    return np.linalg.solve(rho * np.eye(q.size) + P, rho * v.copy() + q)