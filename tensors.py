"""
Proximal operators on unfolded tensors

"""

from sktensor import dtensor
import numpy as np

# exports
__all__ = ['nucnorm']

def nucnorm(x0, rho, gamma, mode):
    """
    Proximal operator for the nuclear norm (sum of the singular values of a matrix)

    Parameters
    ----------
    x0 : dtensor
        The starting or initial point used in the proximal update step

    rho : float
        Momentum parameter for the proximal step (larger value -> stays closer to x0)

    gamma : float
        A constant that weights how strongly to enforce the constraint

    mode : integer
        The index to unfold against

    Returns
    -------
    theta : ndarray
        The parameter vector found after running the proximal update step

    """

    # compute SVD
    u, s, v = np.linalg.svd(dtensor(x0).unfold(mode), full_matrices=False)

    # soft threshold the singular values
    sthr = np.maximum(s - (gamma / float(rho)), 0)

    # reconstruct
    return (u.dot(np.diag(sthr)).dot(v)).fold()
