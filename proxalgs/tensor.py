"""
Tensors
"""
import numpy as np
from .operators import squared_error

__all__ = ['Tensor', 'UnfoldedTensor', 'susvd']


class Tensor(np.ndarray):

    def __new__(cls, arr):
        obj = np.asarray(arr).view(cls)
        return obj

    def unfold(self, ax):
        assert ax in range(self.ndim), "ax less than ndim"
        orig_shape = self.shape
        rolled_axes = pullax(list(range(self.ndim)), ax)
        unfolded = self.transpose(rolled_axes).reshape(orig_shape[ax], -1)
        return UnfoldedTensor(unfolded, ax, orig_shape)

    @property
    def norm(self):
        return np.linalg.norm(self.ravel(), ord=2)


class UnfoldedTensor(np.ndarray):

    def __new__(cls, arr, axis, shape):
        obj = np.asarray(arr).view(cls)
        obj.orig_shape = shape
        obj.axis = axis
        return obj

    @property
    def norm(self):
        return np.linalg.norm(self, ord='fro')

    @property
    def nucnorm(self):
        return np.sum(np.linalg.svd(self, compute_uv=False))

    def svt(self, threshold):
        u, s, v = np.linalg.svd(self, full_matrices=False)
        sthr = np.diag(np.maximum(s - threshold, 0))
        return UnfoldedTensor(u.dot(sthr).dot(v), self.axis, self.orig_shape)

    def __array_finalize__(self, obj):

        if obj is None:
            print('unfolded tensor None')
            return

        self.orig_shape = getattr(obj, 'orig_shape', None)
        self.axis = getattr(obj, 'axis', None)

    def fold(self):
        rolled_axes = pullax(list(range(len(self.orig_shape))), self.axis)
        folded = self.reshape(tuple(self.orig_shape[i] for i in rolled_axes))
        return Tensor(folded.transpose(np.argsort(rolled_axes)))


def pullax(values, idx):
    values.insert(0, values.pop(idx))
    return tuple(values)


def susvd(x, x_obs, rho, penalties):
    """
    Sequential unfolding SVD

    Parameters
    ----------
    x : Tensor

    x_obs : array_like

    rho : float

    penalties : array_like
        penalty for each unfolding of the input tensor
    """

    assert type(x) == Tensor, "Input array must be a Tensor"

    while True:

        # proximal operator for the Fro. norm
        x = squared_error(x, rho, x_obs)

        # sequential singular value thresholding
        for ix, penalty in enumerate(penalties):
            x = x.unfold(ix).svt(penalty / rho).fold()

        yield x
