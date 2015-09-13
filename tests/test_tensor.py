"""
Tests for the tensor module

"""

from proxalgs.tensor import Tensor
import numpy as np


def test_unfolding():

    t = Tensor(np.random.randn(3, 4, 5))

    t0 = t.unfold(0)
    t1 = t.unfold(1)
    t2 = t.unfold(2)

    assert t0.shape == (3, 20)
    assert np.allclose(t, t0.fold())

    assert t1.shape == (4, 15)
    assert np.allclose(t, t1.fold())

    assert t2.shape == (5, 12)
    assert np.allclose(t, t2.fold())


def test_norms():

    # test Frobenius norm
    t = Tensor(np.arange(12).reshape(2, 3, 2))
    assert np.allclose(np.linalg.norm(np.arange(12)), t.norm)
    assert np.allclose(np.linalg.norm(np.arange(12)), t.unfold(2).norm)

    # test nuclear norm
    t = Tensor(np.eye(5, 10).reshape(5, 2, 5))
    assert np.allclose(t.unfold(0).nucnorm, 5.)
