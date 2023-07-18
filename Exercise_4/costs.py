# -*- coding: utf-8 -*-
"""A function to compute the cost."""

import numpy as np

def compute_mse(e):
    """Calculate the mse for vector e.
    Args:
        e: numpy array of shape (N,), N is the number of samples.

    Returns:
        scalar

    >>> compute_mse(np.array([1.5, -.5]))
    0.625
    """
    return 1 / 2 * np.mean(e**2)


def compute_loss(y, tx, w):
    """Calculate the mse loss.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        w: numpy array of shape(D,)

    Returns:
        Scalar

    >>> compute_loss(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), np.array([3., 2.1]))
    47.96262500000001
    """
    e = y - tx.dot(w)
    return compute_mse(e)
